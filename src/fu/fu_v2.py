"""fu_v2.py - bounded gradient-ascent + knowledge-distillation unlearning primitives.

Plain PyTorch, model-agnostic (any nn.Module returning 2-class logits). It does not
touch the existing pilot code; add it next to conpfu_inspired.py and select it by config.

Ideas borrowed (mechanism only, not numbers):
  * l2-ball projection around the starting model      (FedKD-Unlearn, Eq. 19-20)
  * knowledge distillation after ascent               (ConPFU Eq. 5-8 / FedKD-Unlearn Alg. 2)
  * loss cap = loss on unseen data ("stop when forgotten") - our addition
  * canary / trigger forget sets so forgetting is measurable (FedKD-Unlearn uses backdoor ASR)
"""
import copy
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector as _p2v, vector_to_parameters as _v2p


# ----------------------------------------------------------------------------- utils
def flat(model):
    return _p2v(model.parameters()).detach().clone()


@torch.no_grad()
def mean_ce(model, X, y, batch=4096):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), batch):
        tot += F.cross_entropy(model(X[i:i + batch]), y[i:i + batch], reduction="sum").item()
    return tot / len(X)


@torch.no_grad()
def prob_attack(model, X, batch=4096):
    model.eval()
    return torch.cat([F.softmax(model(X[i:i + batch]), -1)[:, 1] for i in range(0, len(X), batch)])


def prob_mae(model_a, model_b, X):
    return (prob_attack(model_a, X) - prob_attack(model_b, X)).abs().mean().item()


def _cycle(n, batch, gen):
    batch = min(batch, n)
    while True:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - batch + 1, batch):
            yield perm[i:i + batch]


@torch.no_grad()
def class_weighted_cap(model, Xcal, ycal, yf):
    """Loss level of *unseen* data, re-weighted to the class mix of the forget set.
    (Client data is heavily class-skewed; a plain calibration mean would be the wrong target.)"""
    model.eval()
    ce = F.cross_entropy(model(Xcal), ycal, reduction="none")
    cap = 0.0
    for c in (0, 1):
        w = (yf == c).float().mean().item()
        if w > 0:
            cap += w * ce[ycal == c].mean().item()
    return cap


def radius_from_reference(theta_start, theta_reference, scale=1.0):
    """Pilot-time calibration: radius = scale * ||theta_reference - theta_start||_2
    (reference = fresh retrain). At deployment there is no retrain; use a proxy such as
    the norm of the target client's last-round update."""
    return scale * (theta_reference - theta_start).norm().item()


def conpfu_temperature(t_base, reduction_f, kappa):
    """ConPFU Eq. 8: T = T_b + F/kappa (higher T = gentler unlearning)."""
    return t_base + reduction_f / kappa


# ----------------------------------------------------------------------------- GA
def ga_projected(model, Xf, yf, Xr, yr, *, radius, loss_cap, steps, lr, lam_r=1.0,
                 batch=256, optimizer="adam", seed=0):
    """Alternating-in-one-loop NegGrad+: ascend on forget CE (capped), descend on retain CE,
    then project back onto the l2 ball ||theta - theta0|| <= radius.

    The cap makes the forget gradient exactly zero once CE_f >= loss_cap, i.e. once the model
    treats forget samples like unseen data. Returns (model_copy, per-step log)."""
    gen = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(model)
    model.train()
    theta0 = flat(model)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(optimizer)
    fi, ri = _cycle(len(Xf), batch, gen), _cycle(len(Xr), batch, gen)
    log = []
    for t in range(steps):
        fb, rb = next(fi), next(ri)
        ce_f = F.cross_entropy(model(Xf[fb]), yf[fb])
        ce_r = F.cross_entropy(model(Xr[rb]), yr[rb])
        loss = -torch.clamp(ce_f, max=loss_cap) + lam_r * ce_r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            d = _p2v(model.parameters()) - theta0
            n = d.norm()
            if n > radius:
                _v2p(theta0 + d * (radius / n), model.parameters())
            log.append({"step": t, "ce_f": ce_f.item(), "ce_r": ce_r.item(),
                        "dist": min(n.item(), float(radius))})
    model.eval()
    return model, log


# ----------------------------------------------------------------------------- KD
def kd_distill(student, teacher, X, y, *, tau, lam_soft, lam_hard, steps, lr,
               batch=256, optimizer="adam", seed=0):
    """loss = lam_soft * tau^2 * KL(teacher || student) + lam_hard * CE(student, y).
    Teacher is frozen (deep-copied). Roles decide the behaviour:
      repair : teacher = original model, student = model after ascent   (FedKD-Unlearn)
      soften : teacher = model after ascent, student = original model   (ConPFU)"""
    gen = torch.Generator().manual_seed(seed)
    student = copy.deepcopy(student)
    teacher = copy.deepcopy(teacher).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    opt = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)(student.parameters(), lr=lr)
    it = _cycle(len(X), batch, gen)
    for _ in range(steps):
        idx = next(it)
        with torch.no_grad():
            zt = teacher(X[idx])
        zs = student(X[idx])
        soft = F.kl_div(F.log_softmax(zs / tau, -1), F.softmax(zt / tau, -1),
                        reduction="batchmean") * tau ** 2
        hard = F.cross_entropy(zs, y[idx])
        loss = lam_soft * soft + lam_hard * hard
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    student.eval()
    return student


# ----------------------------------------------------------------------------- pipeline
def unlearn_v2(model0, Xf, yf, Xr, yr, Xcal, ycal, *, radius, steps, lr,
               kd_role="repair", kd_steps=200, tau=3.0, lam_soft=0.7, lam_hard=0.3,
               kd_lr=1e-3, lam_r=1.0, batch=256, optimizer="adam", cap_scale=1.0, seed=0):
    """Client-side unlearning step. Xcal/ycal = calibration data never used for training or forgetting.
    kd_role in {"none", "repair", "soften"}. Retain data (Xr, yr) is used for KD (F is excluded)."""
    cap = cap_scale * class_weighted_cap(model0, Xcal, ycal, yf)
    m_ga, log = ga_projected(model0, Xf, yf, Xr, yr, radius=radius, loss_cap=cap, steps=steps,
                             lr=lr, lam_r=lam_r, batch=batch, optimizer=optimizer, seed=seed)
    if kd_role == "none":
        out = m_ga
    elif kd_role == "repair":
        out = kd_distill(m_ga, model0, Xr, yr, tau=tau, lam_soft=lam_soft, lam_hard=lam_hard,
                         steps=kd_steps, lr=kd_lr, batch=batch, seed=seed)
    elif kd_role == "soften":
        out = kd_distill(model0, m_ga, Xr, yr, tau=tau, lam_soft=lam_soft, lam_hard=lam_hard,
                         steps=kd_steps, lr=kd_lr, batch=batch, seed=seed)
    else:
        raise ValueError(kd_role)
    return out, {"loss_cap": cap, "ga_log": log, "after_ga": m_ga}


# ----------------------------------------------------------------------------- canaries
def make_label_canary(y, forget_idx, frac, seed):
    """Flip the label of a random `frac` of forget_idx so the model must memorise them.
    Returns (y_new, canary_idx). Apply only to the *original* training data; the retrain
    reference never sees these rows."""
    g = torch.Generator().manual_seed(seed)
    forget_idx = torch.as_tensor(forget_idx)
    k = int(round(frac * len(forget_idx)))
    canary = forget_idx[torch.randperm(len(forget_idx), generator=g)[:k]]
    y_new = y.clone()
    y_new[canary] = 1 - y_new[canary]
    return y_new, canary


def add_trigger(X, feat_idx, value):
    X = X.clone()
    X[:, list(feat_idx)] = value
    return X


def make_trigger_canary(X, y, forget_idx, frac, seed, feat_idx, value, target_label):
    """Stamp a trigger on a random `frac` of forget_idx and set their label to target_label."""
    g = torch.Generator().manual_seed(seed)
    forget_idx = torch.as_tensor(forget_idx)
    k = int(round(frac * len(forget_idx)))
    canary = forget_idx[torch.randperm(len(forget_idx), generator=g)[:k]]
    X_new, y_new = X.clone(), y.clone()
    X_new[canary][:, list(feat_idx)]  # shape check only
    X_new[canary.unsqueeze(1), torch.as_tensor(list(feat_idx)).unsqueeze(0)] = value
    y_new[canary] = target_label
    return X_new, y_new, canary


@torch.no_grad()
def canary_label_acc(model, X, y_flipped):
    """Accuracy w.r.t. the *flipped* label: high = memorised, ~ base rate = forgotten."""
    model.eval()
    return (model(X).argmax(-1) == y_flipped).float().mean().item()


@torch.no_grad()
def asr(model, X_clean, y_clean, feat_idx, value, target_label):
    """Attack success rate of a trigger: fraction of non-target samples sent to target_label."""
    model.eval()
    keep = y_clean != target_label
    Xt = add_trigger(X_clean[keep], feat_idx, value)
    return (model(Xt).argmax(-1) == target_label).float().mean().item()
