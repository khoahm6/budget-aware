"""Global-start alternating GA/GD and optional retain-only client KD adaptation."""
import copy
import torch
from torch import nn
from src.fl.fedavg_core import batches


def forget_step(model, x, y, anchor, lr, regularization, clip):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    ce = nn.functional.cross_entropy(model(x), y)
    penalty = sum((p - a).square().sum() for p, a in zip(model.parameters(), anchor))
    loss = -ce + regularization * penalty
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite FU loss')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return {'forget_ce': ce.item(), 'loss': loss.item(), 'gradient_norm_before_clip': norm.item()}


def alternating(original, forget, retain, config, seed):
    local = copy.deepcopy(original)
    local.train()
    anchor = [p.detach().clone() for p in local.parameters()]
    logs = []
    for cycle in range(config['cycles']):
        for idx in batches(len(forget.y), config['batch_size'], seed + cycle):
            logs.append({'cycle': cycle, **forget_step(local, forget.x[idx], forget.y[idx], anchor,
                         config['forget_lr'], config['regularization'], config['clip'])})
        optimizer = torch.optim.SGD(local.parameters(), lr=config['retain_lr'])
        for idx in batches(len(retain.y), config['batch_size'], seed + cycle):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(local(retain.x[idx]), retain.y[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite retain loss')
            loss.backward()
            nn.utils.clip_grad_norm_(local.parameters(), config['clip'], error_if_nonfinite=True)
            optimizer.step()
            logs.append({'cycle': cycle, 'retain_ce': loss.item()})
    return local, logs


def kd_loss(student_logits, teacher_logits, labels, temperature, hard_weight):
    soft = nn.functional.kl_div(nn.functional.log_softmax(student_logits / temperature, dim=1),
                               nn.functional.softmax(teacher_logits.detach() / temperature, dim=1),
                               reduction='batchmean') * temperature**2
    return hard_weight * nn.functional.cross_entropy(student_logits, labels) + (1-hard_weight) * soft


def distill(original, teacher, retain, config, seed):
    teacher = copy.deepcopy(teacher).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(original).train()
    optimizer = torch.optim.SGD(student.parameters(), lr=config['kd_lr'])
    losses = []
    for epoch in range(config['kd_epochs']):
        for idx in batches(len(retain.y), config['batch_size'], seed + epoch):
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(retain.x[idx])
            loss = kd_loss(student(retain.x[idx]), teacher_logits, retain.y[idx],
                           config['temperature'], config['hard_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite KD loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return student, losses
