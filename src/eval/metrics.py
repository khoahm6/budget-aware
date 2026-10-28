"""Offline utility, retraining similarity and fixed loss-based membership attack."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, log_loss,
                             balanced_accuracy_score, roc_curve)


def utility(y, p):
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'macro_f1': f1_score(y, pred, average='macro'),
            'precision_attack': precision_score(y, pred, zero_division=0),
            'recall_attack': recall_score(y, pred, zero_division=0),
            'pr_auc_ap': average_precision_score(y, p), 'fpr': float(fp/(fp+tn)),
            'roc_auc': roc_auc_score(y, p), 'log_loss': log_loss(y, p, labels=[0,1]),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def similarity(p, q):
    a = np.clip(np.stack([1-p, p], 1).astype(float), 1e-7, 1-1e-7)
    b = np.clip(np.stack([1-q, q], 1).astype(float), 1e-7, 1-1e-7)
    midpoint = (a+b)/2
    return {'prediction_mae': float(np.mean(abs(p-q))),
            'disagreement': float(np.mean((p >= .5) != (q >= .5))),
            'kl_to_retrain': float(np.mean(np.sum(a*np.log(a/b), 1))),
            'js_divergence': float(np.mean(np.sum(a*np.log(a/midpoint)+b*np.log(b/midpoint), 1)/2))}


def membership_score(y, p):
    return np.log(np.clip(np.where(y == 1, p, 1-p), 1e-7, 1))


def calibrate_mia(y, p):
    score = membership_score(y, p)
    return [float(np.quantile(score[y == c], .99, method='higher')) for c in [0, 1]]


def matched_indices(member_y, nonmember_y, seed, cap=2000):
    rng = np.random.default_rng(seed)
    mi, ni = [], []
    for c in [0, 1]:
        m, n = np.flatnonzero(member_y == c), np.flatnonzero(nonmember_y == c)
        size = min(len(m), len(n), cap)
        if not size:
            raise ValueError('MIA class matching has no samples')
        mi.extend(rng.permutation(m)[:size]); ni.extend(rng.permutation(n)[:size])
    return np.array(mi), np.array(ni)


def mia(member_y, member_p, nonmember_y, nonmember_p, thresholds):
    m = membership_score(member_y, member_p)
    n = membership_score(nonmember_y, nonmember_p)
    truth = np.r_[np.ones(len(m)), np.zeros(len(n))]
    score = np.r_[m, n]
    decision = np.r_[m > np.array(thresholds)[member_y], n > np.array(thresholds)[nonmember_y]]
    fpr, tpr, _ = roc_curve(truth, score)
    return {'mia_auc': roc_auc_score(truth, score),
            'mia_balanced_accuracy': balanced_accuracy_score(truth, decision),
            'forget_or_retained_member_rate': float(decision[:len(m)].mean()),
            'nonmember_false_positive_rate': float(decision[len(m):].mean()),
            'mia_tpr_at_fpr_0_01': float(tpr[fpr <= .01].max()) if len(n) >= 1000 else None,
            'members': len(m), 'nonmembers': len(n),
            'operating_point': 'strictly_above_original_calibration_classwise_99th_percentile'}
