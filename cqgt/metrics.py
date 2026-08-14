"""Evaluation metrics (BRIEF.md Sec 2.6). AUPRC is primary; the rest give a
fuller picture of calibration (Brier, ECE) and ranking quality (Kendall-tau,
NDCG@5) on top of the standard classification metrics. Counterfactual
metrics (Spearman rho, Precision@10) compare predicted vs. ground-truth
edge-level Delta-R attribution (T3).
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                              ndcg_score, roc_auc_score, roc_curve)


def prevalence(y_true):
    return float(np.mean(y_true))


def auprc(y_true, p_hat):
    return float(average_precision_score(y_true, p_hat))


def auroc(y_true, p_hat):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_hat))


def f1_at_youden_j(y_true, p_hat):
    """Threshold at Youden's J = argmax(TPR - FPR) on the ROC curve, then F1
    at that threshold."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, p_hat)
    j = tpr - fpr
    best_thresh = thresholds[np.argmax(j)]
    y_pred = (p_hat >= best_thresh).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def brier(y_true, p_hat):
    return float(brier_score_loss(y_true, p_hat))


def ece(y_true, p_hat, n_bins=10):
    """Expected Calibration Error, n_bins equal-width bins over [0, 1]."""
    y_true, p_hat = np.asarray(y_true, dtype=float), np.asarray(p_hat, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_hat, bin_edges[1:-1], right=True), 0, n_bins - 1)
    total = len(y_true)
    err = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = p_hat[mask].mean()
        acc = y_true[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def kendall_tau_ranking(y_score_true, y_score_pred):
    """Kendall's tau between a continuous ground-truth risk score (e.g.
    per-scenario loss_frac) and the model's predicted score, both over the
    same set of (scenario, node) points."""
    if len(y_score_true) < 2:
        return float("nan")
    tau, _ = kendalltau(y_score_true, y_score_pred)
    return float(tau)


def ndcg_at_5_per_scenario(y_true_grouped, p_hat_grouped):
    """y_true_grouped, p_hat_grouped: lists of same-length 1-D arrays, one
    per scenario (e.g. one per (t, m) draw), each over the N nodes in that
    scenario. Returns the mean NDCG@5 across scenarios with >=1 positive."""
    scores = []
    for y_true, p_hat in zip(y_true_grouped, p_hat_grouped):
        if np.sum(y_true) == 0:
            continue
        k = min(5, len(y_true))
        scores.append(ndcg_score(np.asarray(y_true)[None, :], np.asarray(p_hat)[None, :], k=k))
    return float(np.mean(scores)) if scores else float("nan")


def counterfactual_spearman(delta_r_true, delta_r_pred):
    if len(delta_r_true) < 2:
        return float("nan")
    rho, _ = spearmanr(delta_r_true, delta_r_pred)
    return float(rho)


def counterfactual_precision_at_k(delta_r_true, delta_r_pred, k=10):
    """Fraction of the predicted top-k edges (by |Delta-R|) that are also in
    the ground-truth top-k."""
    k = min(k, len(delta_r_true))
    if k == 0:
        return float("nan")
    true_top = set(np.argsort(-np.abs(np.asarray(delta_r_true)))[:k])
    pred_top = set(np.argsort(-np.abs(np.asarray(delta_r_pred)))[:k])
    return len(true_top & pred_top) / k


def full_report(y_true, p_hat):
    """The standard block of classification/calibration metrics reported in
    T1/T2 for a single model on a single (dataset, seed)."""
    return {
        "prevalence": prevalence(y_true),
        "auprc": auprc(y_true, p_hat),
        "auroc": auroc(y_true, p_hat),
        "f1_youden": f1_at_youden_j(y_true, p_hat),
        "brier": brier(y_true, p_hat),
        "ece": ece(y_true, p_hat),
    }
