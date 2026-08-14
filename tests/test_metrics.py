import numpy as np
import pytest

from cqgt.metrics import (auprc, auroc, brier, counterfactual_precision_at_k,
                           counterfactual_spearman, ece, f1_at_youden_j,
                           kendall_tau_ranking, ndcg_at_5_per_scenario, prevalence)


def test_prevalence():
    assert prevalence(np.array([1, 0, 0, 0])) == 0.25


def test_perfect_classifier_scores_well():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auprc(y, p) == 1.0
    assert auroc(y, p) == 1.0
    assert f1_at_youden_j(y, p) == 1.0
    assert brier(y, p) < 0.05


def test_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    assert ece(y, p, n_bins=10) < 0.05


def test_ece_overconfident_is_large():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99])  # overconfident, half wrong
    assert ece(y, p) > 0.4


def test_kendall_tau_perfect_rank():
    true_score = np.array([1.0, 2.0, 3.0, 4.0])
    pred_score = np.array([1.5, 2.5, 3.5, 4.5])
    assert kendall_tau_ranking(true_score, pred_score) == 1.0


def test_ndcg_at_5_per_scenario():
    y_grouped = [np.array([0, 0, 1, 0, 0])]
    p_grouped = [np.array([0.1, 0.2, 0.9, 0.05, 0.05])]
    assert ndcg_at_5_per_scenario(y_grouped, p_grouped) == 1.0


def test_counterfactual_metrics_perfect_match():
    true_dr = [0.5, 0.3, 0.1, -0.2, 0.05]
    pred_dr = [0.4, 0.35, 0.05, -0.25, 0.02]
    assert counterfactual_spearman(true_dr, true_dr) == pytest.approx(1.0)
    assert counterfactual_precision_at_k(true_dr, true_dr, k=2) == 1.0
    rho = counterfactual_spearman(true_dr, pred_dr)
    assert rho > 0.5
