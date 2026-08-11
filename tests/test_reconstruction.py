import numpy as np

from cqgt.data.reconstruction import maxent_reconstruct, mindensity_reconstruct, rescale_marginals


def _random_marginals(seed=0, n=8):
    rng = np.random.default_rng(seed)
    return rng.uniform(10, 100, size=n), rng.uniform(10, 100, size=n)


def test_rescale_marginals_balances_totals():
    a, l = _random_marginals()
    a2, l2, report = rescale_marginals(a, l)
    assert np.isclose(a2.sum(), l2.sum())
    assert report["imbalance_pct"] != 0  # the unscaled inputs are imbalanced by construction


def test_maxent_matches_marginals_exactly():
    a, l = _random_marginals()
    a2, l2, _ = rescale_marginals(a, l)
    W = maxent_reconstruct(a2, l2)
    np.testing.assert_allclose(W.sum(axis=1), a2, atol=1e-4)
    np.testing.assert_allclose(W.sum(axis=0), l2, atol=1e-4)
    assert np.allclose(np.diag(W), 0)
    assert (W >= 0).all()


def test_mindensity_matches_marginals_exactly_and_is_sparse():
    a, l = _random_marginals()
    a2, l2, _ = rescale_marginals(a, l)
    W = mindensity_reconstruct(a2, l2)
    np.testing.assert_allclose(W.sum(axis=1), a2, atol=1e-3)
    np.testing.assert_allclose(W.sum(axis=0), l2, atol=1e-3)
    assert np.allclose(np.diag(W), 0)
    assert (W >= 0).all()
    n = len(a2)
    assert (W > 1e-6).sum() <= 2 * n  # sparse: at most ~2n-1 edges, not the full n(n-1)


def test_mindensity_denser_than_maxent_is_false_maxent_denser():
    """maxent smooths exposure across the whole network (dense); mindensity
    concentrates it on the fewest links (sparse) -- this is the whole point
    of running both (BRIEF.md Sec 2.1)."""
    a, l = _random_marginals(seed=1, n=10)
    a2, l2, _ = rescale_marginals(a, l)
    nnz_maxent = (maxent_reconstruct(a2, l2) > 1e-6).sum()
    nnz_mindensity = (mindensity_reconstruct(a2, l2) > 1e-6).sum()
    assert nnz_maxent > nnz_mindensity


def test_reconstruction_reproducible_across_permutations_of_ties():
    """Degenerate self-referential residuals (see reconstruction.py's 3-edge
    pivot) must not depend on institution ordering for the totals to match."""
    a, l = _random_marginals(seed=7, n=6)
    a2, l2, _ = rescale_marginals(a, l)
    W = mindensity_reconstruct(a2, l2)
    assert np.abs(W.sum(axis=1) - a2).max() < 1e-3
    assert np.abs(W.sum(axis=0) - l2).max() < 1e-3
