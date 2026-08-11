import numpy as np

from cqgt.data.cascade import (aggregate_risk, clear, external_assets_from_marginals,
                                generate_mc_labels_for_panel, ground_truth_delta_r, run_cascade)


def _toy_network(n=4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(1, 10, size=(n, n))
    np.fill_diagonal(W, 0)
    y832 = rng.uniform(500, 1000, size=n)
    m362 = W.sum(axis=1)
    return W, y832, m362


def test_clear_no_shock_fully_pays_when_solvent():
    W, y832, m362 = _toy_network()
    ext = external_assets_from_marginals(y832, m362)
    L = W.sum(axis=0)
    p, net_worth = clear(W, ext)
    # external assets dwarf interbank liabilities here -> full payment
    np.testing.assert_allclose(p, L, atol=1e-6)


def test_clear_conserves_nonnegativity_and_caps_at_liabilities():
    W, y832, m362 = _toy_network()
    ext = external_assets_from_marginals(y832, m362)
    L = W.sum(axis=0)
    ext_shocked = ext * 0.0  # wipe out everyone
    p, net_worth = clear(W, ext_shocked)
    assert (p >= -1e-9).all()
    assert (p <= L + 1e-6).all()


def test_run_cascade_shocked_bank_more_likely_distressed_than_baseline():
    W, y832, m362 = _toy_network()
    y, loss_frac, nw, nw0 = run_cascade(W, y832, m362, shocked_idx=[0], shock_frac=1.0)
    assert loss_frac[0] > 0  # the directly shocked bank must show some loss
    assert y[0] == 1  # a full external-asset wipeout must cross any reasonable threshold


def test_ground_truth_delta_r_returns_one_entry_per_real_edge():
    W, y832, m362 = _toy_network(n=4)
    delta = ground_truth_delta_r(W, y832, m362, shocked_idx=[0], shock_frac=1.0)
    n_edges = int((W > 0).sum())
    assert len(delta) == n_edges
    assert all(isinstance(k, tuple) and len(k) == 2 for k in delta)


def test_aggregate_risk_is_mean_loss_fraction():
    loss_frac = np.array([0.1, 0.2, 0.3, 0.0])
    assert np.isclose(aggregate_risk(loss_frac), loss_frac.mean())


def test_generate_mc_labels_shapes_and_shocked_mask_consistency():
    n, T, n_mc = 4, 5, 6
    rng = np.random.default_rng(0)
    W_panel = np.stack([_toy_network(n, seed=t)[0] for t in range(T)])
    y832_panel = np.stack([_toy_network(n, seed=t)[1] for t in range(T)])
    m362_panel = np.stack([_toy_network(n, seed=t)[2] for t in range(T)])

    y, loss_frac, shocked_mask, diag = generate_mc_labels_for_panel(
        W_panel, y832_panel, m362_panel, p_shock=0.3, seed=0, n_mc=n_mc)

    assert y.shape == (T, n_mc, n)
    assert shocked_mask.shape == (T, n_mc, n)
    assert diag["n_total"] == T * n_mc * n
    # every directly-shocked bank in this toy network (full wipeout) must be
    # labeled positive -- the deterministic-direct-failure property the
    # spillover-only evaluation metric relies on
    assert (y[shocked_mask] == 1).all()


def test_generate_mc_labels_draws_are_independent_across_mc_index():
    """Different mc draws for the same t should not always pick the same
    shocked set (otherwise M=20 would add no statistical power)."""
    n, T, n_mc = 6, 3, 10
    W = _toy_network(n, seed=0)[0]
    W_panel = np.stack([W] * T)
    y832_panel = np.stack([_toy_network(n, seed=0)[1]] * T)
    m362_panel = np.stack([_toy_network(n, seed=0)[2]] * T)
    _, _, shocked_mask, _ = generate_mc_labels_for_panel(
        W_panel, y832_panel, m362_panel, p_shock=0.4, seed=0, n_mc=n_mc)
    draws_at_t0 = [tuple(shocked_mask[0, m]) for m in range(n_mc)]
    assert len(set(draws_at_t0)) > 1
