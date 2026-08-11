import numpy as np

from cqgt.data.temporal import build_fallback_panel, build_real_anchor_panel


def test_real_anchor_panel_preserves_anchor_snapshots_exactly():
    n = 5
    rng = np.random.default_rng(0)
    W_anchors = [rng.uniform(0, 1, size=(n, n)) for _ in range(4)]
    for W in W_anchors:
        np.fill_diagonal(W, 0)
    m_anchors = [rng.uniform(0, 1, size=(n, 3)) for _ in range(4)]

    W_panel, m_panel, anchor_t = build_real_anchor_panel(W_anchors, m_anchors, T=120, seed=0)
    assert len(anchor_t) == 4
    for idx, t in enumerate(anchor_t):
        np.testing.assert_allclose(W_panel[t], W_anchors[idx])
        np.testing.assert_allclose(m_panel[t], m_anchors[idx])


def test_real_anchor_panel_diagonal_always_zero():
    n = 5
    rng = np.random.default_rng(1)
    W_anchors = [rng.uniform(0, 1, size=(n, n)) for _ in range(4)]
    for W in W_anchors:
        np.fill_diagonal(W, 0)
    m_anchors = [rng.uniform(0, 1, size=(n, 3)) for _ in range(4)]
    W_panel, _, _ = build_real_anchor_panel(W_anchors, m_anchors, T=40, seed=0)
    assert np.allclose(np.diagonal(W_panel, axis1=1, axis2=2), 0)


def test_fallback_panel_shape_and_nonnegative():
    W_panel = build_fallback_panel(T=30, seed=0)
    assert W_panel.shape == (30, 12, 12)
    assert (W_panel >= 0).all()
    assert np.allclose(np.diagonal(W_panel, axis1=1, axis2=2), 0)
