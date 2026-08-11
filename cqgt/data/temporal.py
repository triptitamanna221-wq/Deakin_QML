"""Build a T=120 weekly panel. This is SYNTHETIC TEMPORAL STRUCTURE overlaid
on real data, not 120 real weekly observations -- FR Y-15 is annual. Two
builders:

  build_real_anchor_panel: linearly interpolates between the 4 real annual
  network snapshots (2020-2023; a 5th, 2024, exists in the raw data but is
  not a network anchor -- FFIEC's 2024 export omits M362, see NOTES.md and
  cqgt/data/fry15_loader.py) and overlays AR(1) idiosyncratic weight noise
  plus a crisis-window volatility uplift. Every non-anchor week's network is
  fabricated by this interpolation+noise procedure; only the 4 anchor weeks
  themselves are real data. Flagged for BRIEF.md Sec IV-A.

  build_fallback_panel: entirely synthetic (core-periphery SBM base +
  AR(1)-evolved weights + periodic edge resampling), used only when real
  data is unavailable, or as an intentionally denser comparison network for
  GATE 1 diagnostics.
"""
import numpy as np

from cqgt.data.fallback import generate_core_periphery

T_DEFAULT = 120
CRISIS_WINDOW = (70, 85)
AR1_RHO = 0.95
EDGE_RESAMPLE_FRAC = 0.05
NOISE_STD = 0.03
CRISIS_NOISE_MULT = 3.0


def _ar1_noise_path(T, rho, std, seed, crisis_window=None, crisis_mult=1.0):
    rng = np.random.default_rng(seed)
    path = np.zeros(T)
    for t in range(1, T):
        s = std * crisis_mult if (crisis_window and crisis_window[0] <= t <= crisis_window[1]) else std
        path[t] = rho * path[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, s)
    return path


def build_real_anchor_panel(W_anchors, marginal_anchors, T=T_DEFAULT,
                             crisis_window=CRISIS_WINDOW, seed=0):
    """W_anchors: list of 4 (n,n) arrays for years 2020..2023, in order.
    marginal_anchors: list of 4 (n, n_features) arrays, same order.
    Returns (W_panel, marginal_panel): arrays of shape (T,n,n) / (T,n,k).

    Anchor weeks are placed evenly across [0, T-1]; every other week's
    network is (1) linearly interpolated between its bracketing anchors,
    then (2) perturbed by small multiplicative AR(1) noise, common to all
    edges at a given t so the interpolation stays smooth, with variance
    tripled inside `crisis_window` to inject the required stress regime.
    Anchor weeks themselves are NOT perturbed -- they are the real data."""
    n_anchors = len(W_anchors)
    n = W_anchors[0].shape[0]
    anchor_t = np.linspace(0, T - 1, n_anchors).round().astype(int)

    noise = _ar1_noise_path(T, AR1_RHO, NOISE_STD, seed, crisis_window, CRISIS_NOISE_MULT)

    W_panel = np.zeros((T, n, n))
    m_panel = np.zeros((T, n, marginal_anchors[0].shape[1]))
    for t in range(T):
        if t in anchor_t:
            idx = int(np.where(anchor_t == t)[0][0])
            W_panel[t] = W_anchors[idx]
            m_panel[t] = marginal_anchors[idx]
            continue
        lo = int(np.searchsorted(anchor_t, t) - 1)
        lo = max(0, min(lo, n_anchors - 2))
        hi = lo + 1
        frac = (t - anchor_t[lo]) / (anchor_t[hi] - anchor_t[lo])
        W_interp = (1 - frac) * W_anchors[lo] + frac * W_anchors[hi]
        m_interp = (1 - frac) * marginal_anchors[lo] + frac * marginal_anchors[hi]
        scale = max(0.0, 1.0 + noise[t])
        W_panel[t] = W_interp * scale
        np.fill_diagonal(W_panel[t], 0.0)
        m_panel[t] = m_interp * scale
    return W_panel, m_panel, anchor_t


def build_fallback_panel(T=T_DEFAULT, crisis_window=CRISIS_WINDOW, seed=0,
                          ar1_rho=AR1_RHO, edge_resample_frac=EDGE_RESAMPLE_FRAC):
    """Fully synthetic panel: one core-periphery draw as t=0 base topology,
    weights evolved by AR(1), a small fraction of edges resampled each step
    (topology turnover), and elevated shock magnitude + core density inside
    `crisis_window`."""
    rng = np.random.default_rng(seed)
    W0 = generate_core_periphery(seed=seed)
    n = W0.shape[0]
    mask = W0 > 0

    W_panel = np.zeros((T, n, n))
    W_panel[0] = W0
    for t in range(1, T):
        in_crisis = crisis_window[0] <= t <= crisis_window[1]
        std = NOISE_STD * (CRISIS_NOISE_MULT if in_crisis else 1.0)
        shock = rng.normal(0, std, size=(n, n))
        W_t = ar1_rho * W_panel[t - 1] + (1 - ar1_rho) * W0 * (1 + shock)
        W_t = np.clip(W_t, 0, None) * mask

        # resample a small fraction of edges (topology turnover)
        resample_frac = edge_resample_frac * (2.0 if in_crisis else 1.0)
        edge_idx = np.argwhere(mask)
        n_resample = max(1, int(len(edge_idx) * resample_frac))
        chosen = rng.choice(len(edge_idx), size=n_resample, replace=False)
        for idx in chosen:
            i, j = edge_idx[idx]
            W_t[i, j] = W0[i, j] * rng.lognormal(0, std * 3)

        # crisis: extra density/magnitude among the most connected (core) pairs
        if in_crisis:
            core_edges = np.argwhere(mask & (W0 > np.percentile(W0[mask], 75)))
            for i, j in core_edges:
                W_t[i, j] *= 1.0 + rng.uniform(0.1, 0.4)

        np.fill_diagonal(W_t, 0.0)
        W_panel[t] = W_t
    return W_panel
