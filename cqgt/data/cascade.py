"""Eisenberg-Noe (2001) interbank clearing cascade: the single most
important component of this project (BRIEF.md Sec 2.1) -- labels must
depend on network topology, or the whole paper is void.

Convention: W[i, j] = i's claim on j (j owes i), matching the row=assets /
column=liabilities convention used throughout cqgt/data/reconstruction.py.
Bank i's total interbank obligations L_i = W.sum(axis=0)[i] (column sum).

Two quantities are genuinely FR Y-15-derived; one is an assumed literature
parameter (not real data), disclosed here and in NOTES.md:
  external_assets[i] = y832[i] - m362[i]   (real: total exposure minus the
                                             intra-financial-system subset)
  equity[i]          = capital_ratio * y832[i]   (ASSUMED: FR Y-15 reports
                                             no capital/equity figure at
                                             all; capital_ratio defaults to
                                             0.08, a literature-typical
                                             G-SIB leverage-ratio buffer,
                                             not a measured value for any
                                             specific institution)
"""
import numpy as np

DEFAULT_CAPITAL_RATIO = 0.08
DEFAULT_LOSS_THRESHOLD = 0.30  # y_i=1 if equity loss exceeds 30% of pre-shock equity

# Calibration note (see NOTES.md for the full search): for these 12 real
# large banks, external assets are 7-180x their interbank liabilities L
# (they fund mainly outside the interbank market -- realistic for G-SIBs).
# An Eisenberg-Noe PAYMENT shortfall (the only channel through which one
# bank's shock can affect another's) therefore requires shocking a bank's
# external assets almost to zero; partial haircuts (5-50%) wipe out the
# assumed thin (8%) equity buffer without ever causing an actual missed
# payment, so they produce zero spillover by construction, not by bug.
# shock_frac=1.0 below models a shocked bank as a genuine failure event.
DEFAULT_SHOCK_FRAC = 1.0


def external_assets_from_marginals(y832, m362):
    return np.asarray(y832, dtype=float) - np.asarray(m362, dtype=float)


def assumed_equity(y832, capital_ratio=DEFAULT_CAPITAL_RATIO):
    return capital_ratio * np.asarray(y832, dtype=float)


def clear(W, external_assets, max_iter=200, tol=1e-8):
    """Eisenberg-Noe fixed-point clearing. Returns (p, net_worth) where p is
    the clearing payment vector (n,) and net_worth[i] = external_assets[i] +
    received_i(p) - L_i (can be negative = insolvent)."""
    n = W.shape[0]
    L = W.sum(axis=0)  # total interbank obligations per bank (column sums)
    p = L.copy()
    for _ in range(max_iter):
        received = np.zeros(n)
        nz = L > 0
        # received_i = sum_j (W[i,j] / L_j) * p_j  -- i's share of each j's
        # total obligations, times what j actually pays
        share = np.divide(W, L[None, :], out=np.zeros_like(W), where=(L[None, :] > 0))
        received = share @ p
        p_new = np.minimum(L, np.maximum(0.0, external_assets + received))
        if np.abs(p_new - p).max() < tol:
            p = p_new
            break
        p = p_new
    received = (np.divide(W, L[None, :], out=np.zeros_like(W), where=(L[None, :] > 0))) @ p
    net_worth = external_assets + received - L
    return p, net_worth


def run_cascade(W, y832, m362, shocked_idx, shock_frac, capital_ratio=DEFAULT_CAPITAL_RATIO,
                 loss_threshold=DEFAULT_LOSS_THRESHOLD):
    """Apply a proportional haircut `shock_frac` to external_assets of banks
    in `shocked_idx`, clear the network, and return per-bank equity-loss
    fractions and binary distress labels y_i = 1[loss_frac > loss_threshold]."""
    ext0 = external_assets_from_marginals(y832, m362)
    equity0 = assumed_equity(y832, capital_ratio)

    ext_shocked = ext0.copy()
    ext_shocked[shocked_idx] *= (1.0 - shock_frac)

    _, net_worth = clear(W, ext_shocked)
    # pre-shock net worth (no shock, but still cleared through the network,
    # so the baseline reflects normal interbank flows, not just ext0)
    _, net_worth0 = clear(W, ext0)

    equity_loss = net_worth0 - net_worth
    loss_frac = np.divide(equity_loss, equity0, out=np.zeros_like(equity_loss), where=equity0 > 0)
    y = (loss_frac > loss_threshold).astype(int)
    return y, loss_frac, net_worth, net_worth0


def aggregate_risk(loss_frac):
    """R: mean equity-loss fraction across the system (used for the
    counterfactual attribution metric, per cqgt/counterfactual.py)."""
    return float(np.mean(loss_frac))


def generate_labels_for_panel(W_panel, y832_panel, m362_panel, shock_frac=DEFAULT_SHOCK_FRAC,
                               p_shock=0.08, capital_ratio=DEFAULT_CAPITAL_RATIO,
                               loss_threshold=DEFAULT_LOSS_THRESHOLD, seed=0):
    """Per-snapshot Bernoulli(p_shock) shock assignment (each bank
    independently shocked with probability p_shock each period, seeded).
    Returns y (T,N), loss_frac (T,N), and a diagnostics dict separating
    direct-shock positives from spillover (network-mediated) positives --
    the latter is the whole point of the exercise."""
    T, n = y832_panel.shape
    y = np.zeros((T, n), dtype=int)
    loss_frac = np.zeros((T, n))
    n_direct_pos, n_spillover_pos, n_shocked_total = 0, 0, 0
    for t in range(T):
        rng = np.random.default_rng((seed, t))
        shocked = np.where(rng.random(n) < p_shock)[0]
        n_shocked_total += len(shocked)
        if len(shocked) == 0:
            continue
        yt, lf, _, _ = run_cascade(W_panel[t], y832_panel[t], m362_panel[t], shocked,
                                    shock_frac, capital_ratio, loss_threshold)
        y[t], loss_frac[t] = yt, lf
        shocked_mask = np.zeros(n, dtype=bool)
        shocked_mask[shocked] = True
        n_direct_pos += yt[shocked_mask].sum()
        n_spillover_pos += yt[~shocked_mask].sum()
    diagnostics = {
        "base_rate": y.mean(),
        "n_positive": int(y.sum()), "n_total": y.size,
        "n_direct_positive": int(n_direct_pos), "n_spillover_positive": int(n_spillover_pos),
        "mean_banks_shocked_per_t": n_shocked_total / T,
    }
    return y, loss_frac, diagnostics


def generate_mc_labels_for_panel(W_panel, y832_panel, m362_panel, shock_frac=DEFAULT_SHOCK_FRAC,
                                  p_shock=0.08, capital_ratio=DEFAULT_CAPITAL_RATIO,
                                  loss_threshold=DEFAULT_LOSS_THRESHOLD, seed=0, n_mc=20):
    """Like generate_labels_for_panel, but draws `n_mc` INDEPENDENT shock
    realizations per snapshot t, each kept as a separate scenario-example
    rather than being the only draw for that t. This is the statistical-
    power refinement from the GATE 1 negative finding (see NOTES.md): more
    independent (t, mc) examples to estimate whether the network carries a
    real (if small) spillover signal, without changing the underlying
    exogenous-shock design (no leakage risk -- see NOTES.md's discussion of
    why Option 3, correlating shock probability with network position, was
    explicitly rejected).

    Returns y, loss_frac, shocked_mask, all shape (T, n_mc, N), plus
    diagnostics. shocked_mask is the scenario input (which banks were hit
    this draw) and is meant to be fed to the model as a feature -- it is
    not part of the outcome being predicted."""
    T, n = y832_panel.shape
    y = np.zeros((T, n_mc, n), dtype=int)
    loss_frac = np.zeros((T, n_mc, n))
    shocked_mask = np.zeros((T, n_mc, n), dtype=bool)
    n_direct_pos, n_spillover_pos, n_shocked_total = 0, 0, 0
    for t in range(T):
        for m in range(n_mc):
            rng = np.random.default_rng((seed, t, m))
            shocked = np.where(rng.random(n) < p_shock)[0]
            n_shocked_total += len(shocked)
            if len(shocked) == 0:
                continue
            yt, lf, _, _ = run_cascade(W_panel[t], y832_panel[t], m362_panel[t], shocked,
                                        shock_frac, capital_ratio, loss_threshold)
            y[t, m], loss_frac[t, m] = yt, lf
            shocked_mask[t, m, shocked] = True
            n_direct_pos += yt[shocked_mask[t, m]].sum()
            n_spillover_pos += yt[~shocked_mask[t, m]].sum()
    diagnostics = {
        "base_rate": y.mean(),
        "n_positive": int(y.sum()), "n_total": y.size,
        "n_direct_positive": int(n_direct_pos), "n_spillover_positive": int(n_spillover_pos),
        "mean_banks_shocked_per_t": n_shocked_total / (T * n_mc),
        "n_mc": n_mc,
    }
    return y, loss_frac, shocked_mask, diagnostics


def tune_p_shock(W_panel, y832_panel, m362_panel, target_range=(0.08, 0.15),
                  shock_frac=DEFAULT_SHOCK_FRAC, capital_ratio=DEFAULT_CAPITAL_RATIO,
                  loss_threshold=DEFAULT_LOSS_THRESHOLD, seed=0, lo=0.01, hi=0.5, n_iter=25):
    """Bisection search on p_shock to land the panel's aggregate base rate
    inside target_range. Returns (p_shock, diagnostics) for the closest hit."""
    target_mid = sum(target_range) / 2
    best = None
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        _, _, diag = generate_labels_for_panel(W_panel, y832_panel, m362_panel, shock_frac,
                                                mid, capital_ratio, loss_threshold, seed)
        rate = diag["base_rate"]
        if best is None or abs(rate - target_mid) < abs(best[1]["base_rate"] - target_mid):
            best = (mid, diag)
        if target_range[0] <= rate <= target_range[1]:
            return mid, diag
        if rate < target_mid:
            lo = mid
        else:
            hi = mid
    return best


def ground_truth_delta_r(W, y832, m362, shocked_idx, shock_frac, capital_ratio=DEFAULT_CAPITAL_RATIO):
    """For each edge (i,j) with W[i,j] > 0, rerun the cascade with that edge
    zeroed and record R0 - R_cf. This is the ground truth for the
    counterfactual attribution metric (BRIEF.md Sec 2.1 / Sec 4 T3)."""
    _, loss0, _, _ = run_cascade(W, y832, m362, shocked_idx, shock_frac, capital_ratio)
    R0 = aggregate_risk(loss0)

    n = W.shape[0]
    delta = {}
    for i in range(n):
        for j in range(n):
            if W[i, j] <= 0:
                continue
            W_cf = W.copy()
            W_cf[i, j] = 0.0
            _, loss_cf, _, _ = run_cascade(W_cf, y832, m362, shocked_idx, shock_frac, capital_ratio)
            delta[(i, j)] = R0 - aggregate_risk(loss_cf)
    return delta
