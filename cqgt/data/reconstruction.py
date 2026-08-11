"""Reconstruct bilateral interbank exposure matrices from row/column
marginals (per-institution intra-financial-system assets and liabilities).

Real marginals never balance (sum(assets) != sum(liabilities)) because each
institution's FR Y-15 figures include claims on / obligations to
counterparties outside our N=12 panel. IPF/maxent requires balanced
marginals to converge. Per project decision: proportionally rescale both
vectors to their common average total rather than adding a rest-of-world
node (simpler, preserves each institution's relative share of the
financial system it actually reports into; the tradeoff is that the
absolute $ scale of the reconstructed network is not the literal FR Y-15
number -- see NOTES.md for the measured imbalance and this tradeoff).

Two reconstruction methods, run independently, both reported (BRIEF.md Sec
2.1): `maxent_reconstruct` (RAS/iterative-proportional-fitting, smooths
exposures across the whole network -- under-estimates contagion) and
`mindensity_reconstruct` (a greedy, fewest-links allocation in the spirit
of Anand, Craig & von Peter 2015 -- over-estimates contagion by
concentrating exposure on few counterparties). They bracket the true,
unobservable network.
"""
import numpy as np


def rescale_marginals(assets, liabilities):
    """Proportionally rescale both vectors to their common average total so
    IPF has a feasible balanced problem. Returns (assets_scaled,
    liabilities_scaled, report) where report records the pre-rescaling
    imbalance for disclosure."""
    assets = np.asarray(assets, dtype=float)
    liabilities = np.asarray(liabilities, dtype=float)
    total_a, total_l = assets.sum(), liabilities.sum()
    target = (total_a + total_l) / 2.0
    imbalance_pct = 100.0 * (total_a - total_l) / target
    report = {
        "total_assets": total_a, "total_liabilities": total_l,
        "target_total": target, "imbalance_pct": imbalance_pct,
    }
    a_scaled = assets * (target / total_a)
    l_scaled = liabilities * (target / total_l)
    return a_scaled, l_scaled, report


def maxent_reconstruct(assets, liabilities, max_iter=2000, tol=1e-8):
    """RAS / iterative proportional fitting with a zero diagonal, matching
    row sums to `assets` and column sums to `liabilities`. Requires
    sum(assets) == sum(liabilities) (use rescale_marginals first)."""
    assets = np.asarray(assets, dtype=float)
    liabilities = np.asarray(liabilities, dtype=float)
    n = len(assets)
    total = assets.sum()
    if not np.isclose(total, liabilities.sum(), rtol=1e-6):
        raise ValueError("maxent_reconstruct requires balanced marginals; call "
                          "rescale_marginals first.")
    W = np.outer(assets, liabilities) / total
    np.fill_diagonal(W, 0.0)
    for _ in range(max_iter):
        row_sums = W.sum(axis=1)
        row_scale = np.divide(assets, row_sums, out=np.ones(n), where=row_sums > 0)
        W = W * row_scale[:, None]
        np.fill_diagonal(W, 0.0)

        col_sums = W.sum(axis=0)
        col_scale = np.divide(liabilities, col_sums, out=np.ones(n), where=col_sums > 0)
        W = W * col_scale[None, :]
        np.fill_diagonal(W, 0.0)

        err = max(np.abs(W.sum(axis=1) - assets).max(), np.abs(W.sum(axis=0) - liabilities).max())
        if err < tol:
            break
    return W


def mindensity_reconstruct(assets, liabilities, tol=1e-6):
    """Greedy fewest-links allocation: repeatedly match the counterparty
    with the largest remaining lending capacity to the one with the
    largest remaining borrowing need (excluding self-loops), assigning the
    maximal exposure consistent with both remaining capacities. This is a
    simplified greedy variant in the spirit of Anand, Craig & von Peter
    (2015)'s minimum-density method (not their exact LP formulation) --
    produces at most 2N-1 nonzero edges, concentrating exposure rather than
    smoothing it, which over-estimates contagion relative to maxent."""
    assets = np.asarray(assets, dtype=float)
    liabilities = np.asarray(liabilities, dtype=float)
    n = len(assets)
    W = np.zeros((n, n))
    rem_a, rem_l = assets.copy(), liabilities.copy()

    for _ in range(4 * n * n):  # O(n^2) pairs at most ever needed; generous cap
        if rem_a.sum() <= tol or rem_l.sum() <= tol:
            break
        # Try lenders in descending remaining-capacity order; a lender being
        # "stuck" (its only remaining counterparty would be itself) must not
        # abort the whole pass -- fall through to the next-largest lender,
        # which is exactly the case that was previously mishandled (a
        # smaller lender's capacity going unused while a larger lender sits
        # blocked on a same-institution match).
        matched = False
        for i in (idx for idx in np.argsort(-rem_a) if rem_a[idx] > tol):
            j = next((idx for idx in np.argsort(-rem_l) if idx != i and rem_l[idx] > tol), None)
            if j is not None:
                amt = min(rem_a[i], rem_l[j])
                W[i, j] += amt
                rem_a[i] -= amt
                rem_l[j] -= amt
                matched = True
                break
        if not matched:
            break  # only true self-referential mass remains; handled below

    # Degenerate case: after the above, any remaining mass must be strictly
    # self-referential per institution (rem_a[i] and rem_l[i] both > 0 for
    # the same i -- every other institution's marginals are already
    # exhausted, and no i != j pairing is possible). Resolve with a 3-edge
    # transportation-simplex pivot: find an existing edge W[m, k] (m,k != i)
    # with spare capacity and reroute amt through it --
    #   W[i, k] += amt   (satisfies i's row/asset residual)
    #   W[m, k] -= amt   (cancels k's resulting column/liability excess)
    #   W[m, i] += amt   (cancels m's resulting row/asset deficit, and
    #                      satisfies i's column/liability residual)
    # Every institution besides i keeps its exact row and column sum;
    # nothing here is an approximation of the marginals, only of which
    # specific bilateral edges carry the flow.
    for i in range(n):
        amt = min(rem_a[i], rem_l[i])
        while amt > tol:
            candidates = [(m, k) for m in range(n) for k in range(n)
                          if m != i and k != i and m != k and W[m, k] > tol]
            if not candidates:
                break  # leaves a small disclosed residual rather than crashing
            m, k = max(candidates, key=lambda mk: W[mk[0], mk[1]])
            step = min(amt, W[m, k])
            W[i, k] += step
            W[m, k] -= step
            W[m, i] += step
            rem_a[i] -= step
            rem_l[i] -= step
            amt -= step
    return W
