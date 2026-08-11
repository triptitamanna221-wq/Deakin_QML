"""Core-periphery stochastic block model: the synthetic fallback exposure
network used when real FR Y-15 data is absent, and as a deliberately
denser comparison network for GATE 1 diagnostics (BRIEF.md Sec 2.1). Unlike
the fitness-model generator it replaces, this guarantees treewidth >= 3 by
construction (a handful of densely-interconnected core banks force
non-trivial tree decompositions), which is the regime the theory in
BRIEF.md Sec 3 is about.
"""
import networkx as nx
import numpy as np

N_CORE = 4
N_PERIPHERY = 8
P_CORE = 0.9
P_PERIPHERY = 0.05
P_CORE_PERIPHERY = 0.3
PARETO_ALPHA = 2.5


def generate_core_periphery(n_core=N_CORE, n_periphery=N_PERIPHERY, p_core=P_CORE,
                             p_periphery=P_PERIPHERY, p_core_periphery=P_CORE_PERIPHERY,
                             pareto_alpha=PARETO_ALPHA, seed=None):
    """Directed weighted exposure matrix W[i, j] = exposure of i to j. Core
    banks are indices [0, n_core); periphery banks are [n_core, n_core+n_periphery).
    Edge probability depends on which block the (row, column) pair falls in;
    weights are Pareto(alpha) + 1 (heavy-tailed, like real bank exposure
    sizes), diagonal is zero."""
    rng = np.random.default_rng(seed)
    n = n_core + n_periphery
    is_core = np.arange(n) < n_core

    p_matrix = np.where(is_core[:, None] & is_core[None, :], p_core,
                np.where(~is_core[:, None] & ~is_core[None, :], p_periphery, p_core_periphery))
    adj = rng.random((n, n)) < p_matrix
    np.fill_diagonal(adj, False)

    weights = rng.pareto(pareto_alpha, size=(n, n)) + 1.0
    W = adj * weights
    np.fill_diagonal(W, 0.0)
    return W


def to_networkx(W):
    return nx.from_numpy_array(W, create_using=nx.DiGraph)


def synthetic_marginals_from_panel(W_panel, external_multiplier=10.0, seed=0):
    """Entirely synthetic y832/m362/m370-analog marginals for the fallback
    panel, since there is no real FR Y-15 data behind it. m362_analog and
    m370_analog are the network's own row/column sums (self-consistent by
    construction); y832_analog = external_multiplier x (m362+m370)/2 keeps
    the external-asset-to-interbank-liability ratio in the same order of
    magnitude as the real panel (~7-180x, see cqgt/data/cascade.py), so the
    fallback's contagion dynamics are comparable rather than trivially
    easier or harder to trigger than the real-data case."""
    T, n, _ = W_panel.shape
    m362 = W_panel.sum(axis=2)  # row sums per t
    m370 = W_panel.sum(axis=1)  # col sums per t
    y832 = external_multiplier * (m362 + m370) / 2.0
    return y832, m362, m370


def synthetic_extra_fields(y832, seed=0):
    """Synthetic analogs of M376/M390/M405/M408/M411/M422/M426 for the
    fallback panel's feature construction (cqgt/data/features.py), as
    plausible fractions of y832 with idiosyncratic per-institution draws.
    Fully synthetic, disclosed -- there is no real filing behind this path."""
    rng = np.random.default_rng(seed)
    T, n = y832.shape
    frac = {
        "m376": rng.uniform(0.02, 0.15, size=n),
        "m390": rng.uniform(0.05, 0.25, size=n),
        "m405": rng.uniform(0.0, 0.3, size=n),
        "m408": rng.uniform(0.0, 0.1, size=n),
        "m411": rng.uniform(0.05, 0.3, size=n),
        "m422": rng.uniform(0.01, 0.1, size=n),
        "m426": rng.uniform(0.01, 0.1, size=n),
    }
    return {k: y832 * v[None, :] for k, v in frac.items()}
