"""Synthetic interbank exposure network generator (fitness model),
rescaled to match BIS aggregate statistics. NOT real Fedwire data —
label all outputs as 'BIS-calibrated synthetic'.
"""
import numpy as np
import networkx as nx

def generate_fitness_scores(n_banks, alpha=2.5, seed=None):
    rng = np.random.default_rng(seed)
    # Pareto-distributed bank "size" fitness (heavy-tailed, like real bank-size dist.)
    return rng.pareto(alpha, size=n_banks) + 1.0

def generate_exposure_network(n_banks, target_total_exposure,
                               density=0.05, alpha=2.5, seed=None):
    rng = np.random.default_rng(seed)
    fitness = generate_fitness_scores(n_banks, alpha, seed)
    outer = np.outer(fitness, fitness)
    p_edge = density * outer / outer.mean()
    p_edge = np.clip(p_edge, 0, 1)
    np.fill_diagonal(p_edge, 0)

    adj_mask = rng.random((n_banks, n_banks)) < p_edge
    weights = adj_mask * outer * rng.lognormal(0, 0.5, size=(n_banks, n_banks))
    np.fill_diagonal(weights, 0)

    # Rescale so total exposure matches BIS aggregate target
    if weights.sum() > 0:
        weights *= target_total_exposure / weights.sum()

    G = nx.from_numpy_array(weights, create_using=nx.DiGraph)
    return weights, G, fitness

def node_features_from_fitness(fitness, rng_seed=None):
    """Map fitness -> approximate vulnerability features (leverage, Tier1, CDS, liq)."""
    rng = np.random.default_rng(rng_seed)
    n = len(fitness)
    leverage = 8 + 4 * (1 / fitness) + rng.normal(0, 0.5, n)
    tier1 = 0.12 - 0.03 * (1 / fitness) + rng.normal(0, 0.01, n)
    cds = rng.lognormal(mean=np.log(50), sigma=0.4, size=n) * (1 / fitness)
    liq = 1.0 + 0.5 * fitness / fitness.max() + rng.normal(0, 0.1, n)
    return np.stack([leverage, tier1, cds, liq], axis=1)

if __name__ == "__main__":
    W, G, fit = generate_exposure_network(n_banks=100, target_total_exposure=5e11, seed=0)
    X = node_features_from_fitness(fit, rng_seed=0)
    np.savez("data/snapshot_0.npz", W=W, X=X)
    print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
