import numpy as np
from data.generate_exposures import generate_exposure_network, node_features_from_fitness

def build_panel(n_days=252, n_banks=200, target_total=5e11, drift=0.01, seed=0):
    rng = np.random.default_rng(seed)
    panel = []
    fitness = None
    for t in range(n_days):
        s = seed + t
        W, G, fitness_t = generate_exposure_network(n_banks, target_total, seed=s)
        if fitness is not None:
            fitness_t = (1 - drift) * fitness + drift * fitness_t
        fitness = fitness_t
        X = node_features_from_fitness(fitness, rng_seed=s)
        panel.append({"day": t, "W": W, "X": X})
    return panel

if __name__ == "__main__":
    panel = build_panel()
    np.save("data/panel.npy", panel, allow_pickle=True)
