import itertools

ABLATIONS = {
    "no_hamiltonian": {"use_hamiltonian": False},
    "classical_attention": {"use_quantum_attention": False},
    "random_edges": {"use_random_topology": True},
    "hardware_efficient_ansatz": {"use_topology_matched": False},
    "no_macro_coupling": {"gamma": 0.0},
    "no_counterfactual": {"use_counterfactual": False},
    "noisy_sim": {"noise_model": "ibm_calibrated"},
    "full_model": {},
}

def run_all_ablations(train_fn, base_config):
    results = {}
    for name, overrides in ABLATIONS.items():
        cfg = {**base_config, **overrides}
        results[name] = train_fn(cfg)
    return results
