import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from qiskit.quantum_info import Statevector, partial_trace, entropy

def approx_treewidth(G):
    # treewidth_min_fill_in returns (width, decomposition_tree) -- in that order.
    tw, _ = nx.algorithms.approximation.treewidth_min_fill_in(G.to_undirected())
    return tw

def entanglement_entropy(qc_bound, n_qubits, subsystem):
    sv = Statevector.from_instruction(qc_bound)
    rho_A = partial_trace(sv, [q for q in range(n_qubits) if q not in subsystem])
    return entropy(rho_A, base=2)

def correlate_entropy_with_auprc_gap(treewidths, entropies, auprc_gaps):
    """Returns Pearson r between (entropy) and (AUPRC gap), and between
    (treewidth) and (AUPRC gap). If both are weak/insignificant, the
    Proposition is NOT empirically supported and must be reported as such."""
    r_entropy, p_entropy = pearsonr(entropies, auprc_gaps)
    r_tw, p_tw = pearsonr(treewidths, auprc_gaps)
    return {
        "entropy_vs_gap_r": r_entropy, "entropy_vs_gap_p": p_entropy,
        "treewidth_vs_gap_r": r_tw, "treewidth_vs_gap_p": p_tw,
    }
