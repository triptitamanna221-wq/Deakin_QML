"""1-WL separation demonstration (BRIEF.md Sec 3), replacing the withdrawn
entanglement-entropy Proposition in III-E with something provable in three
lines and demonstrable numerically:

    C6 (a single 6-cycle) and 2xC3 (two disjoint triangles) are both
    2-regular graphs. With identical node features, the 1-Weisfeiler-Lehman
    color-refinement algorithm cannot distinguish them: every node's
    multiset of neighbor colors is identical at every refinement round (each
    node always has exactly 2 neighbors, all sharing the same color as
    every other node). Any message-passing GNN bounded by 1-WL (GCN, GAT,
    and in fact any standard MPNN) is provably at most as expressive as
    1-WL (Xu et al. 2019, "How Powerful are Graph Neural Networks?"), so
    GCN/GAT MUST produce identical per-node outputs on the two graphs.

    The topology-matched CQGT circuit is not 1-WL-bounded: its hopping term
    e^{-i H delta} evolves under the graph Laplacian's spectrum, and C6's
    unnormalized Laplacian spectrum {0,1,1,3,3,4} differs from 2xC3's
    {0,0,3,3,3,3} -- a real spectral distinction 1-WL cannot see. This
    script demonstrates that CQGT's <Z_i> readout differs between the two
    graphs while GCN/GAT's outputs are identical to numerical precision,
    using IDENTICAL parameters across both topologies (state_dict copied
    from one model instance to the other) so the only difference is which
    qubit pairs the circuit's gates act on -- not initialization.

    Kept as a SEPARATE claim from the entanglement-entropy vs. treewidth
    study (experiments/treewidth_entropy.py, F3): that remains an empirical
    correlation, not a proof step.
"""
import numpy as np
import torch
from torch_geometric.nn import GATConv, GCNConv

from baselines.gnn import GNNBaseline
from cqgt.model import CQGTModel
from cqgt.seeding import set_seed

N = 6
N_FEATURES = 4
N_MACRO = 3
N_LAYERS = 3

C6_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
TWO_C3_EDGES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]


def _edge_index_and_weight(edges):
    idx = np.array(edges + [(j, i) for (i, j) in edges]).T
    edge_index = torch.tensor(idx, dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


def _W_from_edges(edges, n=N):
    W = np.zeros((n, n))
    for i, j in edges:
        W[i, j] = W[j, i] = 1.0
    return W


def run(seed=0):
    x = torch.ones(N, N_FEATURES)  # identical node features -- the whole point
    results = {}

    for name, conv_cls, kwargs in [("gcn", GCNConv, {}), ("gat", GATConv, {"heads": 2, "concat": False})]:
        set_seed(seed)
        model = GNNBaseline(conv_cls, in_dim=N_FEATURES, **kwargs)
        model.eval()
        ei_c6, ew_c6 = _edge_index_and_weight(C6_EDGES)
        ei_c3, ew_c3 = _edge_index_and_weight(TWO_C3_EDGES)
        with torch.no_grad():
            out_c6 = model(x, ei_c6, ew_c6).numpy()
            out_c3 = model(x, ei_c3, ew_c3).numpy()
        results[name] = {"c6": out_c6, "c3": out_c3, "abs_diff": np.abs(out_c6 - out_c3)}

    set_seed(seed)
    cqgt_c6 = CQGTModel(n_qubits=N, n_features=N_FEATURES, edges=C6_EDGES, n_layers=N_LAYERS, n_macro=N_MACRO)
    cqgt_c3 = CQGTModel(n_qubits=N, n_features=N_FEATURES, edges=TWO_C3_EDGES, n_layers=N_LAYERS, n_macro=N_MACRO)
    cqgt_c3.load_state_dict(cqgt_c6.state_dict())  # identical params; only topology differs
    cqgt_c6.eval()
    cqgt_c3.eval()

    W_c6, W_c3 = _W_from_edges(C6_EDGES), _W_from_edges(TWO_C3_EDGES)
    macro_t = torch.zeros(N_MACRO)
    xb = x.unsqueeze(0)  # (1, N, F)
    with torch.no_grad():
        r_c6 = cqgt_c6.circuit_expectation_z(xb, W_c6, macro_t).squeeze(0).numpy()
        r_c3 = cqgt_c3.circuit_expectation_z(xb, W_c3, macro_t).squeeze(0).numpy()
    results["cqgt"] = {"c6": r_c6, "c3": r_c3, "abs_diff": np.abs(r_c6 - r_c3)}

    return results


def summarize(results):
    lines = []
    for name in ("gcn", "gat", "cqgt"):
        d = results[name]["abs_diff"]
        lines.append(f"{name:>5s}: max|C6 - 2xC3| = {d.max():.3e}  mean = {d.mean():.3e}")
    return "\n".join(lines)


# Okabe-Ito colorblind-safe categorical palette, fixed assignment order.
_COLOR_GCN, _COLOR_GAT, _COLOR_CQGT = "#0072B2", "#E69F00", "#009E73"


def plot_f2(results, path="figures/F2_wl_separation.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = np.arange(1, N + 1)
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for offset, (name, color, label) in zip(
        (-1, 0, 1),
        [("gcn", _COLOR_GCN, "GCN"), ("gat", _COLOR_GAT, "GAT"), ("cqgt", _COLOR_CQGT, "CQGT")],
    ):
        diffs = results[name]["abs_diff"].reshape(-1)
        bars = ax.bar(nodes + offset * width, diffs, width=width, color=color, label=label,
                       edgecolor="white", linewidth=0.5)
        if diffs.max() < 1e-9:
            for b in bars:
                ax.annotate("0", (b.get_x() + b.get_width() / 2, 0), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=7, color="#555555")

    ax.set_xlabel("Node index")
    ax.set_ylabel(r"$|\,\mathrm{output}(C_6) - \mathrm{output}(2{\times}C_3)\,|$")
    ax.set_title("1-WL separation: $C_6$ vs. $2{\\times}C_3$, identical node features")
    ax.set_xticks(nodes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    results = run()
    print(summarize(results))
    for name in ("gcn", "gat", "cqgt"):
        print(f"\n{name} C6:    {np.round(results[name]['c6'].reshape(-1), 6)}")
        print(f"{name} 2xC3:  {np.round(results[name]['c3'].reshape(-1), 6)}")
    out_path = plot_f2(results)
    print(f"\nWrote {out_path}")
