import networkx as nx

from cqgt.data.fallback import generate_core_periphery, to_networkx
from experiments.treewidth_entropy import approx_treewidth


def test_core_periphery_zero_diagonal_and_nonnegative():
    W = generate_core_periphery(seed=0)
    assert (W.diagonal() == 0).all()
    assert (W >= 0).all()


def test_core_periphery_guarantees_treewidth_at_least_3():
    for seed in range(10):
        W = generate_core_periphery(seed=seed)
        G = to_networkx(W)
        tw = approx_treewidth(G)
        assert tw >= 3, f"seed {seed} gave treewidth {tw} < 3"
