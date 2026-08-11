import networkx as nx

from experiments.treewidth_entropy import approx_treewidth


def test_treewidth_returns_width_not_decomposition_tree():
    """Regression test for the swapped-return-order bug: the old code did
    `_, tw = treewidth_min_fill_in(...)`, which assigns the decomposition
    tree (a NetworkX Graph) to `tw`. A correct fix returns an int."""
    G = nx.complete_graph(5)  # K5 has known treewidth 4
    tw = approx_treewidth(G)
    assert isinstance(tw, int)
    assert tw == 4


def test_treewidth_of_cycle_is_two():
    G = nx.cycle_graph(6)  # any cycle with >=3 nodes has treewidth 2
    assert approx_treewidth(G) == 2
