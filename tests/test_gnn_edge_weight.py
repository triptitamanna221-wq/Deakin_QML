import torch

from baselines.gnn import GNNBaseline
from torch_geometric.nn import GCNConv, GATConv


def _toy_graph():
    x = torch.randn(4, 3, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return x, edge_index


def test_gcn_output_changes_with_edge_weight():
    """Regression test: the old forward() accepted edge_weight and silently
    dropped it, so GCN predictions were identical with/without it -- an
    unfair comparison against CQGT, which does use exposure magnitudes."""
    torch.manual_seed(0)
    model = GNNBaseline(GCNConv, in_dim=3)
    x, edge_index = _toy_graph()
    ew_a = torch.tensor([1.0, 1.0, 1.0, 1.0])
    ew_b = torch.tensor([5.0, 0.1, 3.0, 2.0])
    out_a = model(x, edge_index, ew_a)
    out_b = model(x, edge_index, ew_b)
    assert not torch.allclose(out_a, out_b)


def test_gat_output_changes_with_edge_weight():
    torch.manual_seed(0)
    model = GNNBaseline(GATConv, in_dim=3, heads=2, concat=False)
    x, edge_index = _toy_graph()
    ew_a = torch.tensor([1.0, 1.0, 1.0, 1.0])
    ew_b = torch.tensor([5.0, 0.1, 3.0, 2.0])
    out_a = model(x, edge_index, ew_a)
    out_b = model(x, edge_index, ew_b)
    assert not torch.allclose(out_a, out_b)


def test_no_edge_weight_still_works():
    torch.manual_seed(0)
    model = GNNBaseline(GCNConv, in_dim=3)
    x, edge_index = _toy_graph()
    out = model(x, edge_index)
    assert out.shape == (4,)
