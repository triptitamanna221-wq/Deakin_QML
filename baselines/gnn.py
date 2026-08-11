import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv

# GCNConv takes a scalar per-edge weight via `edge_weight`. GATConv and
# TransformerConv take a (possibly multi-dim) edge feature via `edge_attr`
# and need `edge_dim` set at construction time to accept it. SAGEConv has no
# edge-weight mechanism in torch_geometric, so it silently stays topology-only
# -- that is a known, reported limitation of the SAGE baseline, not a bug.
_EDGE_WEIGHT_CONVS = (GCNConv,)
_EDGE_ATTR_CONVS = (GATConv, TransformerConv)


class GNNBaseline(nn.Module):
    def __init__(self, conv_cls, in_dim, hidden=32, out_dim=1, **conv_kwargs):
        super().__init__()
        self.conv_cls = conv_cls
        if conv_cls in _EDGE_ATTR_CONVS:
            conv_kwargs = {**conv_kwargs, "edge_dim": 1}
        self.conv1 = conv_cls(in_dim, hidden, **conv_kwargs)
        self.conv2 = conv_cls(hidden, hidden, **conv_kwargs)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            h = torch.relu(self.conv1(x, edge_index))
            h = torch.relu(self.conv2(h, edge_index))
        elif self.conv_cls in _EDGE_WEIGHT_CONVS:
            h = torch.relu(self.conv1(x, edge_index, edge_weight))
            h = torch.relu(self.conv2(h, edge_index, edge_weight))
        elif self.conv_cls in _EDGE_ATTR_CONVS:
            edge_attr = edge_weight.view(-1, 1)
            h = torch.relu(self.conv1(x, edge_index, edge_attr))
            h = torch.relu(self.conv2(h, edge_index, edge_attr))
        else:
            h = torch.relu(self.conv1(x, edge_index))
            h = torch.relu(self.conv2(h, edge_index))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def get_gnn_baselines(in_dim):
    return {
        "gcn": GNNBaseline(GCNConv, in_dim),
        "graphsage": GNNBaseline(SAGEConv, in_dim),
        "gat": GNNBaseline(GATConv, in_dim, heads=2, concat=False),
        "graph_transformer": GNNBaseline(TransformerConv, in_dim, heads=2, concat=False),
    }
