import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv

class GNNBaseline(nn.Module):
    def __init__(self, conv_cls, in_dim, hidden=32, out_dim=1, **conv_kwargs):
        super().__init__()
        self.conv1 = conv_cls(in_dim, hidden, **conv_kwargs)
        self.conv2 = conv_cls(hidden, hidden, **conv_kwargs)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
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
