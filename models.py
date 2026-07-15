# models.py
# The three GAT variations we're testing against each other.
# 1. Baseline: just stacked GAT + BatchNorm
# 2. Residual: same thing but with skip connections
# 3. LayerNorm: swapping BatchNorm for LayerNorm

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm, ModuleList, Identity
from torch_geometric.nn import GATConv, global_mean_pool

class GATBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, num_layers=3):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout))
        self.norms.append(BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, return_embeddings=False, return_all_layers=False):
        layer_embeddings = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x) if x.shape[0] > 1 else x  # BatchNorm needs >1 node
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_embeddings.append(x)
        node_embeddings = x
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)
        out = self.lin(pooled)
        if return_all_layers:
            return out, layer_embeddings
        if return_embeddings:
            return out, node_embeddings
        return out

class GATResidual(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, num_layers=3):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.projs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(BatchNorm1d(hidden_channels * heads))
        self.projs.append(
            Linear(in_channels, hidden_channels * heads) if in_channels != hidden_channels * heads else Identity()
        )
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(BatchNorm1d(hidden_channels * heads))
            self.projs.append(Identity())
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout))
        self.norms.append(BatchNorm1d(hidden_channels))
        self.projs.append(Linear(hidden_channels * heads, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, return_embeddings=False, return_all_layers=False):
        layer_embeddings = [x]
        for conv, norm, proj in zip(self.convs, self.norms, self.projs):
            res = proj(x)
            x = conv(x, edge_index)
            x = norm(x) if x.shape[0] > 1 else x
            x = F.elu(x)
            x = x + res  # skip connection
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_embeddings.append(x)
        node_embeddings = x
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)
        out = self.lin(pooled)
        if return_all_layers:
            return out, layer_embeddings
        if return_embeddings:
            return out, node_embeddings
        return out

class GATLayerNorm(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, num_layers=3):
        super().__init__()
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(LayerNorm(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(LayerNorm(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout))
        self.norms.append(LayerNorm(hidden_channels))
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, return_embeddings=False, return_all_layers=False):
        layer_embeddings = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_embeddings.append(x)
        node_embeddings = x
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)
        out = self.lin(pooled)
        if return_all_layers:
            return out, layer_embeddings
        if return_embeddings:
            return out, node_embeddings
        return out

MODEL_REGISTRY = {
    "Baseline (no mitigation)": GATBaseline,
    "Residual connections": GATResidual,
    "LayerNorm": GATLayerNorm,
}