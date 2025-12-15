from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GraphConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    LayerNorm,
    MLP,
)


MODULE_DICT = {
    "GCNConv": GCNConv,
    "GATConv": GATConv,
    "SAGEConv": SAGEConv,
    "GINConv": GINConv,
    "GraphConv": GraphConv,
    "GATv2Conv": GATv2Conv,
    "TransformerConv": TransformerConv,
}


def apply_pooling(x, pooling_type, batch=None):
    if pooling_type == "mean":
        return global_mean_pool(x, batch)
    if pooling_type == "max":
        return global_max_pool(x, batch)
    if pooling_type == "sum":
        return global_add_pool(x, batch)
    raise ValueError(f"Unknown pooling type: {pooling_type}")


class GNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.5,
        module="GCNConv",
        layer_norm=False,
        residual=False,
        pooling="mean",
        heads=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling
        self.module = module
        self.heads = heads

        self.gnn_layers = self._build_gnn_layers()
        self.layer_norms = (
            nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
            if layer_norm
            else None
        )

        self.head_proj = None
        if module in {"TransformerConv", "GATConv", "GATv2Conv"} and heads > 1:
            self.head_proj = nn.Linear(hidden_dim * heads, hidden_dim)
            classifier_in = hidden_dim
        else:
            classifier_in = hidden_dim

        self.classifier = MLP([classifier_in, classifier_in // 2, output_dim], dropout=dropout)

    def _build_gnn_layers(self):
        layers = nn.ModuleList()
        conv_class = MODULE_DICT[self.module]

        for i in range(self.num_layers):
            in_dim = self.input_dim if i == 0 else self.hidden_dim
            out_dim = self.hidden_dim

            if self.module in {"GATConv", "GATv2Conv", "TransformerConv"}:
                conv = conv_class(in_dim, out_dim, heads=self.heads, dropout=self.dropout)
            elif self.module == "GINConv":
                mlp = MLP([in_dim, self.hidden_dim, out_dim])
                conv = conv_class(mlp)
            else:
                conv = conv_class(in_dim, out_dim)

            layers.append(conv)
        return layers

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)

        for i, conv in enumerate(self.gnn_layers):
            if self.residual and i > 0:
                prev_x = x

            if self.module == "TransformerConv" and hasattr(conv, "in_channels"):
                if x.size(-1) != conv.in_channels:
                    x = F.linear(x, torch.eye(conv.in_channels, x.size(-1), device=x.device))

            if self.module in {"GATConv", "GATv2Conv", "TransformerConv"} and self.heads > 1:
                x = conv(x, edge_index)
                if hasattr(x, "size") and x.size(-1) == self.hidden_dim * self.heads:
                    if self.head_proj is None:
                        raise ValueError("head_proj expected for multi-head attention layers.")
                    x = self.head_proj(x)
            else:
                x = conv(x, edge_index)

            if self.residual and i > 0 and prev_x.size() == x.size():
                x = x + prev_x

            x = F.relu(x)

            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.layer_norm and self.layer_norms is not None:
                x = self.layer_norms[i](x)

        x = apply_pooling(x, self.pooling, batch)
        return self.classifier(x)
