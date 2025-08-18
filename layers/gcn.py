import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True, dropout=0.1, activation=F.leaky_relu):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.residual = in_dim == out_dim

    def forward(self, X, A):
        """
        X: Node feature matrix, shape [num_nodes, in_dim]
        A: Adjacency matrix, shape [num_nodes, num_nodes]
        """
        A_with_self_loop = A + torch.eye(A.size(0), device=A.device)  # A + I

        deg = A_with_self_loop.sum(dim=-1).clamp(min=1e-6)  # Node degrees (sum of each row)
        D_inv_sqrt = torch.pow(deg, -0.5).unsqueeze(1)  # Inverse square root of degrees, shape [num_nodes, 1]
        A_hat = A_with_self_loop * D_inv_sqrt * D_inv_sqrt.T  # Symmetric normalization: D^{-1/2} (A+I) D^{-1/2}

        h = A_hat @ X  # Aggregate neighbor features: multiply normalized (A+I) with feature matrix
        h = self.lin(h)  # Linear transformation

        if self.residual:
            h = h + X  # Residual addition: h = W*h + X

        h = self.bn(h)  # Batch normalization (more stable when applied before activation)
        if self.activation is not None:
            h = self.activation(h, 0.2) if self.activation == F.leaky_relu else self.activation(h)
        h = self.dropout(h)

        return h


class GCNStack(nn.Module):
    def __init__(self, in_dim, hidden_dims, use_bias=True, dropout=0.1, activation=F.leaky_relu):
        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]  # Convert to single-layer list if input is a single value

        # Build multi-layer GCN
        layers = []
        prev_dim = in_dim
        for out_dim in hidden_dims:
            layers.append(
                GCNLayer(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    use_bias=use_bias,
                    dropout=dropout,
                    activation=activation
                )
            )
            prev_dim = out_dim  # Input dimension of next layer = output dimension of current layer

        self.layers = nn.ModuleList(layers)

    def forward(self, X, A):
        """
        X: Initial node feature matrix, shape [num_nodes, in_dim]
        A: Adjacency matrix, shape [num_nodes, num_nodes]
        """
        z = X
        for layer in self.layers:
            z = layer(z, A)  # Pass features through each layer
        return z
