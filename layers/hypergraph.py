import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """
    Improved Hypergraph Convolution Layer:
    - Supports residual connections
    - Incorporates batch normalization
    - Enhances node-hyperedge interaction attention
    """

    def __init__(self, in_dim, out_dim, dropout=0.1, use_residual=True):
        super().__init__()
        self.lin_v = nn.Linear(in_dim, out_dim)  # Node feature transformation
        self.edge_proj = nn.Linear(in_dim, out_dim, bias=False)  # Hyperedge feature transformation

        # Hyperedge attention parameters
        self.edge_attn = nn.Parameter(torch.zeros(1, out_dim))
        # Node attention parameters (enhance node-hyperedge interaction)
        self.node_attn = nn.Parameter(torch.zeros(1, out_dim))

        self.bn = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.act = nn.LeakyReLU(0.2)

        self.use_residual = use_residual and (in_dim == out_dim)

        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.edge_attn)
        nn.init.xavier_uniform_(self.node_attn)

    def forward(self, X, H):
        """
        X: [N, F] Node features
        H: [N, E] Incidence matrix
        """
        N, E = H.size()
        Xv = self.lin_v(X)  # [N, out_dim] Node feature transformation

        # Calculate hyperedge features (averaged from associated node features)
        deg_e = H.sum(dim=0).clamp(min=1.0)  # [E] Hyperedge degrees
        Xe = (H.t() @ X) / deg_e.unsqueeze(-1)  # [E, F] Hyperedge features
        Xe = self.edge_proj(Xe)  # [E, out_dim] Hyperedge feature transformation

        # Calculate hyperedge attention (hyperedge importance)
        edge_attn_scores = torch.tanh(Xe @ self.edge_attn.t()).squeeze(-1)  # [E]
        edge_attn = torch.softmax(edge_attn_scores, dim=0)  # [E] Normalized hyperedge attention

        # Calculate node attention (node contribution to hyperedges)
        node_attn_scores = torch.tanh(Xv @ self.node_attn.t()).squeeze(-1)  # [N]
        node_attn = torch.softmax(node_attn_scores, dim=0)  # [N] Normalized node attention

        # Apply node attention to incidence matrix
        H_node_attn = H * node_attn.unsqueeze(1)  # [N, E]

        # Degree matrix calculation (normalization)
        Dv = (H_node_attn @ edge_attn.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)  # [N]
        Dv_inv_sqrt = Dv.pow(-0.5)  # [N] Inverse square root of node degrees

        De = (edge_attn * deg_e).clamp(min=1e-6)  # [E] Weighted hyperedge degrees
        De_inv = De.pow(-1.0)  # [E] Inverse of hyperedge degrees

        # Hypergraph convolution operation
        H_we = H_node_attn * edge_attn  # [N, E] Apply hyperedge attention
        H_we_deinv = H_we * De_inv  # [N, E] Apply hyperedge degree normalization

        # Aggregate hyperedge information
        out = H_we_deinv @ (H_node_attn.t() * Dv_inv_sqrt.unsqueeze(1))  # [N, N]
        out = out @ (Xv * Dv_inv_sqrt.unsqueeze(-1))  # [N, out_dim]

        # Residual connection
        if self.use_residual:
            out = out + Xv  # Residual addition

        # Batch normalization + activation + Dropout
        out = self.bn(out)  # Batch normalization
        out = self.act(out)  # Activation function
        out = self.dropout(out)  # Dropout

        return out


class HypergraphStack(nn.Module):
    """
    Multi-layer Hypergraph Convolutional Network Stack:
    - Supports specifying number of layers through num_layers, working with hidden_dims to control network structure
    - Flexible configuration of parameters for each layer
    """

    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.1, use_residual=True):
        super().__init__()
        # Generate hidden_dims list: if not specified, generate based on num_layers and hidden_dim
        # If user provides a list, force its length to match num_layers
        if isinstance(hidden_dim, list):
            # Ensure list length matches number of layers
            if len(hidden_dim) != num_layers:
                raise ValueError(f"hidden_dim list length ({len(hidden_dim)}) does not match num_layers ({num_layers})")
            hidden_dims = hidden_dim
        else:
            # All layers use the same hidden_dim
            hidden_dims = [hidden_dim] * num_layers

        layers = []
        prev_dim = in_dim
        for out_dim in hidden_dims:
            layers.append(
                HypergraphConv(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    use_residual=use_residual
                )
            )
            prev_dim = out_dim  # Input dimension of next layer = output dimension of current layer

        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers  # Save number of layers information

    def forward(self, X, H):
        """
        X: [N, in_dim] Initial node features
        H: [N, E] Incidence matrix
        """
        z = X
        for layer in self.layers:
            z = layer(z, H)  # Pass through each layer
        return z
