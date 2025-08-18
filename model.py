import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers.hypergraph import HypergraphStack, HypergraphConv  # Ensure HypergraphConv is imported
    from .layers.gcn import GCNStack
    from .layers.graph_transformer import make_model
    from .utils import info_nce, correlation_score
except ImportError:
    from layers.hypergraph import HypergraphStack, HypergraphConv
    from layers.gcn import GCNStack
    from layers.graph_transformer import make_model
    from utils import info_nce, correlation_score


class CrossOmicsAttention(nn.Module):
    """Cross-omics attention: allows each omics feature to focus on important information from other omics"""

    def __init__(self, dim, num_omics, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_omics = num_omics
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, omics_features):
        query = torch.stack(omics_features, dim=0)  # [num_omics, N, dim]
        key = value = query
        attn_output, _ = self.attn(query, key, value)  # [num_omics, N, dim]
        attn_output = self.dropout(attn_output)
        output = self.norm(attn_output + query)  # Residual connection + layer normalization
        return [output[i] for i in range(self.num_omics)]  # Split back into list


class HCLGT_DRP(nn.Module):
    def __init__(self, cfg, num_cells, num_drugs, omics_dims, atom_dim):
        super().__init__()
        self.cfg = cfg
        self.num_cells = num_cells
        self.num_drugs = num_drugs
        self.num_omics = len(omics_dims)  # Number of omics types (3)

        # Single-omics feature transformation: deep MLP + batch normalization
        self.omics_mlps = nn.ModuleList()
        for d in omics_dims:
            self.omics_mlps.append(nn.Sequential(
                nn.Linear(d, cfg.cell_dim * 2),
                nn.BatchNorm1d(cfg.cell_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.hg_dropout),
                nn.Linear(cfg.cell_dim * 2, cfg.cell_dim),
                nn.BatchNorm1d(cfg.cell_dim),
                nn.GELU()
            ))

        # Cross-omics attention
        self.cross_omics_attn = CrossOmicsAttention(
            dim=cfg.cell_dim,
            num_omics=self.num_omics,
            num_heads=cfg.num_heads,
            dropout=cfg.hg_dropout
        )

        # Hypergraph convolution: process graph-structured features for each omics
        self.hg = HypergraphStack(
            in_dim=cfg.cell_dim,
            hidden_dim=cfg.hidden,
            num_layers=cfg.num_hg_layers,
            dropout=cfg.hg_dropout
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(cfg.hidden * self.num_omics, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, self.num_omics),
            nn.Softmax(dim=-1)
        )

        # Feature interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden)
        )

        self.final_hg_conv = HypergraphConv(
            in_dim=cfg.hidden,
            out_dim=cfg.hidden,
            dropout=cfg.hg_dropout,
            use_residual=True  # Residual connection to preserve z_final information
        )

        # Association branch
        self.cell_embed = nn.Parameter(torch.randn(num_cells, cfg.hidden))
        self.drug_embed = nn.Parameter(torch.randn(num_drugs, cfg.hidden))
        nn.init.xavier_uniform_(self.cell_embed)
        nn.init.xavier_uniform_(self.drug_embed)
        self.gcn = GCNStack(
            in_dim=cfg.hidden,
            hidden_dims=cfg.hidden,
            dropout=cfg.gcn_dropout
        )

        # Drug branch
        self.atom_dim = atom_dim
        self.drug_transformer = make_model(
            d_atom=atom_dim,
            N=cfg.num_tx_layers,
            d_model=cfg.hidden,
            h=cfg.num_heads,
            dropout=cfg.tx_dropout,
            distance_matrix_kernel=cfg.distance_kernel,
            use_edge_features=cfg.use_edge_features
        )
        self.drug_proj = nn.Linear(cfg.hidden, cfg.proj)

        # Projection layers
        self.cell_proj_u = nn.Linear(cfg.hidden, cfg.proj)
        self.cell_proj_v = nn.Linear(cfg.hidden, cfg.proj)

    # Hypergraph adjacency matrix fusion
    def fuse_hypergraphs(self, H_list):
        """
        Fuse hypergraph adjacency matrices from three omics:
        - Common non-zero positions: preserve (take average)
        - Unique non-zero positions: directly add
        H_list: list containing 3 hypergraph adjacency matrices [H1, H2, H3], all shaped [N, E]
        Returns: fused adjacency matrix H_fused [N, E]
        """
        # Ensure input contains 3 adjacency matrices
        assert len(H_list) == 3, f"Expected 3 omics adjacency matrices, got {len(H_list)}"
        H1, H2, H3 = H_list[0], H_list[1], H_list[2]
        N, E = H1.shape  # Assume all three adjacency matrices have the same shape [N, E]

        # Initialize fused adjacency matrix
        H_fused = torch.zeros_like(H1)

        mask1 = H1 != 0
        mask2 = H2 != 0
        mask3 = H3 != 0

        common_mask = mask1 & mask2 & mask3
        H_fused[common_mask] = (H1[common_mask] + H2[common_mask] + H3[common_mask]) / 3.0

        mask12 = mask1 & mask2 & ~mask3
        H_fused[mask12] = (H1[mask12] + H2[mask12]) / 2.0

        mask13 = mask1 & ~mask2 & mask3
        H_fused[mask13] = (H1[mask13] + H3[mask13]) / 2.0

        mask23 = ~mask1 & mask2 & mask3
        H_fused[mask23] = (H2[mask23] + H3[mask23]) / 2.0

        unique1 = mask1 & ~mask2 & ~mask3
        H_fused[unique1] = H1[unique1]

        unique2 = ~mask1 & mask2 & ~mask3
        H_fused[unique2] = H2[unique2]

        unique3 = ~mask1 & ~mask2 & mask3
        H_fused[unique3] = H3[unique3]

        return H_fused


    def forward_cell_hypergraph(self, omics_list, H_list):

        omics_features = [self.omics_mlps[i](X) for i, X in enumerate(omics_list)]  # Each [N, cell_dim]

        omics_attended = self.cross_omics_attn(omics_features)  # Each [N, cell_dim]

        hg_features = [self.hg(feat, H_list[i]) for i, feat in enumerate(omics_attended)]  # Each [N, hidden]

        z_fused = self.fuse_views(hg_features)  # [N, hidden]

        z_interacted = self.interaction_layer(z_fused)  # [N, hidden]
        z_final = z_interacted + z_fused  # [N, hidden]

        H_fused = self.fuse_hypergraphs(H_list)  # [N, E]

        final_feat = self.final_hg_conv(z_final, H_fused)  # [N, hidden]

        U = self.cell_proj_u(final_feat)  # [N, proj]

        return final_feat, U

    def fuse_views(self, Z_list):
        N = Z_list[0].shape[0]
        z_concat = torch.cat(Z_list, dim=1)  # [N, hidden * num_omics]
        weights = self.fusion_mlp(z_concat)  # [N, num_omics]
        z_fused = torch.zeros_like(Z_list[0])
        for i in range(self.num_omics):
            z_fused += weights[:, i:i + 1] * Z_list[i]
        return z_fused

    def forward_association(self, A_full):
        X0 = torch.cat([self.cell_embed, self.drug_embed], dim=0)
        Z_all = self.gcn(X0, A_full)
        Zc, Zd = Z_all[:self.num_cells], Z_all[self.num_cells:]
        V = self.cell_proj_v(Zc)
        return Zc, Zd, V

    def forward_drug(self, atom_feats, adj, dist, edge_feat=None):
        src_mask = (atom_feats.sum(dim=-1) != 0).float()
        atom_feats = atom_feats.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        adj = adj.unsqueeze(0)
        dist = dist.unsqueeze(0)

        if edge_feat is not None:
            edge_feat = edge_feat.unsqueeze(0)
            edge_feats = edge_feat.permute(0, 3, 1, 2)  # [1, 9, N, N]
        else:
            edge_feats = None

        drug_emb = self.drug_transformer(
            src=atom_feats,
            src_mask=src_mask,
            adj_matrix=adj,
            distances_matrix=dist,
            edges_att=edge_feats
        )
        drug_emb = drug_emb.squeeze(0)
        return self.drug_proj(drug_emb)

    def forward_pair(self, cell_emb, drug_emb):
        s = correlation_score(cell_emb, drug_emb)
        return torch.sigmoid(s)

    def contrastive_loss(self, U, V, temperature=0.1):
        """
        Hypergraph contrastive learning loss: enhances similarity between u_i and v_i of the same cell line,
        suppresses similarity of negative sample pairs.
        - U: Cell features from hypergraph learning [N, proj] (u_i)
        - V: Cell features from association graph learning [N, proj] (v_i)
        - temperature: Temperature coefficient controlling distribution steepness
        """
        N = U.shape[0]  # Number of cell lines

        # With U as anchor: positive samples are V, negative samples are all other U and V
        # Positive pair scores: similarity between u_i and v_i
        pos_score_U = torch.sum(U * V, dim=1)  # [N]
        # Negative samples: all cells' U and V (excluding self)
        # Concatenate U and V into negative sample pool [2N, proj]
        neg_pool_U = torch.cat([U, V], dim=0)  # [2N, proj]
        # Similarity between each u_i and negative sample pool [N, 2N]
        neg_score_U = torch.matmul(U, neg_pool_U.T)  # [N, 2N]
        # Mask: exclude similarity between u_i and u_i, u_i and v_i (keep only true negatives)
        mask_U = ~torch.eye(N, 2 * N, dtype=bool, device=U.device)  # Exclude diagonal (self u_i)
        mask_U[:, N:N + N] = mask_U[:, N:N + N] & ~torch.eye(N, dtype=bool, device=U.device)  # Exclude u_i and v_i
        neg_score_U = neg_score_U[mask_U].view(N, -1)  # [N, 2N-2] (each sample has 2N-2 negative samples)

        # InfoNCE loss (with U as anchor)
        logits_U = torch.cat([pos_score_U.unsqueeze(1), neg_score_U], dim=1)  # [N, 1 + 2N-2]
        logits_U /= temperature
        labels_U = torch.zeros(N, dtype=torch.long, device=U.device)  # Positive samples in column 0
        loss_U = F.cross_entropy(logits_U, labels_U)

        # With V as anchor: positive samples are U, negative samples are all other V and U (symmetric calculation)
        # Positive pair scores: similarity between v_i and u_i
        pos_score_V = torch.sum(V * U, dim=1)  # [N] (symmetric to pos_score_U)
        # Negative samples: all cells' V and U (excluding self)
        neg_pool_V = torch.cat([V, U], dim=0)  # [2N, proj]
        # Similarity between each v_i and negative sample pool [N, 2N]
        neg_score_V = torch.matmul(V, neg_pool_V.T)  # [N, 2N]
        # Mask: exclude similarity between v_i and v_i, v_i and u_i
        mask_V = ~torch.eye(N, 2 * N, dtype=bool, device=V.device)
        mask_V[:, N:N + N] = mask_V[:, N:N + N] & ~torch.eye(N, dtype=bool, device=V.device)
        neg_score_V = neg_score_V[mask_V].view(N, -1)  # [N, 2N-2]

        # InfoNCE loss (with V as anchor)
        logits_V = torch.cat([pos_score_V.unsqueeze(1), neg_score_V], dim=1)  # [N, 1 + 2N-2]
        logits_V /= temperature
        labels_V = torch.zeros(N, dtype=torch.long, device=V.device)
        loss_V = F.cross_entropy(logits_V, labels_V)

        return loss_U * 0.8 + loss_V * 0.2
