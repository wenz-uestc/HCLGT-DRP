import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers.hypergraph import HypergraphStack, HypergraphConv  # 确保导入HypergraphConv
    from .layers.gcn import GCNStack
    from .layers.graph_transformer import make_model
    from .utils import info_nce, correlation_score
except ImportError:
    from layers.hypergraph import HypergraphStack, HypergraphConv
    from layers.gcn import GCNStack
    from layers.graph_transformer import make_model
    from utils import info_nce, correlation_score


class CrossOmicsAttention(nn.Module):
    """组学间交叉注意力：让每组学特征关注其他组学的重要信息"""

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
        output = self.norm(attn_output + query)  # 残差+层归一化
        return [output[i] for i in range(self.num_omics)]  # 拆分回列表


class HCLGT_DRP(nn.Module):
    def __init__(self, cfg, num_cells, num_drugs, omics_dims, atom_dim):
        super().__init__()
        self.cfg = cfg
        self.num_cells = num_cells
        self.num_drugs = num_drugs
        self.num_omics = len(omics_dims)  # 组学数量（如3个：基因、突变、拷贝数）

        # --------------------------
        # 1. 细胞系多组学编码（增强版）
        # --------------------------
        # 单组学特征转换：深层MLP + 批归一化
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

        # 组学交叉注意力
        self.cross_omics_attn = CrossOmicsAttention(
            dim=cfg.cell_dim,
            num_omics=self.num_omics,
            num_heads=cfg.num_heads,
            dropout=cfg.hg_dropout
        )

        # 超图卷积：处理每组学的图结构特征
        self.hg = HypergraphStack(
            in_dim=cfg.cell_dim,
            hidden_dim=cfg.hidden,
            num_layers=cfg.num_hg_layers,
            dropout=cfg.hg_dropout
        )

        # 动态融合机制：用MLP学习融合权重
        self.fusion_mlp = nn.Sequential(
            nn.Linear(cfg.hidden * self.num_omics, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, self.num_omics),
            nn.Softmax(dim=-1)
        )

        # 特征交互层
        self.interaction_layer = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden)
        )

        # 新增：融合邻接矩阵后的超图卷积层（用于最终特征提取）
        self.final_hg_conv = HypergraphConv(
            in_dim=cfg.hidden,
            out_dim=cfg.hidden,
            dropout=cfg.hg_dropout,
            use_residual=True  # 残差连接，保留z_final的信息
        )

        # --------------------------
        # 2. 关联分支
        # --------------------------
        self.cell_embed = nn.Parameter(torch.randn(num_cells, cfg.hidden))
        self.drug_embed = nn.Parameter(torch.randn(num_drugs, cfg.hidden))
        nn.init.xavier_uniform_(self.cell_embed)
        nn.init.xavier_uniform_(self.drug_embed)
        self.gcn = GCNStack(
            in_dim=cfg.hidden,
            hidden_dims=cfg.hidden,
            dropout=cfg.gcn_dropout
        )

        # --------------------------
        # 3. 药物分支
        # --------------------------
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

        # --------------------------
        # 4. 投影层
        # --------------------------
        self.cell_proj_u = nn.Linear(cfg.hidden, cfg.proj)
        self.cell_proj_v = nn.Linear(cfg.hidden, cfg.proj)

    # --------------------------
    # 新增：超图邻接矩阵融合逻辑
    # --------------------------
    def fuse_hypergraphs(self, H_list):
        """
        融合三个组学的超图邻接矩阵：
        - 共同非零位置：保留（取平均值）
        - 独有非零位置：直接添加
        H_list: 列表，包含3个超图邻接矩阵 [H1, H2, H3]，形状均为 [N, E]
        返回：融合后的邻接矩阵 H_fused [N, E]
        """
        # 确保输入是3个邻接矩阵
        assert len(H_list) == 3, f"需要3个组学的邻接矩阵，实际输入{len(H_list)}个"
        H1, H2, H3 = H_list[0], H_list[1], H_list[2]
        N, E = H1.shape  # 假设三个邻接矩阵形状相同 [N, E]

        # 初始化融合邻接矩阵
        H_fused = torch.zeros_like(H1)

        # 1. 找到三个矩阵的非零位置
        mask1 = H1 != 0
        mask2 = H2 != 0
        mask3 = H3 != 0

        # 2. 共同非零位置（三个矩阵都非零）：取平均值
        common_mask = mask1 & mask2 & mask3
        H_fused[common_mask] = (H1[common_mask] + H2[common_mask] + H3[common_mask]) / 3.0

        # 3. 两个矩阵非零、第三个为零的位置：取非零值的平均
        mask12 = mask1 & mask2 & ~mask3
        H_fused[mask12] = (H1[mask12] + H2[mask12]) / 2.0

        mask13 = mask1 & ~mask2 & mask3
        H_fused[mask13] = (H1[mask13] + H3[mask13]) / 2.0

        mask23 = ~mask1 & mask2 & mask3
        H_fused[mask23] = (H2[mask23] + H3[mask23]) / 2.0

        # 4. 独有非零位置（仅一个矩阵非零）：直接保留
        unique1 = mask1 & ~mask2 & ~mask3
        H_fused[unique1] = H1[unique1]

        unique2 = ~mask1 & mask2 & ~mask3
        H_fused[unique2] = H2[unique2]

        unique3 = ~mask1 & ~mask2 & mask3
        H_fused[unique3] = H3[unique3]

        return H_fused

    # --------------------------
    # 改进的多组学融合逻辑（含邻接矩阵融合）
    # --------------------------
    def forward_cell_hypergraph(self, omics_list, H_list):
        """
        流程：
        1. 单组学特征转换 → 2. 组学交叉注意力 → 3. 超图卷积 → 4. 动态融合（节点特征）→
        5. 特征交互 → 6. 融合邻接矩阵 → 7. 最终超图卷积（用融合后的邻接矩阵）
        """
        # 1. 单组学特征预处理
        omics_features = [self.omics_mlps[i](X) for i, X in enumerate(omics_list)]  # 每个 [N, cell_dim]

        # 2. 组学交叉注意力
        omics_attended = self.cross_omics_attn(omics_features)  # 每个 [N, cell_dim]

        # 3. 超图卷积（每组学单独处理）
        hg_features = [self.hg(feat, H_list[i]) for i, feat in enumerate(omics_attended)]  # 每个 [N, hidden]

        # 4. 动态融合（节点特征）
        z_fused = self.fuse_views(hg_features)  # [N, hidden]

        # 5. 特征交互与残差
        z_interacted = self.interaction_layer(z_fused)  # [N, hidden]
        z_final = z_interacted + z_fused  # [N, hidden]

        # 6. 融合超图邻接矩阵
        H_fused = self.fuse_hypergraphs(H_list)  # [N, E]

        # 7. 用融合后的邻接矩阵和z_final进行超图卷积，得到最终特征
        final_feat = self.final_hg_conv(z_final, H_fused)  # [N, hidden]

        # 投影到对比学习空间
        U = self.cell_proj_u(final_feat)  # [N, proj]

        return final_feat, U

    # --------------------------
    # 其他方法保持不变
    # --------------------------
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

    # def contrastive_loss(self, U, V, temperature):
    #     return 0.8 * info_nce(U, V, temperature) + 0.2 * info_nce(V, U, temperature)
    def contrastive_loss(self, U, V, temperature=0.1):
        """
        超图对比学习损失：增强同一细胞系的u_i与v_i的相似性，抑制负样本对的相似性。
        - U: 超图学习得到的细胞特征 [N, proj]（u_i）
        - V: 关联图学习得到的细胞特征 [N, proj]（v_i）
        - temperature: 温度系数，控制分布陡峭程度
        """
        N = U.shape[0]  # 细胞系数量

        # --------------------------
        # 1. 以U为锚点：正样本是V，负样本是所有其他U和V
        # --------------------------
        # 正样本对得分：u_i与v_i的相似度
        pos_score_U = torch.sum(U * V, dim=1)  # [N]
        # 负样本：所有细胞的U和V（排除自身）
        # 将U和V拼接为负样本池 [2N, proj]
        neg_pool_U = torch.cat([U, V], dim=0)  # [2N, proj]
        # 每个u_i与负样本池的相似度 [N, 2N]
        neg_score_U = torch.matmul(U, neg_pool_U.T)  # [N, 2N]
        # 掩码：排除自身u_i与u_i、u_i与v_i的相似性（仅保留真正负样本）
        mask_U = ~torch.eye(N, 2 * N, dtype=bool, device=U.device)  # 排除对角线（自身u_i）
        mask_U[:, N:N + N] = mask_U[:, N:N + N] & ~torch.eye(N, dtype=bool, device=U.device)  # 排除u_i与v_i
        neg_score_U = neg_score_U[mask_U].view(N, -1)  # [N, 2N-2]（每个样本有2N-2个负样本）

        # InfoNCE损失（以U为锚点）
        logits_U = torch.cat([pos_score_U.unsqueeze(1), neg_score_U], dim=1)  # [N, 1 + 2N-2]
        logits_U /= temperature
        labels_U = torch.zeros(N, dtype=torch.long, device=U.device)  # 正样本在第0列
        loss_U = F.cross_entropy(logits_U, labels_U)

        # --------------------------
        # 2. 以V为锚点：正样本是U，负样本是所有其他V和U（对称计算）
        # --------------------------
        # 正样本对得分：v_i与u_i的相似度
        pos_score_V = torch.sum(V * U, dim=1)  # [N]（与pos_score_U对称）
        # 负样本：所有细胞的V和U（排除自身）
        neg_pool_V = torch.cat([V, U], dim=0)  # [2N, proj]
        # 每个v_i与负样本池的相似度 [N, 2N]
        neg_score_V = torch.matmul(V, neg_pool_V.T)  # [N, 2N]
        # 掩码：排除自身v_i与v_i、v_i与u_i的相似性
        mask_V = ~torch.eye(N, 2 * N, dtype=bool, device=V.device)
        mask_V[:, N:N + N] = mask_V[:, N:N + N] & ~torch.eye(N, dtype=bool, device=V.device)
        neg_score_V = neg_score_V[mask_V].view(N, -1)  # [N, 2N-2]

        # InfoNCE损失（以V为锚点）
        logits_V = torch.cat([pos_score_V.unsqueeze(1), neg_score_V], dim=1)  # [N, 1 + 2N-2]
        logits_V /= temperature
        labels_V = torch.zeros(N, dtype=torch.long, device=V.device)
        loss_V = F.cross_entropy(logits_V, labels_V)

        # 总损失：对称融合两个方向的损失
        return loss_U * 0.8 + loss_V * 0.2