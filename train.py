import os, sys, random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # Import weighted sampler
from config import get_config
from data import knn_incidence, build_bipartite_adj, load_processed_data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, average_precision_score
from model import HCLGT_DRP
from utils import set_seed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class PairDataset(Dataset):
    def __init__(self, pairs, drug_id_to_idx):
        self.pairs = [(cell_idx, drug_id_to_idx[drug_id], resp)
                      for cell_idx, drug_id, resp in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cell_idx, drug_idx, resp = self.pairs[idx]
        return torch.tensor(cell_idx), torch.tensor(drug_idx), torch.tensor(resp)


def collate_pairs_features(batch, cell_embs, drug_emb_map, device):
    cells = batch[0].to(device)
    drugs = batch[1].to(device)
    labels = batch[2].float().to(device)

    u = cell_embs[cells]
    v = torch.stack([drug_emb_map[did.item()] for did in drugs])

    return cells, drugs, labels, u, v


def get_drug_embeddings(model, drug_graph_data, drug_id_to_idx, device, is_training=False):
    embeddings = {}
    # Training mode: keep model in train mode, enable gradient calculation
    # Validation mode: switch to eval mode, disable gradient calculation
    if is_training:
        model.train()  # Ensure model is in training mode (enables dropout etc.)
        context = torch.enable_grad()  # Enable gradients
    else:
        model.eval()
        context = torch.no_grad()  # Disable gradients

    with context:
        for drug_id_str, graph_data in drug_graph_data.items():
            drug_idx = drug_id_to_idx[drug_id_str]

            atom_feats = torch.from_numpy(graph_data[0]).clone().detach().requires_grad_(False).to(device,
                                                                                                   dtype=torch.float32)
            adj = torch.from_numpy(graph_data[1]).clone().detach().requires_grad_(False).to(device, dtype=torch.float32)

            dist = None
            if graph_data[2] is not None:
                dist = torch.from_numpy(graph_data[2]).clone().detach().requires_grad_(False).to(device,
                                                                                                 dtype=torch.float32)
            edge_feat = None
            if len(graph_data) > 3 and graph_data[3] is not None:
                edge_feat = torch.from_numpy(graph_data[3]).clone().detach().requires_grad_(False).to(device,
                                                                                                      dtype=torch.float32)
            drug_emb = model.forward_drug(atom_feats=atom_feats, adj=adj, dist=dist, edge_feat=edge_feat)
            embeddings[drug_idx] = drug_emb
    return embeddings


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    if cfg.data_root is None:
        raise ValueError("Please set --data_root to your 'processed_data' folder.")

    # Load data
    dp = load_processed_data(cfg.data_root)
    X_gene, X_mut, X_cna = dp["X_gne"], dp["X_mut"], dp["X_cna"]
    drug_feat = dp["drug_feat"]
    pairs = dp["pairs"]
    num_cells = X_gene.shape[0]
    res = dp["bin_mat"]
    # print(dp["drug_graph_data"])
    # Drug graph data
    drug_graph_data = dp.get("drug_graph_data", {})
    if not drug_graph_data:
        raise ValueError("No drug graph data provided!")

    # Create drug ID to index mapping
    sorted_drug_ids = sorted(drug_graph_data.keys())
    num_drugs = len(sorted_drug_ids)
    drug_id_to_idx = {did: idx for idx, did in enumerate(sorted_drug_ids)}
    idx_to_drug_id = {idx: did for did, idx in drug_id_to_idx.items()}

    # Convert to tensors
    X_gene_t = torch.tensor(X_gene, dtype=torch.float32, device=device)
    X_mut_t = torch.tensor(X_mut, dtype=torch.float32, device=device)
    X_cna_t = torch.tensor(X_cna, dtype=torch.float32, device=device)

    # Parameters
    num_repeats = cfg.num_repeats
    num_splits = cfg.num_splits

    pairs_np = np.array(pairs)
    all_repeat_metrics = []
    all_fold_results = []

    for repeat in range(num_repeats):
        print(f"\n===== Repeat {repeat + 1}/{num_repeats} =====")
        np.random.shuffle(pairs_np)
        folds = np.array_split(pairs_np, num_splits)

        fold_metrics = []

        for fold_idx in range(num_splits):
            print(f"\n--- Fold {fold_idx + 1}/{num_splits} ---")
            val_pairs = folds[fold_idx].tolist()
            train_pairs = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx]).tolist()

            # Convert drug indices to string IDs
            train_pairs = [(c, idx_to_drug_id[d], r) for c, d, r in train_pairs]
            val_pairs = [(c, idx_to_drug_id[d], r) for c, d, r in val_pairs]

            # Create training mask
            train_mask = np.zeros(res.shape, dtype=bool)
            for cell_idx, drug_id, _ in train_pairs:
                drug_idx = drug_id_to_idx[drug_id]
                train_mask[cell_idx, drug_idx] = True
            train_res = res.copy()
            train_res[~train_mask] = 0

            # Build hypergraphs
            H_gne = knn_incidence(X_gene, k=cfg.knn_k, bin_mat=train_res).to(device)
            H_mut = knn_incidence(X_mut, k=cfg.knn_k, bin_mat=train_res).to(device)
            H_cna = knn_incidence(X_cna, k=cfg.knn_k, bin_mat=train_res).to(device)

            # Create datasets
            train_ds = PairDataset(train_pairs, drug_id_to_idx)
            val_ds = PairDataset(val_pairs, drug_id_to_idx)

            # Calculate sample weights to handle class imbalance
            train_labels = np.array([r for _, _, r in train_ds.pairs])
            n_pos = np.sum(train_labels == 1)
            n_neg = len(train_labels) - n_pos

            # Weight = total samples / (number of classes * class samples)
            pos_weight = len(train_labels) / (2 * n_pos)  # Positive class weight
            neg_weight = len(train_labels) / (2 * n_neg)  # Negative class weight
            sample_weights = np.where(train_labels == 1, pos_weight, neg_weight)
            sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

            # Create weighted sampler (samples based on weights to balance batches)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_ds),  # Number of samples equals training set size
                replacement=True  # Allow replacement (ensures minority classes are sufficiently selected)
            )
            # Create training data loader with weighted sampler
            train_dl = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                sampler=sampler,  # Enable weighted sampling
                shuffle=False  # Sampler already implements randomization, disable shuffle
            )
            val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

            # Build bipartite adjacency matrix
            train_pairs_with_idx = [(c, drug_id_to_idx[d], r) for c, d, r in train_pairs]
            train_bipartite_adj = build_bipartite_adj(
                num_cells, num_drugs, train_pairs_with_idx,
                pos_w=cfg.pos_w, neg_w=cfg.neg_w
            ).to(device)

            # Initialize model
            first_drug_graph = drug_graph_data[sorted_drug_ids[0]]
            atom_dim = first_drug_graph[0].shape[1]

            model = HCLGT_DRP(
                cfg,
                num_cells=num_cells,
                num_drugs=num_drugs,
                omics_dims=[X_gene.shape[1], X_mut.shape[1], X_cna.shape[1]],
                atom_dim=atom_dim,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
            bce = nn.BCELoss()

            best_auc, best_acc, best_f1, best_precision = -1, 0, 0, 0

            for epoch in range(1, cfg.epochs + 1):
                model.train()
                epoch_loss = 0.0

                for batch in train_dl:
                    # Recalculate cell embeddings and association embeddings for each batch (ensures fresh computation graph)
                    Zc_hg, U = model.forward_cell_hypergraph([X_gene_t, X_mut_t, X_cna_t], [H_gne, H_mut, H_cna])
                    Zc_gcn, Zd_gcn, V = model.forward_association(train_bipartite_adj)
                    # Recalculate drug embeddings for each batch (ensures fresh gradients)
                    drug_emb_map = get_drug_embeddings(model, drug_graph_data, drug_id_to_idx, device, is_training=True)

                    # Process current batch
                    cells, drugs, labels, u_b, v_b = collate_pairs_features(batch, U, drug_emb_map, device)
                    probs = model.forward_pair(u_b, v_b)
                    L_sup = bce(probs, labels)
                    L_con = model.contrastive_loss(U, V, temperature=cfg.temperature) * cfg.contrastive_weight
                    loss = L_sup + L_con

                    # Backpropagation
                    opt.zero_grad()  # Clear gradients from previous batch
                    loss.backward()  # Calculate gradients for current batch's computation graph
                    opt.step()  # Update parameters

                    epoch_loss += loss.item()

                    # Clean up current batch's computation graph
                    del U, V, drug_emb_map, probs, L_sup, L_con, loss
                    torch.cuda.empty_cache()  # Release GPU memory

                # Validation
                model.eval()
                with torch.no_grad():
                    # Recompute cell embeddings
                    _, U_full = model.forward_cell_hypergraph(
                        [X_gene_t, X_mut_t, X_cna_t],
                        [H_gne, H_mut, H_cna]
                    )

                    # Recompute drug embeddings
                    val_drug_emb_map = get_drug_embeddings(model, drug_graph_data, drug_id_to_idx, device,
                                                           is_training=False)
                    y_true, y_prob = [], []
                    for batch in val_dl:
                        cells, drugs, labels, u_b, v_b = collate_pairs_features(
                            batch, U_full, val_drug_emb_map, device
                        )
                        probs = model.forward_pair(u_b, v_b)
                        y_true.append(labels.cpu())
                        y_prob.append(probs.cpu())

                    y_true = torch.cat(y_true).numpy()
                    y_prob = torch.cat(y_prob).numpy()

                    best_val_f1 = 0.0
                    best_threshold = 0.5
                    for threshold in np.arange(0.1, 0.9, 0.05):
                        y_pred_threshold = (y_prob >= threshold).astype(np.int32)
                        current_f1 = f1_score(y_true, y_pred_threshold, average='macro', zero_division=0)
                        if current_f1 > best_val_f1:
                            best_val_f1 = current_f1
                            best_threshold = threshold

                    # Calculate metrics with optimal threshold
                    y_pred = (y_prob >= best_threshold).astype(np.int32)

                    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    precision = precision_score(y_true, y_pred, average='macro')

                    if auc > best_auc:
                        best_auc, best_acc, best_f1, best_precision = auc, acc, f1, precision

                print(f"Epoch {epoch}/{cfg.epochs} | Train Loss: {epoch_loss / len(train_dl):.5f} | "
                      f"Val AUC: {auc:.5f} | Val ACC: {acc:.5f} | Val Precision: {precision:.5f} | Val F1: {f1:.5f}")

            print(f"[Fold Best] AUC={best_auc:.5f} ACC={best_acc:.5f} Precision={best_precision:.5f} F1={best_f1:.5f}")
            fold_metrics.append((best_auc, best_acc, best_precision, best_f1))

            all_fold_results.append({
                "Repeat": repeat + 1,
                "Fold": fold_idx + 1,
                "AUC": best_auc,
                "Accuracy": best_acc,
                "Precision": best_precision,
                "F1": best_f1
            })

        # Calculate average metrics for the repeat experiment
        fold_metrics = np.array(fold_metrics)
        mean_metrics = fold_metrics.mean(axis=0)
        print(f"\n[Repeat {repeat + 1} Avg] AUC={mean_metrics[0]:.5f} ACC={mean_metrics[1]:.5f} "
              f"Precision={mean_metrics[2]:.5f} F1={mean_metrics[3]:.5f}")
        all_repeat_metrics.append(mean_metrics)

    # Calculate mean and standard deviation across all folds
    final_metrics = np.array(all_repeat_metrics).mean(axis=0)
    final_std = np.array(all_repeat_metrics).std(axis=0)

    # Add mean and standard deviation to results list
    all_fold_results.append({
        "Repeat": "mean",
        "Fold": "",
        "AUC": final_metrics[0],
        "Accuracy": final_metrics[1],
        "Precision": final_metrics[2],
        "F1": final_metrics[3]
    })
    all_fold_results.append({
        "Repeat": "std",
        "Fold": "",
        "AUC": final_std[0],
        "Accuracy": final_std[1],
        "Precision": final_std[2],
        "F1": final_std[3]
    })

    df = pd.DataFrame(all_fold_results)
    df.to_excel("model_metrics.xlsx", index=False, engine="openpyxl")
    print(f"\nResults saved to xlsx")

    # Output final results
    print("\n===== Final Results =====")
    print(f"AUC: {final_metrics[0]:.4f} ± {final_std[0]:.4f}")
    print(f"ACC: {final_metrics[1]:.4f} ± {final_std[1]:.4f}")
    print(f"Precision: {final_metrics[2]:.4f} ± {final_std[2]:.4f}")
    print(f"F1: {final_metrics[3]:.4f} ± {final_std[3]:.4f}")

if __name__ == "__main__":
    main()