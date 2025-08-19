
import argparse

def get_config():
    p = argparse.ArgumentParser(description="HCLGT-DRP (Reproduction, Minimal Working Example)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_root", type=str, default="../data", help="Path to your data folder; if None, use synthetic data")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--num_repeats", type=float, default=5)
    p.add_argument("--num_splits", type=float, default=5)
    p.add_argument("--synthetic", type=lambda x: str(x).lower()=='true', default=True)

    # dims
    p.add_argument("--cell_dim", type=int, default=256)
    p.add_argument("--drug_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--proj", type=int, default=128)

    # hypergraph
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--num_hg_layers", type=int, default=2)
    p.add_argument("--hg_dropout", type=float, default=0.2)

    # fusion
    p.add_argument("--use_shared_unique", type=lambda x: str(x).lower()=='true', default=True)

    # association graph
    p.add_argument("--pos_w", type=float, default=3.0)
    p.add_argument("--neg_w", type=float, default=1.0)
    p.add_argument("--num_gcn_layers", type=int, default=4)
    p.add_argument("--gcn_dropout", type=float, default=0.1)

    # drug transformer
    p.add_argument("--num_tx_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--tx_dropout", type=float, default=0.1)
    p.add_argument("--distance_kernel", type=str, default="softmax", choices=["softmax", "exp"],help="Kernel for distance matrix (softmax/exp)")
    p.add_argument("--use_edge_features", action="store_true", default=True, help="Whether to use edge features in Transformer")

    # contrastive
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--contrastive_weight", type=float, default=1.0)

    return p.parse_args()

