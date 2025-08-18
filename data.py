import os, json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List, Optional
from rdkit import Chem
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType  # For bond type extraction


# ---------- Molecular featurization functions (added edge features) ----------
def one_hot_vector(val, lst):
    """Convert value to one-hot vector"""
    if val not in lst:
        val = lst[-1]  # Map unknown values to the last category
    return [x == val for x in lst]


def fetch_smiles_from_pubchem(cid: int) -> str:
    """Fetch SMILES string from PubChem"""
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/TXT'
    try:
        import requests
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text.strip()
        return None
    except Exception:
        return None


def get_atom_features(atom, one_hot_formal_charge=True):
    """Extract atom features (unchanged)"""
    attributes = []
    # Atom type (common elements + others)
    attributes += one_hot_vector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    # Number of neighboring atoms
    attributes += one_hot_vector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    # Number of hydrogen atoms
    attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    # Formal charge
    if one_hot_formal_charge:
        attributes += one_hot_vector(atom.GetFormalCharge(), [-1, 0, 1])
    else:
        attributes.append(atom.GetFormalCharge())
    # Ring information and aromaticity
    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())
    return np.array(attributes, dtype=np.float32)


def get_bond_features(bond):
    """Extract edge (bond) features and encode as vector"""
    features = []
    bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    features += one_hot_vector(bond.GetBondType(), bond_types)  # 4 dimensions

    features.append(bond.GetIsConjugated())  # 1 dimension

    features.append(bond.IsInRing())  # 1 dimension

    stereo = bond.GetStereo()
    stereo_feat = 0 if stereo == 0 else 1 if stereo == 1 else 2  # Simplified encoding
    features += one_hot_vector(stereo_feat, [0, 1, 2])  # 3 dimensions
    return np.array(features, dtype=np.float32)  # Total dimensions: 4+1+1+3=9


def featurize_mol(mol, add_dummy_node=True, one_hot_formal_charge=True):
    """
    Extract node features, adjacency matrix, distance matrix, and edge features of a molecule
    Returns: node_feats [N, F], adj [N, N], dist [N, N], edge_feats [N, N, E]
    """
    N = mol.GetNumAtoms()  # Number of nodes

    # Node features (atom features)
    node_feats = np.array([get_atom_features(a, one_hot_formal_charge) for a in mol.GetAtoms()])  # [N, F_atom]

    # Adjacency matrix (whether a bond exists)
    adj = np.zeros((N, N), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0  # Undirected graph, symmetric

    # Distance matrix (3D spatial distance)
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                   for k in range(N)])
    dist = pairwise_distances(pos)  # [N, N]

    # Edge features (bond features)
    E = 9  # Dimensionality of edge features (defined by get_bond_features)
    edge_feats = np.zeros((N, N, E), dtype=np.float32)  # Initialize with zero vectors
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_feats[i, j] = feat  # Undirected graph, symmetric
        edge_feats[j, i] = feat

    # Handle dummy node (if needed)
    if add_dummy_node:
        # Node features: add a new row (dummy node), first element is 1 (marks dummy node)
        node_feats = np.pad(node_feats, ((1, 0), (1, 0)), mode='constant')
        node_feats[0, 0] = 1.0  # Marker feature for dummy node

        # Adjacency matrix: add a new row and column (dummy node has no connections)
        adj = np.pad(adj, ((1, 0), (1, 0)), mode='constant')

        # Distance matrix: set distance from dummy node to others to a large value
        dist = np.pad(dist, ((1, 0), (1, 0)), mode='constant', constant_values=1e6)

        # Edge features: add a new row and column (no edge features for dummy node)
        edge_feats = np.pad(edge_feats, ((1, 0), (1, 0), (0, 0)), mode='constant')

    return node_feats, adj, dist, edge_feats


def _read_feature_csv(path: str, index_col: int = 0) -> Tuple[np.ndarray, list]:
    df = pd.read_csv(path)
    if index_col is not None:
        ids = df.iloc[:, index_col].astype(str).tolist()
        X = df.drop(df.columns[index_col], axis=1).to_numpy(dtype=np.float32)
    else:
        ids = df.columns.astype(str).tolist()
        X = df.to_numpy(dtype=np.float32)
    return X, ids


def _align_by_ids(base_ids: list, feat_ids: list, X: np.ndarray) -> np.ndarray:
    dim = X.shape[1]
    id2row = {k: i for i, k in enumerate(feat_ids)}
    out = np.zeros((len(base_ids), dim), dtype=np.float32)
    for i, k in enumerate(base_ids):
        j = id2row.get(str(k), None)
        if j is not None:
            out[i] = X[j]
    return out


def torch_euclidean_dist(tensor: torch.Tensor, dim=0):
    if dim:
        tensor_mul_tensor = torch.mm(torch.t(tensor), tensor)
    else:
        tensor_mul_tensor = torch.mm(tensor, torch.t(tensor))
    diag = torch.diag(tensor_mul_tensor)
    n_diag = diag.size()[0]
    tensor_diag = diag.repeat([n_diag, 1])
    diag = diag.view([n_diag, -1])
    dist = torch.sub(torch.add(tensor_diag, diag), torch.mul(tensor_mul_tensor, 2))
    dist = torch.sqrt(dist)
    return dist


def torch_dist(tensor: torch.Tensor, p=0 or int):
    size = tensor.size()
    tensor_flatten = torch.flatten(tensor)
    tensor_mat = tensor.repeat([1, 1, size[0]])
    tensor_flatten = tensor_flatten.repeat([1, size[0], 1])
    tensor_sub = torch.sub(tensor_mat, tensor_flatten)
    tensor_sub = tensor_sub.view([size[0], size[0], size[1]])
    tensor_sub = torch.abs(tensor_sub)
    if p == 0:
        tensor_sub = torch.pow(tensor_sub, p)
        dist = torch.sum(tensor_sub, dim=2)
        diag = torch.diag(dist)
        dist = torch.sub(dist, torch.diag(diag))
    elif p == 1:
        dist = torch.sum(tensor_sub, dim=2)
    else:
        tensor_sub = torch.pow(tensor_sub, p)
        dist = torch.sum(tensor_sub, dim=2)
        dist = torch.pow(dist, 1/p)
    return dist


def torch_z_normalized(tensor: torch.Tensor, dim=0):
    mean = torch.mean(tensor, dim=1-dim)
    std = torch.std(tensor, dim=1-dim)
    if dim:
        tensor_sub_mean = torch.sub(tensor, mean)
        tensor_normalized = torch.div(tensor_sub_mean, std)
    else:
        size = mean.size()[0]
        tensor_sub_mean = torch.sub(tensor, mean.view([size, -1]))
        tensor_normalized = torch.div(tensor_sub_mean, std.view([size, -1]))
    return tensor_normalized


def exp_similarity(tensor: torch.Tensor, sigma: torch.Tensor, normalize=True):
    if normalize:
        tensor = torch_z_normalized(tensor, dim=1)
    tensor_dist = torch_euclidean_dist(tensor, dim=0)
    exp_dist = torch.exp(-tensor_dist/(2*torch.pow(sigma, 3)))
    return exp_dist


def hyper_sim(feat, bin_mat):
    sigma = 3
    sima = torch.from_numpy(feat).to('cuda:0')
    sigma = torch.tensor(sigma, dtype=torch.float, device='cuda:0')
    sima = exp_similarity(sima, sigma)
    # High-order similarity
    a1 = torch.mm(sima,sima)
    A1 = torch.mul(a1,sima)
    A2 = torch.mul(torch.mm(bin_mat,bin_mat.T),sima)
    A3= torch.mm(torch.mm(sima,bin_mat),torch.mm(sima,bin_mat).T)
    return A1+A2+A3


def knn_incidence(X: np.ndarray, k: int = 10, bin_mat: np.ndarray = None) -> torch.Tensor:
    N = X.shape[0]
    if bin_mat is None:
        bin_mat = np.eye(X.shape[1])  # Assume feature dimension is M=D
    bin_mat_tensor = torch.from_numpy(bin_mat).to('cuda:0', dtype=torch.float32)
    # Calculate high-order similarity matrix
    S = hyper_sim(X, bin_mat_tensor)  # [N, N]
    # Exclude self-similarity
    S.fill_diagonal_(-torch.inf)
    # Build hypergraph incidence matrix H
    H = torch.zeros((N, N), dtype=torch.float32, device='cuda:0')
    k = min(k, max(1, N - 1))
    for i in range(N):
        _, idx = torch.topk(S[i], k=k)
        H[idx, i] = 1.0
    return torch.tensor(H, dtype=torch.float32)


def build_bipartite_adj(num_cells: int, num_drugs: int, edges: List[Tuple[int, int, int]], pos_w: float = 3.0,
                        neg_w: float = 1.0) -> torch.Tensor:
    N = num_cells + num_drugs
    A = torch.zeros((N, N), dtype=torch.float32)
    A += torch.eye(N)
    for c, d, y in edges:
        w = pos_w if int(y) == 1 else neg_w
        u = c
        v = num_cells + d
        A[u, v] = A[v, u] = w
    return A


def load_processed_data(data_root: str):
    """Load data and include edge features of drug graphs"""
    bin_path = os.path.join(data_root, "cell_drug_binary.csv")
    ic50_path = os.path.join(data_root, "cell_drug.csv")
    gene_path = os.path.join(data_root, "gene_feature.csv")
    mut_path = os.path.join(data_root, "mutation_feature.csv")
    cna_path = os.path.join(data_root, "cna_feature.csv")
    drug_feat_path = os.path.join(data_root, "drug_feature.csv")
    drug_name_cid_path = os.path.join(data_root, "drug_name_cid.csv")

    # Load bipartite graph matrix
    bin_df = pd.read_csv(bin_path, index_col=0)
    cell_ids = [str(c) for c in bin_df.index.tolist()]
    drug_ids = [str(i) for i in bin_df.columns.tolist()]
    bin_mat = bin_df.to_numpy(dtype=np.float32)

    ic50 = None
    if os.path.exists(ic50_path):
        ic50_df = pd.read_csv(ic50_path, index_col=0)
        ic50 = ic50_df.reindex(index=cell_ids, columns=drug_ids).to_numpy(dtype=np.float32)

    # Load omics data
    X_gene_raw, gene_ids = _read_feature_csv(gene_path, index_col=0)
    X_mut_raw, mut_ids = _read_feature_csv(mut_path, index_col=0)
    X_cna_raw, cna_ids = _read_feature_csv(cna_path, index_col=0)
    X_gene = _align_by_ids(cell_ids, gene_ids, X_gene_raw)
    X_mut = _align_by_ids(cell_ids, mut_ids, X_mut_raw)
    X_cna = _align_by_ids(cell_ids, cna_ids, X_cna_raw)

    # Load drug fingerprint features
    drug_df = pd.read_csv(drug_feat_path)
    drug_feat_ids = drug_df.iloc[:, 0].astype(str).tolist()
    drug_feat = drug_df.drop(drug_df.columns[0], axis=1).to_numpy(dtype=np.float32)
    drug_feat = _align_by_ids(drug_ids, drug_feat_ids, drug_feat)

    # Load drug graph data (including edge features)
    drug_graph_data = {}
    if os.path.exists(drug_name_cid_path):
        drug_name_cid_df = pd.read_csv(drug_name_cid_path)
        for idx, row in drug_name_cid_df.iterrows():
            drug_name = str(row['drug_name'])
            cid = int(row['cid']) if 'cid' in row else None
            if not cid:
                continue
            # Get SMILES and convert to molecular structure
            smiles = fetch_smiles_from_pubchem(cid)
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            # Generate 3D conformation
            mol = Chem.AddHs(mol)
            embed_success = AllChem.EmbedMolecule(mol, maxAttempts=5000)
            if embed_success == 0:  # Conformation generation successful
                AllChem.UFFOptimizeMolecule(mol)  # Optimize conformation
            mol = Chem.RemoveHs(mol)  # Remove hydrogen atoms
            # Extract features (including edge features)
            node_feats, adj, dist, edge_feats = featurize_mol(mol, add_dummy_node=True)
            # Store as tuple (node features, adjacency matrix, distance matrix, edge features)
            drug_graph_data[drug_name] = (
                node_feats,
                adj,
                dist,
                edge_feats
            )
    print(drug_graph_data)

    # Generate training pairs
    pairs = []
    num_drugs = len(drug_ids)
    num_cells = len(cell_ids)
    for i in range(num_cells):
        for j in range(num_drugs):
            y = bin_mat[i, j]
            if np.isnan(y):
                continue
            pairs.append((i, j, int(y)))

    return {
        "X_gne": X_gene, "X_mut": X_mut, "X_cna": X_cna,
        "drug_feat": drug_feat,
        "drug_graph_data": drug_graph_data,  # Each entry includes edge features
        "cell_ids": cell_ids, "drug_ids": drug_ids,
        "pairs": pairs,
        "bin_mat": bin_mat,
        "ic50": ic50
    }