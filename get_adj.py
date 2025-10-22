import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch
from sklearn.neighbors import kneighbors_graph


def get_gene_adjacency_matrix(data, n_neighbors=10):
    adata = data.copy()
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sim_matrix = np.corrcoef(adata.varm['PCs'])

    if np.isnan(sim_matrix).any() or np.isinf(sim_matrix).any() or np.max(np.abs(sim_matrix)) > np.finfo(np.float64).max:
        invalid_indices = np.where(np.isnan(sim_matrix) | np.isinf(sim_matrix) | (np.abs(sim_matrix) > np.finfo(np.float64).max))
        sim_matrix[invalid_indices] = np.mean(sim_matrix[~np.isnan(sim_matrix) & ~np.isinf(sim_matrix)])

    adj_gene_orig = kneighbors_graph(sim_matrix, n_neighbors=n_neighbors, mode='connectivity', include_self=False)

    return adj_gene_orig


def get_cell_adjacency_matrix(data, n_neighbors=10):
    adata = data.copy()
    sc.pp.scale(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    adj_cell_orig = adata.obsp['connectivities']

    return adj_cell_orig


def adj2list(adj_orig):

    adj_list = []

    row_idx, col_idx = adj_orig.nonzero()
    for i, j in zip(row_idx, col_idx):
        adj_list.append((i, j, adj_orig[i, j]))

    return adj_list


def adj_normalization_tensor(adj_orig):

    adj = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj.eliminate_zeros()

    adj_label, adj_normalization_tensor = preprocess_adj_tensor(adj)

    return adj_label, adj_normalization_tensor

def preprocess_adj_tensor(adj):
    adj_label = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_label, sparse_mx_to_torch_sparse_tensor(adj_normalized)

def normalize_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) #np.vstack在竖直方向上进行堆叠
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)











