import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import expon

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import utils


def sc_load_process(data, csv_to_load=False, h5ad_to_load=False, x10_to_load=False, txt_to_load=False, transpose=False, check_counts=True, filter_min_counts=False, min_gene_counts=1, min_cell_counts=1, logtrans_input=False, normalize_amount=False, normalize_expression_profile=False):
    logging.info('--------> Loading and Preprocessing data ...')

    if csv_to_load:
        counts = pd.read_csv(data, index_col=0)

        if transpose:
            # 一般对于adata.X，行对应观测（即，细胞），列对应特征（即，基因）
            # counts 行是基因，列是细胞
            adata = sc.AnnData(counts.values.T)
            adata.obs_names = counts.columns
            adata.var_names = counts.index
        else:
            # counts 行是细胞，列是基因
            adata = sc.AnnData(counts.values)
            adata.obs_names = counts.index
            adata.var_names = counts.columns
    elif h5ad_to_load:
        adata = sc.read_h5ad(data)
        # adata.X = adata.X.toarray().copy()
    elif x10_to_load:
        # adata = sc.read(data)
        adata = sc.read_10x_mtx(data, var_names='gene_ids')
        adata.X = adata.X.toarray().copy()
    elif txt_to_load:
        # 读取行名
        with open(data) as f:
            row_names = [line.split('\t')[0] for line in f.readlines()[1:]]

        # 读取列名
        with open(data) as f:
            col_names = f.readline().strip().split('\t')[1:]

        # 读取数据，跳过第一行和第一列
        with open(data) as f:
            ncols = len(f.readline().split('\t'))#常见的分隔符是制表符（\t）、逗号（,）
        counts = np.loadtxt(open(data, "rb"), delimiter="\t", skiprows=1, usecols=range(1, ncols))

        if transpose:
            # 一般对于adata.X，行对应观测（即，细胞），列对应特征（即，基因）
            # counts 行是基因，列是细胞
            adata = sc.AnnData(counts.T)
            adata.obs_names = pd.Index(col_names)
            adata.var_names = pd.Index(row_names)
        else:
            # counts 行是细胞，列是基因
            adata = sc.AnnData(counts)
            adata.obs_names = pd.Index(row_names)
            adata.var_names = pd.Index(col_names)

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if check_counts:

        X_subset = adata.X[:10]
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        if sp.sparse.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

        max_value = np.max(adata.X)
        if max_value < 10:
            print("ERROR: max value = {}. Is your data log-transformed? Please provide raw counts".format(max_value))
            exit(1)

        if sum(adata.obs_names.duplicated()):
            print("ERROR: duplicated cell labels. Please provide unique cell labels.")
            exit(1)

        if sum(adata.var_names.duplicated()):
            print("ERROR: duplicated gene labels. Please provide unique gene labels.")
            exit(1)

    adata.raw = adata.copy()

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=min_gene_counts)
        sc.pp.filter_cells(adata, min_counts=min_cell_counts)

    if normalize_amount:
        sc.pp.normalize_total(adata)

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_expression_profile:
        sc.pp.scale(adata)

    adata.process = adata.copy()

    if filter_min_counts:
        logging.info('--------> Successfully loaded {}:{} genes and {} cells, '
                     'preprocessed {} genes and {} cells to process.'
                     '(Keep only the genes that have counts in at least {} cells '
                     'and the cells that have a certain level of expression in at least {} gene.)'
                     .format(data, adata.raw.n_vars, adata.raw.n_obs, adata.n_vars, adata.n_obs, min_gene_counts, min_cell_counts))
    else:
        logging.info('--------> Successfully loaded {}:{} genes and {} cells'.format(data, adata.n_vars, adata.n_obs))

    return adata





def dropout(X_sc, seed, dropout):
    logging.info('Applying a random mask to the real single-cell datasets ...')

    np.random.seed(seed)
    binMask = np.ones(X_sc.shape).astype(bool)
    idx = []
    for c in range(X_sc.shape[0]):
        cells_c = X_sc.X[c, :]
        ind_pos = np.arange(X_sc.shape[1])[cells_c > 0]
        cells_c_pos = cells_c[ind_pos]

        if cells_c_pos.size > 5:
            probs = expon.pdf(cells_c_pos)
            n_masked = 1 + int(dropout * len(cells_c_pos))
            if n_masked >= cells_c_pos.size:
                print("Warning: too many cells masked for gene {} ({}/{})".format(c, n_masked, cells_c_pos.size))
                n_masked = 1 + int(0.5 * cells_c_pos.size)

            masked_idx = np.random.choice(cells_c_pos.size, n_masked, p=probs / probs.sum(), replace=False)
            binMask[c, ind_pos[sorted(masked_idx)]] = False
            idx.append(ind_pos[sorted(masked_idx)])

    dropout_info = [(i, j) for i, sub_lst in enumerate(idx) for j in sub_lst]
    X_dropout = X_sc.copy()
    for i, j in dropout_info:
        X_dropout.X[i][j] = 0

    return X_dropout, dropout_info
