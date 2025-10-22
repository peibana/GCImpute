
# 只细胞方向 batch（Gene 全量，Cell 分批）


import logging
import os

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import get_adj
import utils
import model

import loss





def Graph_AE_handler(X_dropout, args, param):

    logging.info('--------> Starting Graph plus AE (cells batching, sparse subgraph) ...')

    device = param['device']
    use_GAT = args.graph_AE_use_GAT
    learning_rate = args.graph_AE_learning_rate
    factor = args.graph_AE_factor
    total_epoch = args.graph_AE_epoch
    patience = args.graph_AE_patience

    dropout_gene = args.graph_Gene_dropout
    gat_multi_heads_gene = args.graph_Gene_gat_multi_heads
    multi_heads_gene = args.ae_multi_heads_gene
    hid_embed1_gene = args.graph_Gene_hid_embed1
    hid_embed2_gene = args.graph_Gene_hid_embed2

    dropout_cell = args.graph_Cell_dropout
    gat_multi_heads_cell = args.graph_Cell_gat_multi_heads
    multi_heads_cell = args.ae_multi_heads_cell
    hid_embed1_cell = args.graph_Cell_hid_embed1
    hid_embed2_cell = args.graph_Cell_hid_embed2

    # 细胞 batch 尺寸
    Bc = args.graph_AE_cell_batch_size      # e.g. 512/1024
    logging.info(f"[Batch Config] Cell batch size (Bc): {Bc}")

    use_checkpoint = args.use_checkpoint

    output_dir = os.path.join(args.output_dir, 'impute')
    os.makedirs(output_dir, exist_ok=True)
    # 缓存文件路径
    gene_adj_file = os.path.join(output_dir, "adj_gene.pt")
    cell_adj_file = os.path.join(output_dir, "adj_cell.pt")

    # ===================== 全局邻接（一次性构建/读取） =====================
    if os.path.exists(gene_adj_file) and os.path.exists(cell_adj_file):
        logging.info("✅ Found cached adjacency matrices, loading from file ...")
        adj_gene_csr = torch.load(gene_adj_file)
        adj_cell_csr = torch.load(cell_adj_file)
        logging.info(f"Loaded gene adjacency {type(adj_gene_csr)} | cell adjacency {type(adj_cell_csr)}")
    else:
        logging.info("⚡ Cached adjacency not found, computing adjacency matrices ...")
        # gene 全局邻接（scipy.csr）
        adj_gene_csr = get_adj.get_gene_adjacency_matrix(X_dropout)  # (G×G) csr
        # cell 全局邻接（scipy.csr）
        adj_cell_csr = get_adj.get_cell_adjacency_matrix(X_dropout)  # (N×N) csr

        # 保存缓存
        torch.save(adj_gene_csr, gene_adj_file)
        torch.save(adj_cell_csr, cell_adj_file)
        logging.info("✅ Adjacency matrices computed and saved for future runs.")

    X_all = X_dropout.X    # (N×G) numpy
    N, G = X_all.shape

    graph_AE = model.Graph_AE(
        N, G,
        hid_embed1_gene, hid_embed2_gene, dropout_gene, gat_multi_heads_gene, multi_heads_gene,
        hid_embed1_cell, hid_embed2_cell, dropout_cell, gat_multi_heads_cell, multi_heads_cell, use_checkpoint
    ).to(device)

    optimizer = optim.Adam(graph_AE.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    X_recon_full = torch.zeros((N, G), dtype=torch.float32)
    PI_full      = torch.zeros((N, G), dtype=torch.float32)
    THETA_full      = torch.zeros((N, G), dtype=torch.float32)

    best_X_recon_full = torch.zeros((N, G), dtype=torch.float32)
    best_PI_full = torch.zeros((N, G), dtype=torch.float32)
    best_THETA_full = torch.zeros((N, G), dtype=torch.float32)

    # 打乱索引
    cell_indices = np.arange(N)

    PI, X_recon = None, None
    best_epoch, min_loss = 1, float('inf')
    graph_AE_loss_list = []
    lzinbloss = loss.LZINBLoss()

    if use_GAT:
        graph_gene_full = utils.csr_sub_to_edge_index(adj_gene_csr, device)
    else:
        graph_gene_full = utils.csr_sub_to_torch_sparse(adj_gene_csr, device)

    adj_gene_label_full = torch.from_numpy(adj_gene_csr.toarray()).float().to(device)
    pos_weight_gene_full, norm_gene_full = utils.subgraph_posweight_norm(adj_gene_csr, device)

    # 🔥 训练 + 显存监控
    epoch_peaks = []
    for epoch in range(total_epoch):
        if param['device'].type == "cuda":
            torch.cuda.reset_peak_memory_stats(param['device'])

        graph_AE.train()
        optimizer.zero_grad(set_to_none=True)
        total_epoch_loss = 0.0

        if getattr(args, 'shuffle_each_epoch', True):
            np.random.shuffle(cell_indices)

        for c_start in range(0, N, Bc):
            c_end = min(c_start + Bc, N)
            idx_c = cell_indices[c_start:c_end]
            X_sub_np = X_all[idx_c, :]
            X_sub = torch.from_numpy(X_sub_np).float().to(device)

            cell_sub_csr = adj_cell_csr[idx_c][:, idx_c]

            if use_GAT:
                graph_cell_sub = utils.csr_sub_to_edge_index(cell_sub_csr, device)
            else:
                graph_cell_sub = utils.csr_sub_to_torch_sparse(cell_sub_csr, device)

            adj_cell_label_sub = torch.from_numpy(cell_sub_csr.toarray()).float().to(device)
            pos_weight_cell_sub, norm_cell_sub = utils.subgraph_posweight_norm(cell_sub_csr, device)

            adj_gene_recon, adj_gene_info, adj_cell_recon, adj_cell_info, X_recon, PI, THETA = graph_AE(
                X_sub, graph_gene_full, graph_cell_sub, idx_c=idx_c, idx_g=None, use_GAT=use_GAT
            )

            ae_loss = lzinbloss(X_sub, X_recon, PI, THETA)

            if use_GAT:
                graph_loss = loss.loss_function_GAT(
                    preds=(adj_gene_recon, adj_cell_recon),
                    labels=(adj_gene_label_full, adj_cell_label_sub)
                )
            else:
                graph_loss = loss.loss_function_GCN(
                    preds=(adj_gene_recon, adj_cell_recon),
                    labels=(adj_gene_label_full, adj_cell_label_sub),
                    num_nodes_gene=graph_gene_full.shape[0] if graph_gene_full.is_sparse else adj_gene_label_full.shape[0],
                    pos_weight_gene=pos_weight_gene_full,
                    norm_gene=norm_gene_full,
                    adj_gene_info1=adj_gene_info[0], adj_gene_info2=adj_gene_info[1],
                    num_nodes_cell=adj_cell_label_sub.shape[0],
                    pos_weight_cell=pos_weight_cell_sub,
                    norm_cell=norm_cell_sub,
                    adj_cell_info1=adj_cell_info[0], adj_cell_info2=adj_cell_info[1]
                )

            batch_loss = ae_loss + graph_loss

            # 反向
            batch_loss.backward()
            total_epoch_loss += batch_loss.item()

            # ====== 将子块写回 CPU 全矩阵对应位置 ======
            X_recon_full[idx_c, :] = X_recon.detach().cpu()
            PI_full[idx_c, :] = PI.detach().cpu()
            THETA_full[idx_c, :] = THETA.detach().cpu()

            # 释放 cell 块相关
            del X_sub, X_recon, PI, THETA, \
                cell_sub_csr, adj_cell_label_sub, graph_cell_sub
            torch.cuda.empty_cache()

        optimizer.step()
        lr_scheduler.step(total_epoch_loss)

        logging.info(f"----------------> Epoch: {epoch + 1}/{total_epoch}, Current loss: {total_epoch_loss:.6f}")
        graph_AE_loss_list.append(total_epoch_loss)

        # 保存最优
        if total_epoch_loss < min_loss:
            min_loss = total_epoch_loss
            best_epoch = epoch + 1
            torch.save(graph_AE.state_dict(), os.path.join(output_dir, 'Graph_AE_best_model.pkl'))

            # 🔥 同时保存当下最优的结果
            best_X_recon_full = X_recon_full.clone()
            best_PI_full = PI_full.clone()
            best_THETA_full = THETA_full.clone()

        if param['device'].type == "cuda":
            peak_epoch_mem = torch.cuda.max_memory_allocated(param['device']) / 1024 ** 2
            epoch_peaks.append(peak_epoch_mem)
            utils.log_gpu_memory(param['device'], note=f"Epoch {epoch + 1} peak")

    # =============== 训练完成：选择性插补并返回 ===============
    param['epoch_peaks'] = epoch_peaks
    param['graph_AE_loss_list'] = graph_AE_loss_list

    print('Best epoch:', best_epoch)
    print('Min loss:', min_loss)

    # 选择性插补
    X_cpu = torch.from_numpy(X_all).float()
    iszero = X_cpu == 0
    predict_dropout_mask = (best_PI_full > 0.5) & iszero
    X_imputed = torch.where(predict_dropout_mask, best_X_recon_full, X_cpu)

    return X_imputed.numpy()








