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
    logging.info('--------> Starting Graph plus AE ...')

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

    # X_dropout = X_dropout[X_dropout.obs.dca_split == 'train']

    adj_gene_orig = get_adj.get_gene_adjacency_matrix(X_dropout)
    adj_gene_list = get_adj.adj2list(adj_gene_orig)
    adj_gene_label, adj_gene = get_adj.adj_normalization_tensor(adj_gene_orig)

    adj_cell_orig = get_adj.get_cell_adjacency_matrix(X_dropout)
    adj_cell_list = get_adj.adj2list(adj_cell_orig)
    adj_cell_label, adj_cell = get_adj.adj_normalization_tensor(adj_cell_orig)

    adj_gene_label = adj_gene_label.toarray()
    adj_gene_label = torch.from_numpy(adj_gene_label).type(torch.FloatTensor).to(param['device'])

    adj_cell_label = adj_cell_label.toarray()
    adj_cell_label = torch.from_numpy(adj_cell_label).type(torch.FloatTensor).to(param['device'])

    if use_GAT:
        edgeIndex_gene = utils.edgeList2edgeIndex(adj_gene_list)
        edgeIndex_gene = np.array(edgeIndex_gene).T
        graph_gene = torch.from_numpy(edgeIndex_gene).type(torch.LongTensor).to(param['device'])

        edgeIndex_cell = utils.edgeList2edgeIndex(adj_cell_list)
        edgeIndex_cell = np.array(edgeIndex_cell).T
        graph_cell = torch.from_numpy(edgeIndex_cell).type(torch.LongTensor).to(param['device'])
    else:
        graph_gene = adj_gene.to(param['device'])

        pos_weight_gene = float(adj_gene_orig.shape[0] * adj_gene_orig.shape[0] - adj_gene_orig.sum()) / adj_gene_orig.sum()
        norm_gene = adj_gene_orig.shape[0] * adj_gene_orig.shape[0] / float(
            (adj_gene_orig.shape[0] * adj_gene_orig.shape[0] - adj_gene_orig.sum()) * 2)

        pos_weight_gene = torch.from_numpy(np.array(pos_weight_gene)).type(torch.FloatTensor).to(param['device'])
        norm_gene = torch.from_numpy(np.array(norm_gene)).type(torch.FloatTensor).to(param['device'])

        graph_cell = adj_cell.to(param['device'])

        pos_weight_cell = float(
            adj_cell_orig.shape[0] * adj_cell_orig.shape[0] - adj_cell_orig.sum()) / adj_cell_orig.sum()
        norm_cell = adj_cell_orig.shape[0] * adj_cell_orig.shape[0] / float(
            (adj_cell_orig.shape[0] * adj_cell_orig.shape[0] - adj_cell_orig.sum()) * 2)

        pos_weight_cell = torch.from_numpy(np.array(pos_weight_cell)).type(torch.FloatTensor).to(param['device'])
        norm_cell = torch.from_numpy(np.array(norm_cell)).type(torch.FloatTensor).to(param['device'])


    X_process = torch.from_numpy(X_dropout.X).type(torch.FloatTensor).to(param['device'])

    graph_AE = model.Graph_AE(X_process,
                              hid_embed1_gene, hid_embed2_gene, dropout_gene, gat_multi_heads_gene, multi_heads_gene,
                              hid_embed1_cell, hid_embed2_cell, dropout_cell, gat_multi_heads_cell, multi_heads_cell).to(param['device'])

    optimizer = optim.Adam(graph_AE.parameters(), lr=learning_rate)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    PI, X_recon = None, None
    best_PI, best_X_recon = None, None

    min_loss = float('inf')
    best_epoch = 1

    graph_AE_loss_list = []
    for epoch in range(total_epoch):
        graph_AE.train()
        optimizer.zero_grad()

        adj_gene_recon, adj_gene_info, adj_cell_recon, adj_cell_info, X_recon, PI, THETA = graph_AE(graph_gene, graph_cell, use_GAT=use_GAT)

        lzinbloss = loss.LZINBLoss()
        ae_loss = lzinbloss(X_process, X_recon, PI, THETA)


        if use_GAT:
            graph_loss = loss.loss_function_GAT(preds=(adj_gene_recon, adj_cell_recon), labels=(adj_gene_label, adj_cell_label))
        else:
            graph_loss = loss.loss_function_GCN(preds=(adj_gene_recon, adj_cell_recon),
                                            labels=(adj_gene_label, adj_cell_label),
                                            num_nodes_gene=X_process.shape[1], pos_weight_gene=pos_weight_gene,
                                            norm_gene=norm_gene, adj_gene_info1=adj_gene_info[0],
                                            adj_gene_info2=adj_gene_info[1],
                                            num_nodes_cell=X_process.shape[0], pos_weight_cell=pos_weight_cell,
                                            norm_cell=norm_cell, adj_cell_info1=adj_cell_info[0],
                                            adj_cell_info2=adj_cell_info[1])

        epoch_loss = ae_loss + graph_loss

        epoch_loss.backward()

        cur_loss = epoch_loss.item()
        optimizer.step()


        lr_scheduler.step(epoch_loss)

        logging.info(f"----------------> Epoch: {epoch + 1}/{total_epoch}, Current loss: {cur_loss:.6f}")
        graph_AE_loss_list.append(cur_loss)


        output_dir = args.output_dir
        if not os.path.exists(output_dir):  
            os.makedirs(output_dir)

        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = epoch+1
            model_path = os.path.join(output_dir, f'Graph_AE_best_model.pkl')
            torch.save(graph_AE.state_dict(), model_path)

            # 🔥 同时保存当下最优的结果
            best_X_recon = X_recon.clone()
            best_PI = PI.clone()

    param['graph_AE_loss_list'] = graph_AE_loss_list

    print('Best epoch:', best_epoch)
    print('Min loss:', min_loss)

    iszero = X_process == 0
    predict_dropout_of_all = best_PI > 0.5
    predict_dropout_mask = iszero * predict_dropout_of_all
    X_imputed = torch.where(predict_dropout_mask, best_X_recon, X_process)

    return X_imputed.detach().cpu().numpy()






