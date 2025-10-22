import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np



def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = torch.sum(torch.tensor(~torch.isnan(x),dtype = torch.float32))
    return torch.tensor(torch.where(torch.equal(nelem, 0.), 1., nelem), dtype = x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.sum(x), nelem)



class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = torch.tensor(y_true, dtype = torch.float32)
        y_pred = torch.tensor(y_pred, dtype = torch.float32) * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        theta = torch.minimum(self.theta,torch.tensor(1e6))

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))


        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.sum(final), nelem)
            else:
                final = torch.sum(final)


        return final

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0-self.pi+eps)

        y_true = torch.tensor(y_true, dtype = torch.float32)
        y_pred = torch.tensor(y_pred, dtype = torch.float32) * scale_factor
        theta = torch.minimum(self.theta,torch.tensor(1e6))

        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.sum(result)

        result = _nan2inf(result)

        return result



class LZINBLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X: Tensor, X_recon: Tensor = None, PI: Tensor = None, THETA: Tensor = None):

        eps = self.eps
        max1 = max(THETA.max(), X_recon.max()).data
        if THETA.isinf().sum() != 0:
            THETA = torch.where(THETA.isinf(), torch.full_like(THETA, max1), THETA)
        if X_recon.isinf().sum() != 0:
            X_recon = torch.where(X_recon.isinf(), torch.full_like(X_recon, max1), X_recon)

        if PI.isnan().sum() != 0:
            PI = torch.where(PI.isnan(), torch.full_like(PI, eps), PI)
        if THETA.isnan().sum() != 0:
            THETA = torch.where(THETA.isnan(), torch.full_like(THETA, eps), THETA)
        if X_recon.isnan().sum() != 0:
            X_recon = torch.where(X_recon.isnan(), torch.full_like(X_recon, eps), X_recon)

        eps = torch.tensor(1e-10)

        THETA = torch.minimum(THETA, torch.tensor(1e6))
        t1 = torch.lgamma(THETA + eps) + torch.lgamma(X + 1.0) - torch.lgamma(X + THETA + eps)
        t2 = (THETA + X) * torch.log(1.0 + (X_recon / (THETA + eps))) + (X * (torch.log(THETA + eps) - torch.log(X_recon + eps)))
        nb = t1 + t2
        nb = torch.where(torch.isnan(nb), torch.zeros_like(nb) + max1, nb)
        nb_case = nb - torch.log(1.0 - PI + eps)
        zero_nb = torch.pow(THETA / (THETA + X_recon + eps), THETA)
        zero_case = -torch.log(PI + ((1.0 - PI) * zero_nb) + eps)
        res = torch.where(torch.less(X, 1e-8), zero_case, nb_case)
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + max1, res)
        return torch.mean(res)


def loss_function_GAT(preds, labels):
    adj_gene_recon, adj_cell_recon = preds
    adj_gene_label, adj_cell_label = labels
    cost_gene = F.binary_cross_entropy_with_logits(adj_gene_recon, adj_gene_label)
    cost_cell = F.binary_cross_entropy_with_logits(adj_cell_recon, adj_cell_label)
    return cost_gene + cost_cell


def loss_function_GCN(preds, labels,
                      num_nodes_gene, pos_weight_gene, norm_gene, adj_gene_info1, adj_gene_info2,
                      num_nodes_cell, pos_weight_cell, norm_cell, adj_cell_info1, adj_cell_info2):
    adj_gene_recon, adj_cell_recon = preds
    adj_gene_label, adj_cell_label = labels

    cost_gene = norm_gene * F.binary_cross_entropy_with_logits(adj_gene_recon, adj_gene_label, pos_weight=adj_gene_label * pos_weight_gene)
    kl_gene = -0.5 / num_nodes_gene * torch.mean(torch.sum(1 + 2 * adj_gene_info2 - adj_gene_info1.pow(2) - adj_gene_info2.exp().pow(2), 1))
    cost1 = cost_gene + kl_gene

    cost_cell = norm_cell * F.binary_cross_entropy_with_logits(adj_cell_recon, adj_cell_label, pos_weight=adj_cell_label * pos_weight_cell)
    kl_cell = -0.5 / num_nodes_cell * torch.mean(torch.sum(1 + 2 * adj_cell_info2 - adj_cell_info1.pow(2) - adj_cell_info2.exp().pow(2), 1))
    cost2 = cost_cell + kl_cell

    return cost1 + cost2