import torch
from torch import nn
from layer import GAT, GraphConvolution, InnerProductDecoder, MultiplyLayer, SubLinear
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
import torch.utils.checkpoint as cp


class Graph_AE(nn.Module):

    def __init__(self,
                 num_cell, num_gene,
                 hid_embed1_gene, hid_embed2_gene, dropout_gene, gat_multi_heads_gene, multi_heads_gene,
                 hid_embed1_cell, hid_embed2_cell, dropout_cell, gat_multi_heads_cell, multi_heads_cell, use_checkpoint=False):
        super(Graph_AE, self).__init__()

        self.num_cell = num_cell
        self.num_gene = num_gene
        self.use_checkpoint = use_checkpoint

        self.gat_multi_heads_gene = gat_multi_heads_gene
        self.multi_heads_gene = multi_heads_gene
        self.hid_embed1_gene = hid_embed1_gene
        self.hid_embed2_gene = hid_embed2_gene
        self.dropout_gene = dropout_gene

        self.gnn_gene = Graph(self.num_cell, self.hid_embed1_gene, self.hid_embed2_gene, self.num_gene, self.dropout_gene, self.gat_multi_heads_gene)
        self.ae_gene = AE(self.num_cell, self.hid_embed1_gene, self.hid_embed2_gene, self.multi_heads_gene, self.num_gene, self.num_cell)

        self.gat_multi_heads_cell = gat_multi_heads_cell
        self.multi_heads_cell = multi_heads_cell
        self.hid_embed1_cell = hid_embed1_cell
        self.hid_embed2_cell = hid_embed2_cell
        self.dropout_cell = dropout_cell

        self.gnn_cell = Graph(self.num_gene, self.hid_embed1_cell, self.hid_embed2_cell, self.num_cell, self.dropout_cell, self.gat_multi_heads_cell)
        self.ae_cell = AE(self.num_gene, self.hid_embed1_cell, self.hid_embed2_cell, self.multi_heads_cell, self.num_cell, self.num_gene)

    def forward(self, X, graph_gene, graph_cell,idx_c,idx_g, use_GAT=False):

        if self.use_checkpoint:
            def gnn_gene_forward(x, g,idx_c,idx_g, flag_gene, use_GAT):
                return self.gnn_gene(x.T, g,idx_c,idx_g, flag_gene, use_GAT)

            def gnn_cell_forward(x, g,idx_c,idx_g, flag_gene, use_GAT):
                return self.gnn_cell(x, g,idx_c,idx_g, flag_gene, use_GAT)

            adj_gene_hidden1, adj_gene_hidden2, adj_gene_recon, adj_gene_info = cp.checkpoint(
                gnn_gene_forward, X, graph_gene,idx_c,idx_g, True, use_GAT, use_reentrant=False
            )
            adj_cell_hidden1, adj_cell_hidden2, adj_cell_recon, adj_cell_info = cp.checkpoint(
                gnn_cell_forward, X, graph_cell,idx_c,idx_g, False, use_GAT, use_reentrant=False
            )

            def ae_gene_forward(x, h1, h2, A, B,idx_c,idx_g, flag_gene):
                return self.ae_gene(x.T, h1, h2, A, B,idx_c,idx_g, flag_gene)

            def ae_cell_forward(x, h1, h2, A, B,idx_c,idx_g, flag_gene):
                return self.ae_cell(x, h1, h2, A, B,idx_c,idx_g, flag_gene)

            M_gene, PI_gene, THETA_gene = cp.checkpoint(
                ae_gene_forward, X, adj_gene_hidden1, adj_gene_hidden2, adj_gene_recon, adj_cell_recon,idx_c,idx_g, True, use_reentrant=False
            )
            M_cell, PI_cell, THETA_cell = cp.checkpoint(
                ae_cell_forward, X, adj_cell_hidden1, adj_cell_hidden2, adj_cell_recon, adj_gene_recon,idx_c,idx_g, False, use_reentrant=False
            )

        else:

            # 基因图
            adj_gene_hidden1, adj_gene_hidden2, adj_gene_recon, adj_gene_info = \
                self.gnn_gene(X.T, graph_gene, idx_c=None, idx_g=idx_g, use_GAT=use_GAT)

            # 细胞图
            adj_cell_hidden1, adj_cell_hidden2, adj_cell_recon, adj_cell_info = \
                self.gnn_cell(X, graph_cell, idx_c=idx_c, idx_g=None, use_GAT=use_GAT)

            M_gene, PI_gene, THETA_gene = self.ae_gene(X.T, adj_gene_hidden1, adj_gene_hidden2, adj_gene_recon, adj_cell_recon,idx_c,idx_g)

            M_cell, PI_cell, THETA_cell = self.ae_cell(X, adj_cell_hidden1, adj_cell_hidden2, adj_cell_recon, adj_gene_recon,idx_c,idx_g)

        sigma = 0.5
        M = (1 - sigma) * torch.t(M_gene) + sigma * M_cell
        PI = (1 - sigma) * torch.t(PI_gene) + sigma * PI_cell
        THETA = (1 - sigma) * torch.t(THETA_gene) + sigma * THETA_cell

        return adj_gene_recon, adj_gene_info, adj_cell_recon, adj_cell_info, M, PI, THETA



class Graph(nn.Module):

    def __init__(self, inputdim, hid_embed1, hid_embed2, outputdim, dropout, gat_multi_heads):
        super(Graph, self).__init__()

        self.gat = GAT(num_of_layers=2,
                        num_heads_per_layer=[gat_multi_heads, gat_multi_heads],
                        num_features_per_layer=[inputdim, hid_embed1, hid_embed2],
                        dropout=dropout)
        self.decode = InnerProductDecoder(0, act=lambda x: x)

        self.adj_hidden1 = GraphConvolution(inputdim, hid_embed1, dropout, act=torch.tanh)
        self.adj_hidden2 = GraphConvolution(hid_embed1, hid_embed2, dropout, act=F.relu)
        self.adj_output1 = GraphConvolution(hid_embed2, outputdim, dropout, act=lambda x: x)
        self.adj_output2 = GraphConvolution(hid_embed2, outputdim, dropout, act=lambda x: x)

    def encode_gcn(self, x, adj,idx_c,idx_g, flag_gene):
        if flag_gene:
            hidden1 = self.adj_hidden1(x, adj,idx_c,None)
            hidden2 = self.adj_hidden2(hidden1, adj,None,None)
            output1 = self.adj_output1(hidden2, adj,None,idx_g)
            output2 = self.adj_output2(hidden2, adj,None,idx_g)
        else:
            hidden1 = self.adj_hidden1(x, adj, idx_g, None)
            hidden2 = self.adj_hidden2(hidden1, adj, None, None)
            output1 = self.adj_output1(hidden2, adj, None, idx_c)
            output2 = self.adj_output2(hidden2, adj, None, idx_c)

        return hidden1, hidden2, output1, output2

    def reparameterize(self, output1, output2):
        std = torch.exp(output2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(output1)

    def encode_gat(self, in_nodes_features, edge_index):
        return self.gat((in_nodes_features, edge_index))

    def forward(self, x, adj,idx_c,idx_g, flag_gene, use_GAT=False):
        adj_output1, adj_output2 = None, None
        if use_GAT:
            adj_hidden1, adj_hidden2 = self.encode_gat(x, adj)
            adj_recon = self.decode(adj_hidden2)

        else:
            adj_hidden1, adj_hidden2, adj_output1, adj_output2 = self.encode_gcn(x, adj,idx_c,idx_g, flag_gene)
            adj_recon = self.reparameterize(adj_output1, adj_output2)

        return adj_hidden1, adj_hidden2, adj_recon, (adj_output1, adj_output2)


class AE(nn.Module):
    def __init__(self, dim, hidden1, hidden2, heads, num_A, num_B):

        super().__init__()
        self.Gnoise = GaussianNoise(sigma=0.1)

        self.input = SubLinear(dim, hidden1, mode="in")

        self.bn1 = BatchNorm1d(hidden1)
        self.encode = Linear(hidden1, hidden2)
        self.bn2 = BatchNorm1d(hidden2)
        self.decode = Linear(hidden2, hidden1)
        self.bn3 = BatchNorm1d(hidden1)

        self.output_PI = SubLinear(hidden1, dim, mode="out")
        self.output_M = SubLinear(hidden1, dim, mode="out")
        self.output_THETA = SubLinear(hidden1, dim, mode="out")

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)

        self.block1 = Block(hidden1, hidden1 // 2, heads, hidden1 // 2, hidden1, hidden1 // 2)
        self.block2 = Block(hidden2, hidden2 // 2, heads, hidden2 // 2, hidden2, hidden2 // 2)

        self.express_inner = MultiplyLayer(num_A=num_A, num_B=num_B, act=self.relu)

    def forward(self, x, adj_hidden1, adj_hidden2, A, B,idx_c,idx_g, flag_gene):
        if flag_gene:
            x = self.express_inner((x, A, B,idx_c,idx_g))

            enc_h1 = self.bn1(self.relu(self.input(self.Gnoise(x), idx_c)))

            enc_h1 = self.block1(enc_h1, adj_hidden1)

            enc_h2 = self.bn2(self.relu(self.encode(self.Gnoise(enc_h1))))

            enc_h2 = self.block2(enc_h2, adj_hidden2)

            dec_h1 = self.bn3(self.relu(self.decode(enc_h2)))

            PI = self.sigmoid(self.output_PI(dec_h1, idx_c))
            M = self.MeanAct(self.output_M(dec_h1, idx_c))
            THETA = self.DispAct(self.output_THETA(dec_h1, idx_c))

        else:
            x = self.express_inner((x, A, B, idx_g, idx_c))
            enc_h1 = self.bn1(self.relu(self.input(self.Gnoise(x), idx_g)))
            enc_h1 = self.block1(enc_h1, adj_hidden1)
            enc_h2 = self.bn2(self.relu(self.encode(self.Gnoise(enc_h1))))
            enc_h2 = self.block2(enc_h2, adj_hidden2)
            dec_h1 = self.bn3(self.relu(self.decode(enc_h2)))
            PI = self.sigmoid(self.output_PI(dec_h1, idx_g))
            M = self.MeanAct(self.output_M(dec_h1, idx_g))
            THETA = self.DispAct(self.output_THETA(dec_h1, idx_g))

        return PI, M, THETA


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

        self.register_buffer("noise", torch.tensor(0.0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x

            sampled_noise = self.noise.repeat(*x.size()).to(x.device).float().normal_() * scale
            x = x + sampled_noise
        return x





class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_
        attention = attention.squeeze(2)

        return (attention[0]+attention[1])/2


class CAM(nn.Module):
    def __init__(self, features, in_features):
        super(CAM, self).__init__()

        self.pool = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Conv2d(features, features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(features),
                        nn.Conv2d(features, in_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_features))

        self.adapt = nn.Sequential(
                        nn.Conv2d(features, features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(features),
                        nn.Conv2d(features, in_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_features))


        self.delta_gen = nn.Sequential(
                        nn.Conv2d(in_features*2, in_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_features),
                        nn.Conv2d(in_features, 2, kernel_size=1, bias=False)
                        )
        self.delta_gen[2].weight.data.zero_()
        self.reprojection = nn.Conv2d(in_features, features, 1)

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w/s, h/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid)
        return output

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage):
        n, c, h, w = low_stage.shape
        high_stage = self.pool(low_stage)
        low_stage = self.adapt(low_stage)

        high_stage_up = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low_stage, high_stage_up), 1)
        delta = self.delta_gen(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta)

        output = torch.cat((high_stage, low_stage), 1).squeeze(2)
        return (output[0]+output[1])/2

class Block(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels, features, in_features):
        super(Block, self).__init__()
        self.ea = EfficientAttention(in_channels, key_channels, head_count, value_channels)
        self.cam = CAM(features, in_features)


    def forward(self, x, y):
        input_ = torch.stack([x, y]).permute([0, 2, 1]).unsqueeze(2)
        input_1 = self.ea(input_).T
        input_2 = self.cam(input_).T

        return (input_1 + input_2)/2

