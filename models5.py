import copy
import math

import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

# from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
from dgl import knn_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class GIN(torch.nn.Module):
    def __init__(self, param):
        super(GIN, self).__init__()
        from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv, KNNGraph
        self.num_layers = 3
        self.dropout = nn.Dropout(param['dropout_ratio'])
        # self.dropout = nn.Dropout(0.2)

        self.x_mlp = nn.Sequential(
            nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(param['ppi_hidden_dim'])
        )
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.2),
                                                     nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.2),
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])),
                                       aggregator_type='sum',
                                       learn_eps=True))
        self.x_mlp2 = nn.Sequential(
            nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
            nn.ReLU(),
            nn.LayerNorm(param['ppi_hidden_dim'])
        )
        self.layers2 = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers2.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                      ),
                                        aggregator_type='mean',
                                        learn_eps=True))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

        # self.g = dgl.graph(torch.load('../PPI_G/g.pt'))

    def forward(self, g, x, ppi_list, idx):
        # print(g)
        ls = list(range(0, x.shape[0]))
        node = np.random.choice(ls, int(x.shape[0] * 1.0), replace=False)

        # L-sample
        g_all = []
        ty = [-1, 30, 20]
        for i in range(len(ty)):
            g_all.append(dgl.sampling.sample_neighbors(g, node, ty[i]).to(device))


        x_T = []  # node_features
        x_T1 = []  # node_features
        # x_T2 = []
        x_ori = torch.clone(x).to(device)
        x = self.x_mlp(x_ori)
        x_T1.append(x)
        # x2 = self.x_mlp2(x_ori)
        # x_T2.append(x2)
        kin = [10, 20, 30]
        x0 = torch.clone(x)
        # x_P = []
        for i in range(len(ty)):
            x = self.layers[0](g_all[i], x0) - x0
            x = self.layers[1](g_all[i], x)
            g0 = knn_graph(x, k=kin[i])
            x = self.layers2[0](g0, x) - x
            x = self.dropout(x)
            x_T.append(x)
        x_o = []
        meansum = torch.stack(x_T, dim=0).mean(dim=0)
        for i in range(len(x_T)):
            x = F.dropout(F.relu(self.linear(x_T[i])), p=0.5, training=self.training)
            node_id = np.array(ppi_list)[idx]
            x1 = x[node_id[:, 0]]
            x2 = x[node_id[:, 1]]
            x = torch.mul(x1, x2)
            x = self.fc(x)
            x_o.append(x.unsqueeze(0))
            # x_o.append(x)
        # x_p = x_o
        x_o = torch.cat(x_o, dim=0)  # unsqueeze(0) + cat = stack??
        # x_o = torch.stack(x_o, dim=0)
        # print(x_o.shape)#torch.Size([3, 4427, 7])
        # print(len(x_T))#3
        # print(meansum.shape)#torch.Size([1663, 1024])
        # exit()
        return torch.mean(x_o, dim=0), x_T, meansum

class GIN_zhr(torch.nn.Module):
    def __init__(self, param):
        super(GIN_zhr, self).__init__()
        from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv, KNNGraph
        self.num_layers = 3
        self.dropout = nn.Dropout(param['dropout_ratio'])
        # self.dropout = nn.Dropout(0.2)

        self.x_mlp = nn.Sequential(
            nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(param['ppi_hidden_dim'])
        )
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.2),
                                                     nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.2),
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])),
                                       aggregator_type='sum',
                                       learn_eps=True))
        self.x_mlp2 = nn.Sequential(
            nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
            nn.ReLU(),
            nn.LayerNorm(param['ppi_hidden_dim'])
        )
        self.layers2 = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers2.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                      ),
                                        aggregator_type='mean',
                                        learn_eps=True))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

        # self.g = dgl.graph(torch.load('../PPI_G/g.pt'))

    def forward(self, g, x, ppi_list, idx):
        x_T1 = []  # node_features
        x_ori = torch.clone(x).to(device)
        x = self.x_mlp(x_ori)
        kin = [30, 20, 30]
        x0 = torch.clone(x)
        x = self.layers[0](g, x0) - x0
        x = self.layers[1](g, x)#0.75
        #x = self.layers[1](g, x)-x 0.69
        g0 = knn_graph(x, k=kin[0])
        x = self.layers2[0](g0, x)
        x = self.dropout(x)

        x = F.dropout(F.relu(self.linear(x)), p=0.5, training=self.training)
        node_id = np.array(ppi_list)[idx]
        x1 = x[node_id[:, 0]]
        x2 = x[node_id[:, 1]]
        x = torch.mul(x1, x2)
        x = self.fc(x)
        # x_o.append(x)
        # x_p = x_o
        # x_o = torch.cat(x_o, dim=0)  # unsqueeze(0) + cat = stack??
        # x_o = torch.stack(x_o, dim=0)
        # print(x_o.shape)#torch.Size([3, 4427, 7])
        # print(len(x_T))#3
        # print(meansum.shape)#torch.Size([1663, 1024])
        # exit()
        return x
"""
torch.stack(x_T, dim=0): 将张量列表 x_T 沿指定维度 dim=0 堆叠成一个新的张量
示例：
# 创建示例张量
x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x2 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
x3 = torch.tensor([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
堆叠后的张量：
tensor([[[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]],

        [[ 7.,  8.,  9.],
         [10., 11., 12.]],

        [[13., 14., 15.],
         [16., 17., 18.]]])
计算均值后的张量：
tensor([[ 7.,  8.,  9.],
        [10., 11., 12.]])

"""


# class GIN_best(torch.nn.Module):
#     def __init__(self, param):
#         super(GIN_best, self).__init__()
#         from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv, KNNGraph
#         from VGAE2 import VGAEModel2
#         self.num_layers = 3
#         self.dropout = nn.Dropout(param['dropout_ratio'])
#         # self.dropout = nn.Dropout(0.2)
#
#         self.x_mlp = nn.Sequential(
#             nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
#             nn.ReLU(),
#             nn.BatchNorm1d(param['ppi_hidden_dim'])
#         )
#         self.layers = nn.ModuleList()
#         for i in range(self.num_layers):
#             self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
#                                                      nn.ReLU(),
#                                                      nn.Dropout(0.2),
#                                                      nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
#                                                      nn.ReLU(),
#                                                      nn.Dropout(0.2),
#                                                      nn.BatchNorm1d(param['ppi_hidden_dim'])),
#                                        aggregator_type='sum',
#                                        learn_eps=True))
#         self.x_mlp2 = nn.Sequential(
#             nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
#             nn.ReLU(),
#             nn.LayerNorm(param['ppi_hidden_dim'])
#         )
#         self.layers2 = nn.ModuleList()
#         for i in range(self.num_layers):
#             self.layers2.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
#                                                       ),
#                                         aggregator_type='mean',
#                                         learn_eps=True))
#
#         self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
#         self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])
#
#         self.g = dgl.graph(torch.load('../PPI_G/g.pt'))
#
#     def forward(self, g, x, ppi_list, idx):
#
#         ls = list(range(0, x.shape[0]))
#         node = np.random.choice(ls, int(x.shape[0] * 1.0), replace=False)
#
#         g_all = []
#         ty = [-1, 50, 25]
#         for i in range(len(ty)):
#             g_all.append(dgl.sampling.sample_neighbors(g, node, ty[i]).to(device))
#
#         x_T = []
#         x_T2 = []
#         x_ori = torch.clone(x).to(device)
#         x = self.x_mlp(x_ori)
#         x_T.append(x)
#         x2 = self.x_mlp2(x_ori)
#         x_T2.append(x2)
#         kin = [50, 50, 50, 5, 1]
#         x0 = torch.clone(x)
#         for i in range(len(ty)):
#             x = self.layers[0](g_all[i], x0) - x
#             x = self.layers[1](g_all[i], x)
#             g0 = knn_graph(x, k=kin[i])
#             # x = self.layers2[i](g0, x) - x
#             x = self.layers2[0](g0, x) - x
#             x = self.dropout(x)
#             x_T.append(x)
#         x_o = []
#         for i in range(len(x_T)):
#             x = F.dropout(F.relu(self.linear(x_T[i])), p=0.5, training=self.training)
#             node_id = np.array(ppi_list)[idx]
#             x1 = x[node_id[:, 0]]
#             x2 = x[node_id[:, 1]]
#             x = torch.mul(x1, x2)
#             x = self.fc(x)
#             x_o.append(x.unsqueeze(0))
#         x_o = torch.cat(x_o, dim=0)
#         return torch.mean(x_o, dim=0), x, x

class GCN_Encoder(nn.Module):
    def __init__(self, param, data_loader):
        super(GCN_Encoder, self).__init__()

        self.data_loader = data_loader
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList().to(device)
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
        self.layers.append(HeteroGraphConv({'SEQ': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                                            'STR_KNN': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                                            'STR_DIS': GraphConv(param['input_dim'], param['prot_hidden_dim'])},
                                           aggregate='sum')).to(device)

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                                                'STR_KNN': GraphConv(param['prot_hidden_dim'],
                                                                     param['prot_hidden_dim']),
                                                'STR_DIS': GraphConv(param['prot_hidden_dim'],
                                                                     param['prot_hidden_dim'])}, aggregate='sum')).to(
                device)

    def forward(self, vq_layer):

        # exit()
        prot_embed_list = []
        # print(self.data_loader.device)
        # exit()
        for iter, batch_graph in enumerate(self.data_loader):
            batch_graph = batch_graph.to(device)
            # 如果你的图中有节点特征和边特征，也需要移动它们
            # print(batch_graph)
            # print(batch_graph.device)
            # exit()
            h = self.encoding(batch_graph)
            z, _, _ = vq_layer(h)
            batch_graph.ndata['h'] = torch.cat([h, z], dim=-1).to(device)
            prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)

    def encoding(self, batch_graph):

        x = batch_graph.ndata['x']
        # print(x.device)  # x.device=cuda:0
        # exit()
        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x


class GCN_Decoder(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder, self).__init__()

        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                                                'STR_KNN': GraphConv(param['prot_hidden_dim'],
                                                                     param['prot_hidden_dim']),
                                                'STR_DIS': GraphConv(param['prot_hidden_dim'],
                                                                     param['prot_hidden_dim'])}, aggregate='sum'))

        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
        self.layers.append(HeteroGraphConv({'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                                            'STR_KNN': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                                            'STR_DIS': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])},
                                           aggregate='sum'))

    def decoding(self, batch_graph, x):

        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.fc[l](x['amino_acid'])

            if l != self.num_layers - 1:
                x = self.dropout(self.norms[l](F.relu(x)))
            else:
                pass

        return x


class CodeBook(nn.Module):
    def __init__(self, param, data_loader):
        super(CodeBook, self).__init__()

        self.param = param

        self.Protein_Encoder = GCN_Encoder(param, data_loader)
        self.Protein_Decoder = GCN_Decoder(param)

        self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

    def forward(self, batch_graph):
        z = self.Protein_Encoder.encoding(batch_graph)
        e, e_q_loss, encoding_indices = self.vq_layer(z)

        x_recon = self.Protein_Decoder.decoding(batch_graph, e)
        recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

        mask = torch.bernoulli(
            torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])).bool().to(device)
        mask_index = mask[encoding_indices]
        e[mask_index] = 0.0

        x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)

        x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
        mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))

        return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        x = x.to(device)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices

    def get_code_indices(self, x):
        distances = (
                torch.sum(x ** 2, dim=-1, keepdim=True) +
                torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
                2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )

        encoding_indices = torch.argmin(distances, dim=1)

        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)
