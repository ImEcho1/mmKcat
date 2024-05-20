import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool


def normalize_adj(adj):
    rowsum = adj.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    adj = torch.mm(torch.mm(r_mat_inv_sqrt, adj), r_mat_inv_sqrt)
    return adj


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(GCNLayer, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features // 2)
        self.linear_2 = nn.Linear(out_features // 2, out_features)
        self.device = device

        self.init_weight()

    def init_weight(self):
        init.kaiming_normal_(self.linear_1.weight)
        init.kaiming_normal_(self.linear_2.weight)

    def forward(self, x, adj):
        x = self.linear_1(x)
        x = self.linear_2(x)
        adj = adj + torch.eye(adj.size(0)).cuda()
        adj = normalize_adj(adj)
        x = torch.spmm(adj.to(torch.float32), x.to(torch.float32))
        return x


class MaterialGCN(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(MaterialGCN, self).__init__()
        self.device = device
        self.layer = GCNLayer(in_features, out_features, device)

    def forward(self, x, adj):
        x = self.layer(x, adj)
        x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long).to(self.device))
        return x
