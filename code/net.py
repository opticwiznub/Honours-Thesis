import torch
import torch.nn as nn
import pdb
import copy

def torch_normalize_adj(adj):
    adj = adj + torch.eye(adj.shape[0]).cpu()
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).cpu()
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)

class net_gcn(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
        self.normalize = torch_normalize_adj
    
    def forward(self, x, adj, val_test=False):
        
        adj = torch.mul(adj, self.adj_mask1_train)
        adj = torch.mul(adj, self.adj_mask2_fixed)
        adj = self.normalize(adj)
        #adj = torch.mul(adj, self.adj_mask2_fixed)
        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask