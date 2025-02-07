import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# Special SpMM function for sparse matrix multiplication
class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b
    
class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class MetaModalityHybrid(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(MetaModalityHybrid, self).__init__()
        # Define layers for multi-modality feature fusion
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, modality1, modality2):
        x1 = self.fc1(modality1)
        x2 = self.fc2(modality2)
        # Combine modalities
        combined = torch.cat((x1, x2), dim=1)
        return self.output_layer(combined)  

# Graph Convolution Layer (GCN)
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define weights
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input, adj):
        # Apply graph convolution: H' = A * X * W + b
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# MultiHead Graph Attention (Modified for multi-modal input)
class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in * 2  # Assuming concatenation of modalities (e.g., text + image)
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, self.f_in, f_out))  # Adjust input size for multi-modal fusion
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            edge = adj._indices()
            h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)))
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)

            output.append(h_prime.unsqueeze(0))
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# Projection head (no change needed but could be modified for multi-modal fusion)
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        if x is not None:
            x = self.l1(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.l2(x)
            return x