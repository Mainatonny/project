import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree
from torch_geometric.nn import GATConv


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):  # 需要有向边聚合
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j] * deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x


class GAT_act(nn.Module):
    def __init__(self, hidden):
        super(GAT_act, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i + e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x


class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden, bias=True)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2) + torch.mul(1 - gate, x1)
        return x


class GAT_E_to_R(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_E_to_R, self).__init__()
        self.a_h1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_h3 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t3 = nn.Linear(r_hidden, 1, bias=False)
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)
        self.s_r = torch.nn.Linear(2 * e_hidden, r_hidden)
        self.s_r1 = torch.nn.Linear(2 * e_hidden, r_hidden)

    def forward(self, x_e, edge_index, rel, rel_size):
        edge_index_h, edge_index_t = edge_index
        x_res = torch.cat([x_e[edge_index_t], x_e[edge_index_h]], dim=1)
        x_res2 = self.s_r1(x_res)
        for i in rel.unique():  # 可以考虑只是初始值
            x_res_h = torch.mean(x_e[edge_index_h[torch.nonzero(rel == i)]].float(), 0)
            x_res_t = torch.mean(x_e[edge_index_t[torch.nonzero(rel == i)]].float(), 0)
            x_r = torch.cat([x_res_h, x_res_t], dim=1)  # 考虑相加
            x_res[torch.nonzero(rel == i)] = x_r
        x_res = self.s_r(x_res)
        x_r_h = self.w_h(x_e[edge_index_h])  # [e_hidden,r_hidden]
        x_r_t = self.w_t(x_e[edge_index_t])
        e1 = self.a_h1(x_r_h).squeeze() + self.a_h2(x_r_h + x_res2 + x_r_t).squeeze()
        e2 = self.a_t1(x_r_h + x_res2 + x_r_t).squeeze() + self.a_t2(x_r_t).squeeze()
        e3 = self.a_h3(x_r_h + x_res2 + x_r_t).squeeze() + self.a_t3(x_res).squeeze()
        alpha = softmax(F.leaky_relu(e1).float(), rel)  # 先以rel作为索引分组，之后在组内softmax
        x_r_h1 = spmm(torch.cat([rel.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, rel.max() + 1, edge_index_h.size(0),
                      x_r_h)
        alpha = softmax(F.leaky_relu(e2).float(), rel)
        x_r_t1 = spmm(torch.cat([rel.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, rel.max() + 1, edge_index_t.size(0),
                      x_r_t)
        alpha = softmax(F.leaky_relu(e3).float(), rel)
        x_rres = spmm(torch.cat([rel.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, rel.max() + 1, edge_index_t.size(0),
                      x_res)
        x_res1 = x_r_h1 + x_r_t1 + x_rres
        x_res = x_res1[rel] + x_res + x_res2
        return x_res


class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_feature, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_features, hidden_feature, heads=heads[0], dropout=0.6)
        self.conv2 = GATConv(hidden_feature * heads[0], out_feature, concat=False, heads=heads[1],
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GATt_to_R(nn.Module):
    def __init__(self, e_hidden, r_hidden, t_hidden):
        super(GATt_to_R, self).__init__()
        self.s_r1 = torch.nn.Linear(2 * t_hidden, r_hidden)
        self.t_c1 = torch.nn.Linear(e_hidden, t_hidden)  # 共享的类型变换层
        self.t_r = torch.nn.Linear(t_hidden, r_hidden)  # 类型空间中concat得来
        self.s_r = torch.nn.Linear(2 * t_hidden, r_hidden)  #
        self.a_1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_3 = nn.Linear(r_hidden, 1, bias=False)
        self.a_4 = nn.Linear(r_hidden, 1, bias=False)
        self.a_5 = nn.Linear(r_hidden, 1, bias=False)
        self.a_6 = nn.Linear(r_hidden, 1, bias=False)
        self.a_7 = nn.Linear(r_hidden, 1, bias=False)
        self.a_8 = nn.Linear(t_hidden, 1, bias=False)
        self.a_9 = nn.Linear(t_hidden, 1, bias=False)
        self.a_10 = nn.Linear(t_hidden, 1, bias=False)
        self.a_11 = nn.Linear(r_hidden, 1, bias=False)
        self.a_12 = nn.Linear(r_hidden, 1, bias=False)
        self.res = nn.Linear(r_hidden, t_hidden)
        self.w_h = nn.Linear(t_hidden, r_hidden)
        self.w_t = nn.Linear(t_hidden, r_hidden)
        self.w_r = nn.Linear(r_hidden, t_hidden)
        self.w_r1 = nn.Linear(r_hidden, t_hidden)

    def forward(self, x_e, edge_index, rel, x_res1, rel_size):
        edge_index_h, edge_index_t = edge_index
        s_convert_t = self.t_c1(x_e)
        s_convert_t = torch.cat([s_convert_t[edge_index_h], s_convert_t[edge_index_t]], dim=0)
        s_convert_t = torch.as_tensor(s_convert_t.view(2, edge_index_h.size(0), -1))
        r_in_t = torch.cat([s_convert_t[0].float(), s_convert_t[1].float()], dim=1)
        r_in_t1 = r_in_t
        for i in rel.unique():
            x_res_h = torch.mean(s_convert_t[0][torch.nonzero(rel == i)].float(), 0)
            x_res_t = torch.mean(s_convert_t[1][torch.nonzero(rel == i)].float(), 0)
            x_r = torch.cat([x_res_h, x_res_t], dim=1)
            r_in_t[torch.nonzero(rel == i)] = x_r

        x_res2 = self.s_r1(r_in_t1)

        e1 = self.a_1(x_res2).squeeze() + self.a_5(x_res1).squeeze()
        alpha = softmax(F.leaky_relu(e1).float(), rel)
        x_r_h1 = spmm(torch.cat([rel.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, rel.max() + 1, edge_index_h.size(0),
                      x_res2)

        x_type = x_r_h1
        x_res1 = x_res1 + x_type[rel]

        return torch.cat([x_res1, r_in_t1], dim=1)


class GATr_to_e(nn.Module):
    def __init__(self, e_hidden, r_hidden, t_hidden):
        super(GATr_to_e, self).__init__()
        self.w_r = nn.Linear(r_hidden + 2 * t_hidden, e_hidden)
        self.w_r1 = nn.Linear(r_hidden + 2 * t_hidden, e_hidden)
        self.w_r2 = nn.Linear(r_hidden + 2 * t_hidden, e_hidden)
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_h1 = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r1 = nn.Linear(e_hidden, 1, bias=False)
        self.a_r2 = nn.Linear(e_hidden, 1, bias=False)
        self.a_r3 = nn.Linear(e_hidden, 1, bias=False)


    def forward(self, x_e, x_r, edge_index, rel_size):
        edge_index_h, edge_index_t = edge_index
        e_h = x_e[edge_index_h]
        e_r = self.w_r(x_r).squeeze()
        eh = self.a_h(e_h).squeeze() + self.a_r1(e_r).squeeze()
        alpha = softmax(F.leaky_relu(eh).float(), edge_index_h)
        x_e_h = F.relu(spmm(torch.cat([edge_index_h.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                            e_r))
        x_e = x_e + x_e_h

        e_t = x_e[edge_index_t]
        e_r = self.w_r1(x_r).squeeze()
        et = self.a_t(e_t).squeeze() + self.a_r2(e_r).squeeze()
        alpha = softmax(F.leaky_relu(et).float(), edge_index_t)
        x_e_t = F.relu(spmm(torch.cat([edge_index_t.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                            e_r))
        x_e = x_e + x_e_t

        e_h = x_e[edge_index_h]
        e_r = self.w_r2(x_r).squeeze()
        eh = self.a_h1(e_h).squeeze() + self.a_r3(e_r).squeeze()
        alpha = softmax(F.leaky_relu(eh).float(), edge_index_h)
        x_e_h = F.relu(spmm(torch.cat([edge_index_h.view(1, -1), rel_size.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                            e_r))
        x = x_e + x_e_h
        return x


class FTDEA(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100, t_hidden=100):
        super(FTDEA, self).__init__()
        self.s_r = torch.nn.Linear(2 * e_hidden, r_hidden)
        # 三层GCN获取初步聚合拓扑结构的实体表示
        self.gcn1 = GCN()
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN()
        self.highway2 = Highway(e_hidden)
        self.gat_e_to_r = GAT_E_to_R(e_hidden, r_hidden)
        # 类型空间加强关系表示
        self.gat_t_to_r = GATt_to_R(e_hidden, r_hidden, t_hidden)
        self.gatr_to_e = GATr_to_e(e_hidden, r_hidden, t_hidden)
        self.gat1 = GAT_act(e_hidden)
        self.fc = nn.Linear(int(e_hidden*2), 1)

    def forward(self, x, edge_index, rel, edge_index_all, rel_size):
        x = self.highway1(x, self.gcn1(x, edge_index_all))
        x = self.highway2(x, self.gcn2(x, edge_index_all))
        x_res = self.gat_e_to_r(x, edge_index, rel, rel_size)  # 获取关系语义表示
        x_res = torch.zeros_like(x_res).to('cuda')#消融实验
        r_f = self.gat_t_to_r(x, edge_index, rel, x_res, rel_size)  # 类型增强的关系表示
        # r_f = torch.zeros_like(r_f).to('cuda')消融实验
        x = self.gatr_to_e(x, r_f, edge_index, rel_size) # x_e, x_r, edge_index, rel_size
        # x=torch.zeros_like(x).to('cuda')消融实验
        x_e = torch.cat([x, self.gat1(x, edge_index_all)], dim=1)  # 最终实体表示
        x_e = self.fc(x_e).squeeze()
        return x_e