#layer.py
import numpy as np
import torch
from torch.nn import Parameter, Dropout
from torch.nn import Sequential, Linear, ReLU, CELU, RReLU, LeakyReLU, GRU
from torch.nn.init import kaiming_uniform_, zeros_
import torch.nn.functional as F
from torch_geometric.nn import (
    NNConv, GCNConv, GATConv,
    global_mean_pool, global_add_pool, global_sort_pool,
    BatchNorm, LayerNorm, PairNorm, GraphSizeNorm,
    GlobalAttention
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax



class _None(torch.nn.Module):
    def __init__(self, *args, **kwargs):  
        super(_None, self).__init__()

    def forward(self, x, batch=None):
        return x


class _BatchNorm(torch.nn.Module):
    def __init__(self, in_channels):
        super(_BatchNorm, self).__init__()
        self.norm = BatchNorm(in_channels)

    def forward(self, x, batch=None):
        return self.norm(x)


class _LayerNorm(torch.nn.Module):
    def __init__(self, in_channels):
        super(_LayerNorm, self).__init__()
        self.norm = LayerNorm(in_channels)

    def forward(self, x, batch=None):
        return self.norm(x, batch)


class _PairNorm(torch.nn.Module):
    def __init__(self, in_channels):
        super(_PairNorm, self).__init__()
        self.norm = PairNorm()

    def forward(self, x, batch=None):
        return self.norm(x, batch)


class _GraphSizeNorm(torch.nn.Module):
    def __init__(self, in_channels):
        super(_GraphSizeNorm, self).__init__()
        self.norm = GraphSizeNorm()

    def forward(self, x, batch=None):
        return self.norm(x)



class _NNConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_in_dim):
        super(_NNConv, self).__init__()
        nn = Sequential(Linear(edge_in_dim, 32), ReLU(), Linear(32, in_dim * out_dim))
        self.conv = NNConv(in_dim, out_dim, nn, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class _GCNConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_in_dim):
        super(_GCNConv, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index)


class _GATConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_in_dim):
        super(_GATConv, self).__init__()
        self.conv = GATConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index)



class TripletMessage(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(TripletMessage, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)
        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        return alpha.view(-1, self.heads, 1) * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        return torch.matmul(aggr_out, self.weight_scale) + self.bias


class TripletMessageLight(MessagePassing):
    def __init__(self, node_channels, edge_channels, negative_slope=0.2, **kwargs):
        super(TripletMessageLight, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.node_channels = node_channels
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, 2 * node_channels + edge_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_triplet_att)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        triplet = torch.cat([x_i, edge_attr, x_j], dim=-1)
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        return alpha.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out + self.bias


class _TripletMessage(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_in_dim):
        super(_TripletMessage, self).__init__()
        self.conv = TripletMessage(in_dim, edge_in_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class _TripletMessageLight(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_in_dim):
        super(_TripletMessageLight, self).__init__()
        self.conv = TripletMessageLight(in_dim, edge_in_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)



class GlobalPool5(torch.nn.Module):
    def __init__(self, **params):
        super(GlobalPool5, self).__init__()

    def forward(self, x, batch):
        mean = global_mean_pool(x, batch)
        summ = global_add_pool(x, batch)
        topk = global_sort_pool(x, batch, k=3)
        return torch.cat([mean, summ, topk], dim=-1)


class GlobalLAPool(torch.nn.Module):
    def __init__(self, in_channels, **params):
        super(GlobalLAPool, self).__init__()
        gate = Linear(in_channels, 1)
        nn = Linear(in_channels, 2 * in_channels)
        self.pool = GlobalAttention(gate_nn=gate, nn=nn)

    def forward(self, x, batch):
        return self.pool(x, batch)



class LinearBlock(torch.nn.Module):
    def __init__(self, in_dim=32, out_dim=64,
                 norm=_None, dropout=_None(), act='ReLU'):
        super(LinearBlock, self).__init__()
        self.norm = norm(in_dim) if callable(norm) else _None()
        self.dropout = dropout if isinstance(dropout, torch.nn.Module) else _None()
        self.linear = Linear(in_dim, out_dim)
        self.act = eval(act)() if isinstance(act, str) else (act() if callable(act) else _None())

    def forward(self, x, batch=None):
        x = self.norm(x, batch)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x)
        return x


class MessageBlock(torch.nn.Module):
    def __init__(self, in_dim=32, out_dim=64, in_edge_dim=13,
                 norm=_None, dropout=Dropout(0.2), conv=_NNConv, act='ReLU', res=True):
        super(MessageBlock, self).__init__()
        self.norm = norm(in_dim) if callable(norm) else _None()
        self.dropout = dropout if isinstance(dropout, torch.nn.Module) else _None()
        self.conv = conv(in_dim, out_dim, in_edge_dim) if callable(conv) else _NNConv(in_dim, out_dim, in_edge_dim)
        self.gru = GRU(in_dim, out_dim) if getattr(conv, '__name__', '') not in ['_GCNConv', '_GATConv'] else None
        self.act = eval(act)() if isinstance(act, str) else (act() if callable(act) else _None())
        self.res = res

    def forward(self, x, edge_index, edge_attr, h=None, batch=None):
        identity = x
        if h is None:
            h = x.unsqueeze(0)
        x = self.norm(x, batch)
        x = self.dropout(x)
        x = self.conv(x, edge_index, edge_attr)
        if self.gru is not None:
            x = torch.celu(x)
            x, h = self.gru(x.unsqueeze(0), h)
            x = x.squeeze(0)
        if self.res:
            x = x + identity
        x = self.act(x)
        return x, h



def dot_and_global_pool5(mol_out, pro_out, mol_batch, pro_batch):
    mol_node_slice = torch.cumsum(torch.from_numpy(np.bincount(mol_batch.cpu())), 0)
    pro_node_slice = torch.cumsum(torch.from_numpy(np.bincount(pro_batch.cpu())), 0)
    batch_size = mol_batch.max() + 1
    out = mol_out.new_zeros([batch_size, 5])
    for i in range(batch_size):
        m_start = mol_node_slice[i - 1].item() if i != 0 else 0
        p_start = pro_node_slice[i - 1].item() if i != 0 else 0
        m_end = mol_node_slice[i].item()
        p_end = pro_node_slice[i].item()
        item = torch.matmul(mol_out[m_start:m_end], pro_out[p_start:p_end].T)
        out[i] = torch.stack([item.max(), item.mean(), item.median(), item.min(), item.std()])
    return out
