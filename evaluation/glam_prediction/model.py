import torch
from torch_geometric.nn import Set2Set
from torch.nn import Dropout, RReLU, LeakyReLU, CELU
from layer import (
    _None, _LayerNorm, _BatchNorm, _PairNorm, _GraphSizeNorm,
    GlobalPool5, GlobalLAPool,
    LinearBlock, MessageBlock,
    _NNConv, _TripletMessage, _TripletMessageLight, _GCNConv, _GATConv
)

def norm_factory(norm_str, fixed_dim=None):
    cls = eval(norm_str)
    if fixed_dim is not None:
        return lambda _: cls(fixed_dim)
    else:
        return lambda in_channels: cls(in_channels)


def model_args(args):
    _other_args_name = ['dataset_root', 'dataset', 'split', 'seed', 'gpu', 'note', 'batch_size', 'epochs', 'loss',
                        'optim', 'k', 'lr', 'lr_reduce_rate', 'lr_reduce_patience', 'early_stop_patience',
                        'verbose_patience', 'split_seed', 'test']
    model_args_dict = {k: v for k, v in args.__dict__.items() if k not in _other_args_name}
    return model_args_dict


def init_weith_with_gain(modules):
    for m in modules:
        if isinstance(m, LinearBlock):
            torch.nn.init.xavier_uniform_(m.linear.weight, gain=4)


class Architecture(torch.nn.Module):
    def __init__(self, mol_in_dim=15,
                 mol_edge_in_dim=4,
                 hid_dim_alpha=4, e_dim=2048, out_dim=1,
                 mol_block='_NNConv', message_steps=3,
                 mol_readout='GlobalLAPool',
                 pre_norm='_None', graph_norm='_None', flat_norm='_None', end_norm='_None',
                 pre_do='Dropout(0.1)', graph_do='Dropout(0.1)', flat_do='Dropout(0.2)', end_do='Dropout(0.1)',
                 pre_act='RReLU', graph_act='CELU', flat_act='LeakyReLU',
                 graph_res=0):
        super(Architecture, self).__init__()

        hid_dim = mol_in_dim * hid_dim_alpha

       
        fixed_norm_dim = 1  

        self.pre_norm = norm_factory(pre_norm, fixed_dim=fixed_norm_dim)
        self.graph_norm = norm_factory(graph_norm, fixed_dim=fixed_norm_dim)
        self.flat_norm = norm_factory(flat_norm, fixed_dim=fixed_norm_dim)
        self.end_norm = norm_factory(end_norm, fixed_dim=fixed_norm_dim)

        self.pre_do = eval(pre_do)
        self.graph_do = eval(graph_do)
        self.flat_do = eval(flat_do)
        self.end_do = eval(end_do)

        self.pre_act = eval(pre_act)
        self.graph_act = eval(graph_act)
        self.flat_act = eval(flat_act)

        self.mol_block = eval(mol_block)

        self.mol_lin0 = LinearBlock(mol_in_dim, hid_dim,
                                    norm=self.pre_norm, dropout=self.pre_do, act=self.pre_act)

        self.mol_conv = MessageBlock(hid_dim, hid_dim, mol_edge_in_dim,
                                     norm=self.graph_norm, dropout=self.graph_do,
                                     conv=self.mol_block, act=self.graph_act, res=graph_res)

        self.message_steps = message_steps

        self.mol_readout = eval(mol_readout)(in_channels=hid_dim, processing_steps=3)
        _mol_ro = 5 if mol_readout == 'GlobalPool5' else 2

        self.mol_flat = LinearBlock(_mol_ro * hid_dim, e_dim,
                                    norm=self.flat_norm, dropout=self.flat_do, act=self.flat_act)

        self.lin_out1 = LinearBlock(e_dim, out_dim,
                                    norm=self.end_norm, dropout=self.end_do, act=_None)

    def forward(self, data_mol):
        xm = self.mol_lin0(data_mol.x, batch=data_mol.batch)

        hm = None
        for _ in range(self.message_steps):
            xm, hm = self.mol_conv(xm, data_mol.edge_index, data_mol.edge_attr, h=hm, batch=data_mol.batch)

        outm = self.mol_readout(xm, data_mol.batch)
        outm = self.mol_flat(outm)
        out = self.lin_out1(outm)
        return out


Model = Architecture
