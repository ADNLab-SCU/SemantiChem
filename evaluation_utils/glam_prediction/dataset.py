import torch
import numpy as np
from rdkit import Chem
from torch_scatter import scatter


def one_of_k_encoding(x, allowable_set):
    """One-hot encoding for categorical features"""
    if x not in allowable_set:
        pass  
    return list(map(lambda s: x == s, allowable_set))


def get_mol_nodes_edges(mol):
    """Convert RDKit molecule to node features, edge indices and edge features"""
    N = mol.GetNumAtoms()
    atom_type = []
    atomic_number = []
    aromatic = []
    hybridization = []
    
    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    
    edge_index = torch.LongTensor([row, col])
    
    edge_type = [one_of_k_encoding(t, [Chem.rdchem.BondType.SINGLE, 
                                       Chem.rdchem.BondType.DOUBLE, 
                                       Chem.rdchem.BondType.TRIPLE, 
                                       Chem.rdchem.BondType.AROMATIC]) 
                 for t in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    row, col = edge_index

    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) 
                   for t in atom_type]
    
    x_hybridization = [one_of_k_encoding(h, [Chem.rdchem.HybridizationType.SP, 
                                             Chem.rdchem.HybridizationType.SP2, 
                                             Chem.rdchem.HybridizationType.SP3]) 
                       for h in hybridization]
    
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    
    x = torch.cat([torch.FloatTensor(x_atom_type), 
                   torch.FloatTensor(x_hybridization), 
                   x2], dim=-1)

    return x, edge_index, edge_attr