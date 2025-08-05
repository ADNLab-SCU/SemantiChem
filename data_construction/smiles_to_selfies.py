# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd  # 确保导入 pandas
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import matplotlib.pyplot as plt
import selfies as sf
import re

data = pd.read_csv("G4LDB.csv")


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None  # 对于无效的SMILES，返回None
def is_valid_smiles(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        return False
    if re.search(r'[<>]', smiles):
        return False
    return True


# 对两个数据集的SMILES进行标准化
data['canonical_smiles'] = data['smiles'].apply(canonicalize_smiles)
data.to_csv("smiles.csv", index = False)
data['selfies']=data['canonical_smiles'].apply(lambda x: sf.encoder(x) if is_valid_smiles(x) else None)

data.to_csv("sf.csv", index = False)

print("计算完成，结果已保存")
