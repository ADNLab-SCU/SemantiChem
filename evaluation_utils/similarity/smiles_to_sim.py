# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd  
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import matplotlib.pyplot as plt

# change it to fit your workflow
csv1_path = "./v2_finalrun.csv"
csv2_path = "./g4ldb_training.csv"
csv_output_path = "./v2_finalsim.csv"

data1 = pd.read_csv(csv1_path)
data2 = pd.read_csv(csv2_path)


# MACCS
def compute_fingerprints(data):
    names = []
    morgan_fps = []
    names_smiles_error = []
    smiles_error = []

    for index, row in data.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                names.append(row['Id'])  # "id"
                morgan_fps.append(fp)
            else:
                names_smiles_error.append(row['Id'])
                smiles_error.append(row['SMILES'])
        except Exception as e:
            names_smiles_error.append(row['Id'])
            smiles_error.append(row['SMILES'])

    return names, morgan_fps, names_smiles_error, smiles_error


names1, morgan_fps1, names_smiles_error1, smiles_error1 = compute_fingerprints(data1)
names2, morgan_fps2, names_smiles_error2, smiles_error2 = compute_fingerprints(data2)


max_similarity_records = []

for i in range(len(morgan_fps1)):

    similarities = DataStructs.BulkTanimotoSimilarity(morgan_fps1[i], morgan_fps2)

    max_similarity = max(similarities)
    max_index = similarities.index(max_similarity)
    

    max_similarity_records.append({
        'csv1_id': names1[i],
        'csv1_smiles': data1.loc[i, 'SMILES'],
        'csv2_id': names2[max_index],
        #'csv2_smiles': data2.loc[max_index, 'canonical_smiles'],
        'csv2_smiles': data2.loc[max_index, 'SMILES'],
        'similarity': max_similarity
    })


max_similarity_df = pd.DataFrame(max_similarity_records)
max_similarity_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

print("finished:", csv_output_path)
