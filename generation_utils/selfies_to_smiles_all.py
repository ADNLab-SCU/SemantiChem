import os
import numpy as np
import pandas as pd  
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import matplotlib.pyplot as plt
import selfies as sf
import re

def decode_selfies(selfies):
    try:
        return sf.decoder(selfies)
    except:
        return "INVALID"  

def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return "INVALID"
    mol = Chem.MolFromSmiles(str(smiles))
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return "INVALID"  

def is_valid_smiles(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        return False
    if re.search(r'[<>]', smiles):
        return False
    return True


current_directory = os.getcwd()


for filename in os.listdir(current_directory):

    if filename.endswith(".csv"):

        data = pd.read_csv(filename)
        

        data['smiles'] = data['Selfies'].apply(decode_selfies)
        data['SMILES'] = data['smiles'].apply(canonicalize_smiles)
        

        data.to_csv(filename, index=False)
        
        print(f"{filename} is processed.")

print("finished.")
