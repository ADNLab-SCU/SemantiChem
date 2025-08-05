# predict_batch.py

import os
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data

from dataset import get_mol_nodes_edges
from model import Architecture

# ===== Load model from checkpoint =====
def load_model_from_ckpt(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = {
        'mol_in_dim': 15,
        'mol_edge_in_dim': 4,
        'hid_dim_alpha': 4,
        'e_dim': 512,
        'out_dim': 1,
        'mol_block': '_TripletMessageLight',
        'message_steps': 3,
        'mol_readout': 'GlobalPool5',
        'pre_norm': '_None',
        'graph_norm': '_None',
        'flat_norm': '_LayerNorm',
        'end_norm': '_LayerNorm',
        'pre_do': 'Dropout(0.1)',
        'graph_do': '_None()',
        'flat_do': 'Dropout(0.1)',
        'end_do': 'Dropout(0.1)',
        'pre_act': 'RReLU',
        'graph_act': 'CELU',
        'flat_act': 'LeakyReLU',
        'graph_res': 1,
    }

    model = Architecture(**args).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, device

# ===== Convert SMILES string to PyG graph =====
def smiles_to_data(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        x, edge_index, edge_attr = get_mol_nodes_edges(mol)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data
    except:
        return None

# ===== Batch prediction from CSV =====
def predict_from_csv(ckpt_path, input_csv, output_csv, smiles_col="SMILES", id_col="id"):
    df = pd.read_csv(input_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column: '{smiles_col}' in input CSV.")
    if id_col not in df.columns:
        raise ValueError(f"Missing column: '{id_col}' in input CSV.")

    model, device = load_model_from_ckpt(ckpt_path)
    smiles_list = df[smiles_col].tolist()

    preds = []
    skipped = []

    with torch.no_grad():
        for i, smi in enumerate(tqdm(smiles_list, desc="🔬 Predicting")):
            data = smiles_to_data(smi)
            if data is None:
                preds.append(None)
                skipped.append((i, smi))
                continue
            data = data.to(device)
            output = model(data)
            score = torch.sigmoid(output).item()
            preds.append(score)

    df["pred_score"] = preds
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Prediction complete. Results saved to: {output_csv}")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} invalid SMILES. Examples:")
        for idx, smi in skipped[:5]:
            print(f"  Row {idx}: {smi}")

# ===== Main script entry =====
if __name__ == "__main__":
    # === Configure paths here ===
    ckpt_path = "./checkpoint/best_save.ckpt"         # Path to your model checkpoint
    input_csv = "ChemE_g4_free_10t_09p_converted.csv"               # Input CSV with ID and SMILES
    output_csv = "predicted_results.csv"            # Output CSV with predictions

    predict_from_csv(ckpt_path, input_csv, output_csv)
