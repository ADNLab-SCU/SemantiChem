import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# === Molecular Property Calculation Function ===
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MW": Descriptors.MolWt(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol)
    }

# === Process a Single File ===
def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    prop_list = []

    for smi in df["SMILES"]:
        try:
            props = compute_properties(smi)
            if props:
                prop_list.append(props)
            else:
                prop_list.append({k: None for k in ["LogP", "TPSA", "MW", "HBD", "HBA"]})
        except:
            prop_list.append({k: None for k in ["LogP", "TPSA", "MW", "HBD", "HBA"]})

    props_df = pd.DataFrame(prop_list)
    result_df = pd.concat([df, props_df], axis=1)
    result_df.to_csv(output_path, index=False)

    print(f"\n=== {input_path} Property Summary ===")
    summary = props_df.describe().loc[["mean", "std"]]
    print(summary.round(3))

# === File Path Configurations ===
file_configs = [
    ("Mpro.csv", "Mpro_with_props.csv"),
    ("ChemE-Mpro.csv", "ChemE-Mpro_with_props.csv"),
]

# === Run Processing ===
for input_file, output_file in file_configs:
    process_file(input_file, output_file)
