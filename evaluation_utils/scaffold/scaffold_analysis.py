import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    """Convert a SMILES to Murcko scaffold SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    return None

def load_scaffolds_from_csv(path, smiles_column='SMILES'):
    df = pd.read_csv(path)
    scaffolds = set()
    for s in df[smiles_column]:
        try:
            scaffold = get_scaffold(s)
            if scaffold:
                scaffolds.add(scaffold)
        except Exception:
            continue
    return scaffolds

# === Load scaffold sets ===
chem_g4_scaffolds = load_scaffolds_from_csv("ChemE-G4.csv")
pocket2mol_scaffolds = load_scaffolds_from_csv("pocket2mol.csv")
g4ldb_scaffolds = load_scaffolds_from_csv("G4LDB.csv")

# === Analysis function ===
def scaffold_stats(model_name, gen_scaffolds, g4ldb_scaffolds):
    shared = gen_scaffolds & g4ldb_scaffolds
    novel = gen_scaffolds - g4ldb_scaffolds
    total = len(gen_scaffolds)
    shared_n = len(shared)
    novelty_pct = 100 * len(novel) / total if total > 0 else 0

    print(f"\nModel: {model_name}")
    print(f"  # Generated Scaffolds: {total}")
    print(f"  # Shared with G4LDB:   {shared_n}")
    print(f"  Scaffold Novelty (%):  {novelty_pct:.1f}")

# === Run for both models ===
scaffold_stats("ChemE-G4", chem_g4_scaffolds, g4ldb_scaffolds)
scaffold_stats("Pocket2Mol", pocket2mol_scaffolds, g4ldb_scaffolds)
