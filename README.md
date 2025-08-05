# 🧬 Task-Driven Discovery of Nucleic Acid Ligands Using Large Language Models

SemantiChem is a task-driven framework for nucleic acid ligand generation using large language models (LLMs). This repository presents a modular and reproducible workflow for instruction fine-tuning and evaluation of large language models (LLMs) to generate chemically valid molecules targeting nucleic acid (NA) structures, with a focus on G-quadruplex (G4) motifs. The pipeline integrates:

- **Molecular generation** via instruction-tuned LLMs (based on the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)),
- **Activity prediction** using a graph neural network (GNN) adapted from [GLAM](https://github.com/yvquanli/GLAM),
- **Scaffold-level evaluation** and **similarity-based visualization**.

---

## 📁 Directory Structure
```
├── 1.data_construction/
│   ├── smiles_to_selfies.py
│   ├── type1.py
│   ├── type2.py
│   └── type3.py
├── 2.molecule_generation/
│   ├── request.py
│   └── selfies_to_smiles_all.py
├── 3.evaluation/
│   ├── glam_prediction/
│   ├── property/
│   ├── scaffold/
│   ├── similarity/
│   └── tmap/
├── LICENSE
├── requirements.txt
├── README.md
```
---

## 🚀 How to Reproduce

### Step 1: Construct Q&A Data

Convert G4 ligand SMILES into SELFIES and construct instruction-style Q&A pairs using pre-defined templates (e.g., `type1`, `type2`, `type3`).

### Step 2: Pretrain and Fine-Tune LLMs

We explored two distinct LLMs for scaffold-aware molecular generation:

- **Instruct-G4**: Based on [Meta-LLaMA-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), a general-purpose model without domain-specific pretraining.  
- **ChemE-G4**: Based on *LLaMA-3.1-ChemEinstein* ([ZeroXClem/Llama3.1-BestMix-Chem-Einstein-8B](https://huggingface.co/ZeroXClem/Llama3.1-BestMix-Chem-Einstein-8B)), pretrained on SMILES data for structural fluency.

Training pipeline:

1. **SELFIES Pretraining**: LoRA adaptation using ~10k molecules from PubChem10M_SELFIES.  
2. **Instruction Tuning**: Fine-tuning on curated G4 ligands with scaffold-conditioned prompts.

🔗 Finetuned models (available on Hugging Face):

- [Instruct-G4](https://huggingface.co/ADNLab-SCU/Instruct-G4)  
- [ChemE-G4](https://huggingface.co/ADNLab-SCU/ChemE-G4)

### Step 3: Generate Molecules

Use API-based inference to generate SELFIES-format molecules, then convert to SMILES for downstream analysis.

### Step 4: Evaluate Results

- **Activity prediction** using a GLAM-based GNN model  
- **Molecular properties**: QED, SA score  
- **Scaffold retention and novelty**  
- **Similarity and diversity** analysis via MACCS + TMAP  

---

## 📊 Dataset Overview

The pipeline uses the following datasets:

- **G4LDB Ligands**: Expert-curated G4-binding molecules from [G4LDB 3.0](https://www.g4ldb.com/#/).
- **SELFIES Pretraining Set**: A ~10k molecule subset from [PubChem10M_SELFIES](https://huggingface.co/datasets/alxfgh/PubChem10M_SELFIES).
- **Generated Molecules**: Created by fine-tuned LLMs (SELFIES format), then canonicalized as SMILES for evaluation.

Users may substitute datasets as long as formats are consistent.

---

## 💡 Notes for Usage

- Model training and inference follow the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) API and YAML configuration format.
- GNN-based prediction is located in `3.evaluation/glam_prediction/`, with pretrained checkpoints provided.
- Scaffold and similarity evaluation modules rely on RDKit and MACCS fingerprints.
- The entire workflow is modular and components can be executed independently or as a pipeline.

---

## 📑 Citation

*This project is part of ongoing research. Please cite the corresponding paper when available.*

```bibtex
@article{ADNLab2025ChemE,
  title = {Task-Driven Discovery of Nucleic Acid Ligands Using Large Language Models},
  author = {ADNLab},
  journal = {To be submitted},
  year = {2025}
}



