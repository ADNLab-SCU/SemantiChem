# SemantiChem: Companion Repository

## _“Function-Driven Molecular Design Enabled by Instruction-Tuned Large Language Models”_

This repository provides the source code supporting **SemantiChem**, an instruction-tuned generative framework for *function-driven molecular design*. The framework maps functional design objectives expressed in natural language directly to chemically meaningful molecular structures, without relying on predefined geometric constraints, molecular scaffolds, or pocket-centric assumptions.

The codebase supports molecular generation, evaluation, and analysis workflows used to study biomolecular targets exhibiting different recognition regimes, including nucleic acid G-quadruplexes, RNA structures, and protein targets. It implements the computational components underlying the generative, evaluation, and benchmarking procedures described in the manuscript.



- **Molecular generation** via instruction-tuned LLMs (based on the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)),
- **Activity prediction** using a graph neural network (GNN) adapted from [GLAM](https://github.com/yvquanli/GLAM),
- **Scaffold-level evaluation** and **similarity-based visualization**.

### Note
This repository contains the source code used in the computational experiments. Model weights and full computational outputs are provided separately (see below).

All proprietary or curated datasets used in the manuscript are provided in **Supplementary Data** upon submission.
Public datasets (e.g., PubChem10M SELFIES, G4LDB 3.0) are linked here for convenience.

---

## 📁 Directory Structure
```
├── data_utils/
│   ├── smiles_to_selfies.py
│   ├── type1.py
│   ├── type2.py
│   └── type3.py
├── generation_utils/
│   ├── request.py
│   └── selfies_to_smiles_all.py
├── evaluation_utils/
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

## 🔬 Methodological Workflow Overview

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

### 🖥 Software Environment and Execution

The code in this repository has been tested under the following environment:
- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- Transformers 4.38
- RDKit 2023.09
- Operating Systems: Ubuntu 22.04, Windows 11 (WSL2)

Hardware
- A GPU (≥16GB VRAM) is recommended for efficient inference
- CPU-only mode is supported for small-scale or demo runs

---

### ⚙ Installation Guide

```
conda create -n semantichem python=3.10
conda activate semantichem
pip install -r requirements.txt
```

Typical installation time: ~10 minutes on a standard desktop computer with internet access.

After installation, no internet connection is required except for the initial download of model weights from Hugging Face.

---

### ▶ Minimal Inference Example (for peer review)

This repository contains the source code used in the computational experiments.
Model weights are hosted on Hugging Face and are accessible to editors and reviewers during peer review via a read-only access token.

Step 1 — Authenticate with Hugging Face

huggingface-cli login --token **<REVIEW_TOKEN>**

Step 2 — Run a minimal generation example

Example prompt (similar to those used in the manuscript and Supplementary Information):

“Please show me a G-quadruplex ligand which hasn’t been reported. Be sure the structure is novel and unique.”

Run inference using the generation script in this repository:

```
python 2.molecule_generation/request.py \
  --model ADNLab-SCU/Instruct-G4 \
  --prompt "Please show me a G-quadruplex ligand which hasn’t been reported. Be sure the structure is novel and unique."
```

Expected output

The script produces a list of generated molecules represented as SELFIES and/or SMILES strings, for example:
- SELFIES: [C][=C][N][Ring1][=C]...
- SMILES: C1=CC=NC2=NC=NC(=N2)N1

Expected runtime

| Mode | Runtime|
|------|--------|
| GPU	| ~1–2 minutes|
| CPU	| ~5–10 minutes|


---

### 🧪 Running on Your Own Prompts

You may substitute your own instruction text:

```
python 2.molecule_generation/request.py \
  --model ADNLab-SCU/Instruct-G4 \
  --prompt "<YOUR_PROMPT>"
```

Batch prompts can be constructed following the prompt templates described in the manuscript and Supplementary Note 2.

---

### 🔗 Relationship Between Code, Model Weights, and Supplementary Data

The complete computational workflow described in the manuscript is distributed across the following components:

| Component | Location | Role in Reproducibility |
|-----------|----------|-------------------------|
| Source code | This GitHub repository | Data construction, molecular generation, and evaluation scripts |
| Model weights | Hugging Face (ADNLab-SCU models) | Finetuned LLM checkpoints used for molecular generation |
| Training configurations | Supplementary Note 2 | YAML files defining model training and fine-tuning setup |
| Prompt templates | Supplementary Note 2.4 | Instruction templates used for generation |
| Complete generation outputs | Supplementary Note 0.2 | Full sets of molecules generated by all models |
| Evaluation results | Supplementary Notes 3–8 | GNN predictions, docking scores, TMAP embeddings, and benchmarking data |

The Supplementary Data archive (hosted on Figshare and linked in the manuscript) contains all intermediate and final computational outputs required to reproduce the quantitative analyses presented in the paper.

---

### 🔁 Reproducibility Pathway

A typical reproduction workflow is:
1.	Use prompt templates from **Supplementary Note 2.4**
2.	Run inference with finetuned model weights (Hugging Face) using scripts in **2.molecule_generation/**
3.	Evaluate generated molecules using modules in **3.evaluation/**
4.	Compare outputs with reference datasets provided in **Supplementary Notes 0 and 3–8**

This structure ensures that the generation, evaluation, and benchmarking procedures reported in the manuscript can be independently verified.

---

## 📑 Citation

*This project is part of ongoing research. Please cite the corresponding paper when available.*

```bibtex
@article{ADNLab2025ChemE,
  title = {Function-Driven Molecular Design Enabled by Instruction-Tuned Large Language Models},
  author = {ADNLab},
  journal = {To be submitted},
  year = {2025}
}
```