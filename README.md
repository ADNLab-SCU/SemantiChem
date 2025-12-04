# SemantiChem: Companion Code Repository (Submission Version)

### Note

This repository provides a minimal set of utility scripts used in the computational experiments of our manuscript.
It contains **code only**, and does not include training scripts, datasets, model checkpoints, or full pipeline documentation.

The codebase includes modules for:

-  basic molecular format conversion (e.g., SMILES ↔ SELFIES),
- RDKit-based property calculations,
- simple scaffold and similarity analyses,
- optional interfaces for running model inference.

All curated or proprietary datasets used in the study are provided in the **Supplementary Data** upon submission.
A complete and fully documented version of this repository will be released upon publication.

---

## Directory Structure
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

## Usage

Each module can be run independently.
Comments within the scripts describe expected inputs and outputs.
Additional experimental details and intermediate data are provided in the Supplementary Information accompanying the manuscript.
