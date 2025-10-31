# 🧬 LiverBind: Deep Learning–Based Profiling of Hepatotoxic Environmental Pollutants Across the Human Liver Proteome

**LiverBind** is a deep learning framework designed to predict protein–compound binding affinities across the human liver proteome. The model integrates multimodal representations of small molecules and liver proteins, combining graph neural networks for structures with sequence–structure embeddings for sequences. Trained on experimentally verified interactions, LiverBind provides high-accuracy affinity predictions and interpretable binding site insights.
 It enables proteome-scale screening of environmental pollutants to identify hepatotoxic candidates and vulnerable liver targets, offering a foundation for large-scale toxicological risk assessment.

------

## ⚙️ Environment Setup

### 1. Clone the Repository

```
git clone https://github.com/yourusername/LiverBind.git
cd LiverBind
```

### 2. Create Environment

It is recommended to use **Python ≥ 3.9** and **PyTorch ≥ 2.0**.
 Install dependencies:

```
pip install torch numpy pandas scikit-learn tqdm rdkit biopython matplotlib
```

------

## 🧩 External Dependencies for Protein Preprocessing

The script `protein_preprocess.py` requires several external bioinformatics tools for structural and evolutionary feature extraction.

| Program                | Purpose                           | Path in Code                                                 | Notes                                                        |
| ---------------------- | --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DSSP**               | Secondary structure calculation   | `dssp_executable_path = "./bin/dssp"`                        | Download binary from [DSSP official site](https://swift.cmbi.umcn.nl/gv/dssp/) |
| **PSI-BLAST**          | Generate PSSM profiles            | `psiblast_executable_path = "./Blast/ncbi-blast-2.13.0+/bin/psiblast"` | Install [NCBI BLAST+ ≥ 2.13.0](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) |
| **PSI-BLAST database** | Swiss-Prot database for PSI-BLAST | `psiblast_db_file = "./Blast/ncbi-blast-2.13.0+/database/swissprot"` | Build or download Swiss-Prot (see NCBI documentation)        |
| **HHblits**            | HMM profile generation            | `hhblits_db = "./hh-suite/databases/uniclust30_2018_08/uniclust30_2018_08"` | Install HH-suite 3, use [UniClust30](https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/) |

> 🧠 Before running `protein_preprocess.py`, ensure all executables and databases are correctly installed and the paths updated in the script.

------

## 📘 Data Preparation

All required data files should be placed in the `data/` directory:

- `train_dataset.csv` and `test_dataset.csv`
   Contain compound–protein pairs and their corresponding binding affinity values.

- `train_protein/`
   Must contain **pre-downloaded protein structure files (PDB format)**.
   Example:

  ```
  data/train_protein/P12345.pdb
  data/train_protein/Q9XYZ1.pdb
  ```

Ensure that protein IDs in the CSV file correspond to PDB file names.

------

## 🚀 How to Run

### **Training**

To start model training:

```
bash scripts/train.sh
```

### Testing

To evaluate a trained model:

```
bash scripts/test.sh
```

------

## 🧠 Configuration

Global parameters (e.g., learning rate, dropout, task type) are managed in `options.py`.

You can modify the task type:

```
task = "regression"       # For binding affinity prediction
# task = "classification"  # For toxicity classification
```

All other hyperparameters such as `batch_size`, `lr_initial`, and `dropout` are preconfigured and can be overridden via command line if needed.

------

## 🧪 Notes

- Ensure that **protein structures** are pre-downloaded and named consistently with the dataset.
- Install and configure all **external dependencies** (DSSP, PSI-BLAST, HHblits) before running protein preprocessing.
- Both **regression** and **classification** tasks are supported depending on the dataset and task configuration.
