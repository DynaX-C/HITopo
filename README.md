# HITopo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19134285.svg)](https://doi.org/10.5281/zenodo.19134285)            
This is the official repository for “Hierarchical Interaction Topology for Physics-Inspired Molecular Representation Learning”.

## Overview

HITopo is a physics-inspired molecular representation framework that organises molecular interactions along two complementary dimensions: interaction type and interaction order. It separates bonded and non-bonded atomic interactions, represents higher-order angular and dihedral relationships through bond- and angle-centred topologies, and enables incidence-guided coupling across interaction orders.

HITopo separates representation topology from the choice of message-passing operator and can therefore be instantiated with different graph neural network backbones, including GCN, GAT, and GINE. Protein–ligand binding affinity prediction is used as the principal validation task, where HITopo consistently improves these backbones across multiple generalisation settings.

This repository provides resources for environment setup, dataset preparation, model training, inference, evaluation, and example applications.

## Installation
We recommend using Conda to manage the environment and dependencies. The required packages can be installed with the provided configuration file:
```bash
conda env create -f environment.yml
conda activate HITopo
```
**Note:** OpenBabel is required for molecular file processing. On Ubuntu/Debian systems, install it with:
```bash
sudo apt update
sudo apt install openbabel
```

## Datasets and Resources
We use three benchmark data-splitting protocols covering different levels of generalization difficulty:

- **Temporal-split setting**: available [here](https://github.com/guaguabujianle/GIGN/tree/main/GIGN/data)
- **Standard PDBbind2020 setting**: available [here](https://github.com/camlab-ethz/GEMS/tree/main/PDBbind_data)
- **CleanSplit protocol**: available [here](https://github.com/camlab-ethz/GEMS/tree/main/PDBbind_data)

The original PDBbind data are publicly available from [here](https://www.pdbbind-plus.org.cn/).  
To facilitate reproduction, the preprocessed datasets and pretrained checkpoints used in this work can be downloaded from [here](https://doi.org/10.5281/zenodo.19134285).

## Build Your Own Dataset
In addition to the provided benchmarks, users can construct their own datasets from raw protein–ligand structures.

### Data format and index file  
The dataset should be organized in a structure similar to PDBbind:
```
/path/to/data/  
├── pdbid_1/  
│ ├── protein.pdb  
│ └── ligand.mol2  
├── pdbid_2/  
│ ├── protein.pdb  
│ └── ligand.mol2
```

In addition, an index file in CSV format is required, containing at least the following columns:

- `pdbid`
- `-logKd/Ki`

If you are using the original PDBbind data, you can convert the provided index file into CSV format using:

```bash
python generate_data_csv.py \
    --index_file ./v2020/index/INDEX_general_PL_data.2020 \
    --output pdbbind2020.csv
```
Optionally, you can define custom data splits by providing a JSON file:
```bash
python generate_data_csv.py \
    --index_file ./v2020/index/INDEX_general_PL_data.2020 \
    --split_json ./data/Data_CleanSplit/PDBbind_data_split_cleansplit.json \
    --split_name train \
    --output train.csv
```

### Data preprocessing
After preparing the structure files and index file, the dataset can be processed in two steps.

#### Step 1: Extract ligand and binding pocket
```bash
python dataprep.py \
    --index_file /path/to/index.csv \
    --work_dir /path/to/data_dir
```

#### Step 2: Construct hierarchical graph representations
```bash
python dataset.py \
    --index_file /path/to/index.csv \
    --work_dir /path/to/data_dir \
    --output dataset.pt
```
The generated `dataset.pt` contains multiple levels of graph representations:
```python
graphs = {
    "atom_graphs": atom_graphs,
    "nb_atom_graphs": nb_atom_graphs,
    "bd_atom_graphs": bd_atom_graphs,
    "bond_graphs": bond_graphs,
    "angle_graphs": angle_graphs,
    "dihedral_graphs": dihedral_graphs
}
```
These graphs correspond to different interaction levels, including atomic interactions, non-bonded interactions, and higher-order geometric structures.

## Training
We provide preprocessed datasets for direct use, which can be downloaded from [here](https://doi.org/10.5281/zenodo.19134285).

For example, to train the GINE backbone on the CleanSplit setting:
```bash
python train.py --config config/config_cleansplit_gine.json
```
To switch to a different backbone, such as GAT:
```bash
python train.py --config config/config_cleansplit_gat.json
```
To train on a different data-splitting protocol, such as the temporal-split setting:
```bash
python train.py --config config/config_temporalsplit_gine.json
```
Different HITopo variants can also be selected using the `--variant` argument, for example:
```bash
python train.py --config config/config_cleansplit_gine.json --variant noci
```
Available variants include `full`, `noci`, `noho`, `unig`, `base`, and `nb`.

The number of HITopo blocks can be adjusted with `--num_layers`, for example:
```bash
python train.py --config config/config_cleansplit_gine.json --num_layers 2
```
Detailed configuration options are provided in the configuration files under `./config/`.

## Evaluation
To facilitate evaluation, we provide pretrained checkpoints that can be downloaded from [here](https://doi.org/10.5281/zenodo.19134285).

For example, to evaluate a model on the CleanSplit setting:
```bash
python test.py --config config/config_cleansplit_gine.json --model_paths '["./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f0.pt"]'
```
Multi-model ensemble prediction is also supported. For example, to evaluate using five checkpoints:
```bash
python test.py --config config/config_cleansplit_gine.json --model_paths '["./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f0.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f1.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f2.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f3.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f4.pt"]'
```
Other options (e.g., `--variant`, `--num_layers`) are consistent with those used during training.

## Inference
To run prediction on a single protein–ligand complex:
```bash
python inference.py \
    --config ./config/config_cleansplit_gine.json \
    --protein_pdb /path/to/your_protein.pdb \
    --ligand_mol2 /path/to/your_ligand.mol2 \
    --model_paths '["./checkpoints/..."]'
```
The input protein should be provided in PDB format and the ligand in MOL2 format.

## Physical Consistency Analysis
We also provide a separate preprocessing pipeline for the physical consistency analysis.

### Data preprocessing
The overall procedure is similar to the standard dataset construction pipeline, but uses different scripts:
```bash
python dataprep_min.py \
    --index_file /path/to/index.csv \
    --work_dir /path/to/data_dir

python dataset_min.py \
    --index_file /path/to/index.csv \
    --work_dir /path/to/data_dir \
    --output dataset.pt
```
This pipeline depends on **AmberTools** for energy minimization before graph construction. Please note that this step can be computationally expensive and may take a substantial amount of time, especially for large datasets. To facilitate reproduction, the preprocessed data used for this analysis can be downloaded from [here](https://doi.org/10.5281/zenodo.19134285).

### Energy calculation
After energy minimization, MM/GBSA calculations can be performed using:
```bash
bash analysis/run_gbsa.sh ./data_processed/CASF16_min
```
This step computes residue-level energy decomposition, which is used for the physical consistency analysis. Please ensure that **AmberTools** is properly installed and configured before running this step.

### Model attribution
After obtaining the processed dataset and residue-level energy decomposition, model attribution can be computed using:
```bash
python residue_attribution.py \
    --config ./config/config_cleansplit_gine.json \
    --graph_pt_path ./data_processed/residue_attr/casf16.pt \
    --save_dir ./residue_attr_casf16 \
    --model_paths '["./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f0.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f1.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f2.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f3.pt", "./checkpoints/cleansplit_gine/best_full_gine_cleansplit_f4.pt"]'
```

### Quantitative analysis
After computing model attribution and MM/GBSA energy decomposition, several quantitative analyses can be performed.

#### 1. Extract matched residue-level contributions
First, extract the corresponding residue-level contributions from MM/GBSA and model attribution:
```bash
python analysis/summary.py \
    --model_attr_dir residue_attr_casf16/ \
    --work_dir data_processed/CASF16_min/ \
    --out_dir ./GBSA_casf16
```

#### 2. Compute Spearman correlation
Then, compute the per-complex Spearman correlation between model attribution and MM/GBSA residue contributions:
```bash
python analysis/spearman.py \
    --attr_dir residue_attr_casf16/ \
    --energy_dir GBSA_casf16 \
    --energy_suffix pocket_decomp \
    --out_csv spearman_per_complex_casf16.csv
```

#### 3. Compute sign agreement accuracy and favorable-residue recall
Finally, evaluate sign agreement accuracy and favorable-residue recall:
```bash
python analysis/accuracy_recall.py \
    --attr_dir residue_attr_casf16/ \
    --energy_dir GBSA_casf16/ \
    --out_csv accuracy_recall.csv
```
These analyses quantify the agreement between model attribution and physics-based residue-level energy patterns from MM/GBSA.
