# GSCF-Net (Graph–SMILES Cross-attention Fusion Network)
GSCF-Net fuses graph-structured representations (molecular graphs) and token-level
representations from a language model (e.g. ChemBERTa) using cross-attention. The
hybrid architecture is designed for molecular property prediction tasks. This
repository contains training/evaluation scripts, dataset wrappers, and model
implementations used for experiments and reproducible evaluation.

## Key features

- Cross-attention fusion between a GNN encoder and a language model encoder
- Support for alpha/beta attention-ratio experiments (control direction/weighting)
- Integration with ChemBERTa (RoBERTa-family) tokenizer/model
- Differential learning rates for GNN / LM / fusion-head parameter groups
- Evaluation on common molecular property datasets (BBBP, Tox21, ClinTox, HIV, BACE, SIDER)

## Repository structure

- `GSCF-Net_finetune.py` — main training/evaluation script (ratio/attention experiments)
- `models/` — model implementations (e.g. `cross_attention_ratio.py`)
- `dataset/` — dataset wrappers and preprocessing (`hybrid_dataset.py`)
- `data/` — dataset files (expected file locations used by the scripts)
- `ChemBERTa-77M-MLM/` — local ChemBERTa tokenizer/model files (optional)
- `KIIT_Chemical_draft.pdf` — paper/draft describing the method and equations

## Quick Start

```bash
git clone https://github.com/thkim-01/GSCF-Net.git
cd GSCF-Net
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a torch build that matches your CUDA (or CPU-only as needed)
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas numpy tqdm tensorboard
# Optional: conda install -c conda-forge rdkit
```

Download a dataset into `data/` (see "Supported datasets" below), then run:

```bash
python GSCF-Net_finetune.py --dataset BBBP --config config_cross_attention_ratio.yaml --repeats 3
```

## Configuration

## Supported datasets

The scripts use `DATASET_CONFIGS` defined inside the main training script. Expected
data locations (relative to the repository root):

- BBBP — `data/bbbp/BBBP.csv`
- Tox21 — `data/tox21/tox21.csv` (multi-target)
- ClinTox — `data/clintox/clintox.csv` (multi-target)
- HIV — `data/hiv/HIV.csv`
- BACE — `data/bace/bace.csv`
- SIDER — `data/sider/sider.csv` (multi-target)

Datasets are wrapped by `dataset/hybrid_dataset.py`. SMILES tokenization uses
`RobertaTokenizer` (ChemBERTa) as configured by `cross_attention_specific`.
