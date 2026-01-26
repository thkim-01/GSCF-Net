# GSCF-Net (Graph–SMILES Cross-attention Fusion Network)

## Requirements (short)

- Python 3.7+
- PyTorch (GPU-enabled build recommended for training)
- transformers (Hugging Face)
- scikit-learn, numpy, pandas, tqdm, tensorboard
- RDKit (optional, recommended for molecule preprocessing)

Example installation (virtual environment recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a torch build that matches your CUDA (or CPU-only as needed)
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas numpy tqdm tensorboard
# For RDKit, we recommend conda: conda install -c conda-forge rdkit
```

## Quick start

Assuming datasets and a YAML config are prepared, a typical training run is:

```bash
# Single dataset (example: BBBP) using the attention-ratio config
python GSCF-Net_finetune.py --dataset BBBP --config config_cross_attention_ratio.yaml --repeats 3

# Run all supported datasets
python GSCF-Net_finetune.py --dataset all --config config_cross_attention_ratio.yaml --repeats 3
```

Outputs produced by the script:

- TensorBoard logs: `runs_ratio/<task_name>_<timestamp>/`
- Best model checkpoint: `runs_ratio/.../best_model.pth`
- CSV results (detailed + summaries): saved under `results/`