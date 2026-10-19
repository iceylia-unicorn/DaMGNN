# DaMGNN-PPI

PyTorch/DGL implementation of **Degree-Adaptive Multi-view Graph Learning for Protein-Protein Interaction Prediction**.

## Repository layout

```
.
├── models5.py
├── train5.py
├── models_mape.py
├── configs/
│   └── param_configs.example.json
├── DATA_FORMAT.md
├── requirements.txt
└── environment.yml
```

## Installation

Create an environment first:

```bash
conda env create -f environment.yml
conda activate damgnn
```

Install PyTorch for your hardware from the [official selector](https://pytorch.org/get-started/locally/), then install the matching PyG extension wheels following the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Install the remaining packages with:

```bash
pip install -r requirements.txt
```

Use matching PyTorch, DGL, and PyG builds for your CUDA version.

## Data and checkpoints

Place datasets, checkpoints, and outputs under `data/`, `checkpoints/`, and `results/`.

The input bundle format is described in [DATA_FORMAT.md](DATA_FORMAT.md).

## Running an experiment

Example:

```bash
python train5.py --dataset SHS27k --split_mode random --seed 0 \
  --ckpt_path checkpoints/vae_model.ckpt
```

See `configs/param_configs.example.json` for parameters.

## Citation and licence

Citation metadata is provided in [CITATION.cff](CITATION.cff). This repository is provided under the [MIT License](LICENSE).
