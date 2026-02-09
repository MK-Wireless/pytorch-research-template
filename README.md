# PyTorch Research Template

A lightweight, research-oriented PyTorch template for training and evaluating neural networks with an emphasis on:
- clarity
- reproducibility
- clean structure
- minimal dependencies

This is intentionally **not** a production framework. It’s a readable baseline you can adapt for research experiments.

## Repo Structure

pytorch-research-template/
├── README.md
├── requirements.txt
├── train.py
├── eval.py
├── models/
│ └── mlp.py
├── data/
│ └── dummy_dataset.py
├── utils/
│ ├── seed.py
│ ├── metrics.py
│ ├── logger.py
│ └── checkpoint.py
└── configs/
└── default.yaml


## Features
- Deterministic training via centralized seeding
- Clean train/eval loops
- Simple YAML config
- Minimal logging + timing
- Checkpoint save/load

## Quickstart

```bash
pip install -r requirements.txt
python train.py
python eval.py
