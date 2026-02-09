# PyTorch Research Template

This repository provides a lightweight, research-oriented PyTorch template
for training and evaluating neural networks with an emphasis on clarity,
reproducibility, and extensibility.

The code structure reflects how I typically prototype and run machine learning
experiments in research settings. It is intentionally minimal and designed to
be easy to read, modify, and extend.

This repository is meant as a **code sample / template**, not as a production
framework or a source of novel research results.

## Features
- Deterministic training via centralized seeding
- Clean training and evaluation loops
- Modular model and utility structure
- Simple checkpointing and logging
- YAML-based experiment configuration
- Explicit runtime measurement

## Repository Structure

pytorch-research-template/
├── README.md
├── requirements.txt
├── train.py
├── eval.py
├── models/
│   └── mlp.py
├── data/
│   └── dummy_dataset.py
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   ├── logger.py
│   └── checkpoint.py
└── configs/
    └── default.yaml


- `train.py` / `eval.py`: minimal training and evaluation entry points  
- `models/`: example model definitions  
- `data/`: small synthetic dataset for demonstration purposes  
- `utils/`: reusable utilities (seeding, logging, metrics, checkpoints)  
- `configs/`: YAML configuration files for experiments  

## Usage

```bash
pip install -r requirements.txt
python train.py
python eval.py

