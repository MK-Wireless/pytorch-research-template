# PyTorch Research Template

This repository provides a lightweight, research-oriented PyTorch template
for training and evaluating neural networks with an emphasis on clarity,
reproducibility, and extensibility.

The code structure reflects how I typically prototype and run machine learning
experiments in research settings.

## Features
- Deterministic training via centralized seeding
- Clean training and evaluation loops
- Modular model and utility structure
- Simple checkpointing and logging
- YAML-based experiment configuration

## Usage
```bash
pip install -r requirements.txt
python train.py
python eval.py
