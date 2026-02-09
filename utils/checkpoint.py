import os
import torch


def save_checkpoint(state: dict, path: str) -> None:
    """
    Save a training checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    """
    Load a training checkpoint.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)

