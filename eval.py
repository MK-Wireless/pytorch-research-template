import time
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from models.mlp import MLP
from data.dummy_dataset import DummyClassificationDataset
from utils.seed import seed_everything
from utils.metrics import accuracy
from utils.checkpoint import load_checkpoint


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    cfg = load_config("configs/default.yaml")

    # Seed (for deterministic dataset split)
    seed_everything(cfg["seed"])

    # Device
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Dataset (same as train)
    dataset = DummyClassificationDataset(
        num_samples=cfg["data"]["num_samples"],
        num_features=cfg["data"]["num_features"],
        num_classes=cfg["data"]["num_classes"],
    )

    n_train = int(len(dataset) * cfg["data"]["train_split"])
    n_val = len(dataset) - n_train
    _, val_set = random_split(dataset, [n_train, n_val])

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # Model
    model = MLP(
        input_dim=cfg["data"]["num_features"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["data"]["num_classes"],
    ).to(device)

    # Load checkpoint
    ckpt_path = f"{cfg['checkpoint']['dir']}/best.pt"
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Eval
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total if total > 0 else 0.0
    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f}s")

