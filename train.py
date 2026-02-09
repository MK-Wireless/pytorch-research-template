import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.mlp import MLP
from data.dummy_dataset import DummyClassificationDataset
from utils.seed import seed_everything
from utils.metrics import accuracy
from utils.logger import AverageMeter, Timer, log_epoch
from utils.checkpoint import save_checkpoint


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    # Load config
    cfg = load_config("configs/default.yaml")

    # Seed
    seed_everything(cfg["seed"])

    # Device
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = DummyClassificationDataset(
        num_samples=cfg["data"]["num_samples"],
        num_features=cfg["data"]["num_features"],
        num_classes=cfg["data"]["num_classes"],
    )

    n_train = int(len(dataset) * cfg["data"]["train_split"])
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
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

    # Optimizer & loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    timer = Timer()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        timer.start()
        model.train()
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("acc")

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(accuracy(logits, y), x.size(0))

        # Validation
        model.eval()
        val_acc_meter = AverageMeter("val_acc")
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_acc_meter.update(accuracy(logits, y), x.size(0))

        elapsed = timer.elapsed()

        log_epoch(
            epoch,
            {
                "loss": loss_meter.avg,
                "acc": acc_meter.avg,
                "val_acc": val_acc_meter.avg,
            },
            elapsed,
        )

        # Checkpoint
        if cfg["checkpoint"]["save_best"] and val_acc_meter.avg > best_val_acc:
            best_val_acc = val_acc_meter.avg
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                },
                f"{cfg['checkpoint']['dir']}/best.pt",
            )

    print(f"Training finished. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f}s")

