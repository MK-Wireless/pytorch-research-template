import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
    return correct / total if total > 0 else 0.0
