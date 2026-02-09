import time


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class Timer:
    """
    Simple wall-clock timer.
    """
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


def log_epoch(epoch: int, metrics: dict, elapsed: float) -> None:
    """
    Print a one-line summary for an epoch.
    """
    msg = f"[Epoch {epoch:03d}] "
    msg += " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    msg += f" | time: {elapsed:.2f}s"
    print(msg)

