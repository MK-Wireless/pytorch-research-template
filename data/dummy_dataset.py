import torch
from torch.utils.data import Dataset


class DummyClassificationDataset(Dataset):
    """
    Simple synthetic dataset for classification.
    """
    def __init__(self, num_samples: int, num_features: int, num_classes: int):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes

        self.data = torch.randn(num_samples, num_features)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]
