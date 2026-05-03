from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

class SplitData:
    def __init__(self, splits_dir: Path):
        self.splits_dir = Path(splits_dir)

    def load(self) -> dict:
        X_train = np.load(self.splits_dir / "X_train.npy")
        X_val = np.load(self.splits_dir / "X_val.npy")
        X_test = np.load(self.splits_dir / "X_test.npy")
        y_train = np.load(self.splits_dir / "y_train.npy")
        y_val = np.load(self.splits_dir / "y_val.npy")
        y_test = np.load(self.splits_dir / "y_test.npy")
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

def compute_class_weights(y: np.ndarray, power: float = 1.0) -> np.ndarray:
    y = np.asarray(y)
    num_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = (1.0 / counts[nonzero]) ** power
    if weights[nonzero].sum() > 0:
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return weights

def make_weighted_sampler(y: np.ndarray, power: float = 1.0) -> WeightedRandomSampler:
    weights = compute_class_weights(y, power=power)
    sample_weights = weights[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
