import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

class CIFAR10Indexed(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index  # include raw index

class Purchase100(Dataset):
    """Assumes a CSV with features and label in last column; adjust as needed."""
    def __init__(self, csv_path, train=True, transform=None):
        df = pd.read_csv(csv_path)
        # Split 80/20 by index for train/test
        n = len(df)
        split = int(0.8 * n)
        if train:
            df = df.iloc[:split]
        else:
            df = df.iloc[split:]
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])
        return x, y, idx

def _coerce_dataset_args(name, root, purchase_path):
    """Allow passing either a config dict or individual args."""
    if isinstance(name, dict):
        cfg = name
        name = cfg.get('dataset', 'cifar10')
        root = cfg.get('data_root', root)
        purchase_path = cfg.get('purchase_path', purchase_path)
    return name, root, purchase_path

def get_datasets(name='cifar10', root='./data', purchase_path=None):
    name, root, purchase_path = _coerce_dataset_args(name, root, purchase_path)
    if name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        tf_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = CIFAR10Indexed(root, train=True, transform=tf_train, download=True)
        test = CIFAR10Indexed(root, train=False, transform=tf_test, download=True)
        num_classes = 10
    elif name == 'purchase100':
        train = Purchase100(purchase_path, train=True)
        test = Purchase100(purchase_path, train=False)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset {name}")
    return train, test, num_classes

def sisa_partition(train_set, K, S, seed=1337):
    """Return mapping from index -> (shard, slice), and lists of indices per (shard, slice).
    Splits by index order into K shards, then splits each shard into S chronological slices.
    """
    n = len(train_set)
    indices = np.arange(n)
    # Deterministic: do NOT shuffle to preserve 'chronology' by index
    shards = np.array_split(indices, K)

    mapping = {}
    per_shard_slice = {}  # (k, s) -> np.array of indices
    for k, shard_idx in enumerate(shards):
        slices = np.array_split(shard_idx, S)
        for s, sl in enumerate(slices):
            per_shard_slice[(k, s)] = sl
            for idx in sl:
                mapping[int(idx)] = (k, s)

    # Also return per-shard cumulative slices for quick lookup
    per_shard_cumulative = {}
    for k in range(K):
        cum = []
        for s in range(S):
            cum.append(per_shard_slice[(k, s)])
        per_shard_cumulative[k] = cum

    return mapping, per_shard_slice, per_shard_cumulative
