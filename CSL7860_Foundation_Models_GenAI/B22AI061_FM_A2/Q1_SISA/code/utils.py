import os, random, time
import numpy as np
import torch

class Timer:
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
        self.elapsed = self.end - self.start

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(pref='auto'):
    if pref == 'cpu':
        return torch.device('cpu')
    if pref == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # auto
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
