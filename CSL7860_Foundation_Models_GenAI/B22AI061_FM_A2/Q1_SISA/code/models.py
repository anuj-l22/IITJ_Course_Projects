import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """A small CNN for CIFAR-10 with <= 1M params (default channels ~ 32/64/128).
    """
    def __init__(self, num_classes=10, channels=(32, 64, 128), dropout=0.2):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),

            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),

            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

class MLP(nn.Module):
    """Compact MLP for Purchase-100 style tabular data."""
    def __init__(self, in_dim=600, num_classes=100, hidden=(512, 256), dropout=0.2):
        super().__init__()
        dims = [in_dim] + list(hidden)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

def build_model(name='small_cnn', **kwargs):
    if name == 'small_cnn':
        return SmallCNN(num_classes=10, channels=tuple(kwargs.get('channels', (32,64,128))), dropout=kwargs.get('dropout', 0.2))
    if name == 'mlp':
        return MLP(in_dim=kwargs.get('in_dim', 600), num_classes=kwargs.get('num_classes', 100),
                   hidden=tuple(kwargs.get('hidden', (512,256))), dropout=kwargs.get('dropout', 0.2))
    raise ValueError(f"Unknown model {name}")
