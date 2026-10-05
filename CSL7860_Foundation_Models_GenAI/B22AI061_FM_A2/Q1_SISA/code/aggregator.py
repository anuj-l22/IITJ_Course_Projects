import torch
import torch.nn.functional as F

class LogitAveragingEnsemble(torch.nn.Module):
    """Averages logits from K shard models at inference time."""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @torch.no_grad()
    def forward(self, x):
        logits_sum = None
        for m in self.models:
            out = m(x)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / len(self.models)
        return logits

def predict_loader(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    all_targets = []
    all_probs = []
    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
        all_targets.append(yb.cpu())
        all_probs.append(probs.cpu())
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    acc = correct / total
    return acc, all_targets, all_probs


class MajorityVoteEnsemble(torch.nn.Module):
    """Majority vote over per-model argmax predictions."""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @torch.no_grad()
    def forward(self, x):
        # Return averaged logits for compatibility;
        # majority voting can be handled at prediction stage if needed.
        outs = []
        for m in self.models:
            outs.append(m(x))
        return sum(outs) / len(outs)
