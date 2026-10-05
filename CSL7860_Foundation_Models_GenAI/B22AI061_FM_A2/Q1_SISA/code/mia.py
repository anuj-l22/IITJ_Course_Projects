import os, argparse, numpy as np, pandas as pd, torch, json
from .aggregator import LogitAveragingEnsemble
from .attacks import mia_from_probs
from .utils import load_checkpoint
from .models import build_model
from .data import get_datasets
from torch.utils.data import DataLoader, Subset

def load_ensemble(dir_path, K, model_ctor, device, fallback_dir=None):
    models = []
    meta_fallback = None
    meta_path = os.path.join(dir_path, 'meta.json')
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, 'r'))
            meta_fallback = meta.get('from_baseline')
        except Exception:
            meta_fallback = None
    if fallback_dir:
        meta_fallback = meta_fallback or fallback_dir

    for k in range(K):
        ckpt = os.path.join(dir_path, f'checkpoints/shard{k}_final.pt')
        if not os.path.isfile(ckpt) and meta_fallback:
            alt = os.path.join(meta_fallback, f'checkpoints/shard{k}_final.pt')
            if os.path.isfile(alt):
                ckpt = alt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for shard {k} in {dir_path}")
        state = load_checkpoint(ckpt, map_location=device)
        m = model_ctor()
        m.load_state_dict(state['model'])
        m.to(device).eval()
        models.append(m)
    return LogitAveragingEnsemble(models).to(device).eval()

def deleted_indices_for(run_dir):
    path = os.path.join(run_dir, 'deleted.csv')
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if 'deleted_idx' in df.columns:
            return set(int(i) for i in df['deleted_idx'].tolist())
    return set()

def fallback_from_meta(run_dir):
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, 'r'))
            return meta.get('from_baseline')
        except Exception:
            return None
    return None

def load_probs(run_dir, fname, fallback=None):
    primary = os.path.join(run_dir, fname)
    if not os.path.isfile(primary) and fallback:
        alt = os.path.join(fallback, fname)
        if os.path.isfile(alt):
            primary = alt
    if not os.path.isfile(primary):
        raise FileNotFoundError(f"Missing {fname} in {run_dir}")
    return np.load(primary)

def train_subset_loader(train_set, subset_indices, batch_size, num_workers):
    if len(subset_indices) == 0:
        # fall back to a small random subset to avoid empty AUC computation
        rng = np.random.default_rng(42)
        all_idx = np.arange(len(train_set))
        subset = rng.choice(all_idx, size=min(1000, len(all_idx)), replace=False)
    else:
        subset = np.array(sorted(list(subset_indices)))
    return DataLoader(Subset(train_set, subset), batch_size=batch_size, shuffle=False, num_workers=num_workers)

def collect_probs(model, loader, device):
    import torch.nn.functional as F
    probs = []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    if probs:
        return np.concatenate(probs, axis=0)
    return np.zeros((0,))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=str, required=True)
    ap.add_argument('--naive', type=str, required=False, default=None)
    ap.add_argument('--unlearned', nargs='*', default=[])
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--config', type=str, default='code/config.yaml')
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, test_set, num_classes = get_datasets(cfg['dataset'], cfg['data_root'], cfg.get('purchase_path', None))
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    model_ctor = lambda: build_model(cfg['model'], channels=cfg.get('channels', (32,64,128)), dropout=cfg.get('dropout', 0.2), num_classes=num_classes)


    rows = []

    # Baseline
    baseline_dir = args.baseline
    baseline_ens = load_ensemble(baseline_dir, cfg['K'], model_ctor, device)
    baseline_deleted = deleted_indices_for(baseline_dir)
    # Fallback to a deletion set from naive/unlearned for fair comparison
    if len(baseline_deleted) == 0:
        if args.naive:
            baseline_deleted = deleted_indices_for(args.naive)
        elif args.unlearned:
            baseline_deleted = deleted_indices_for(args.unlearned[0])
    train_loader_baseline = train_subset_loader(train_set, baseline_deleted, batch_size, num_workers)
    baseline_train_probs = collect_probs(baseline_ens, train_loader_baseline, device)
    baseline_test_probs = load_probs(baseline_dir, 'test_probs.npy')
    auc_baseline = mia_from_probs(baseline_train_probs, baseline_test_probs)
    rows.append({'model': 'baseline', 'mia_auc': float(auc_baseline)})

    # Naive full retrain (if provided)
    if args.naive:
        naive_dir = args.naive
        naive_fallback = fallback_from_meta(naive_dir)
        naive_ens = load_ensemble(naive_dir, cfg['K'], model_ctor, device, fallback_dir=naive_fallback)
        naive_deleted = deleted_indices_for(naive_dir)
        train_loader_naive = train_subset_loader(train_set, naive_deleted, batch_size, num_workers)
        naive_train_probs = collect_probs(naive_ens, train_loader_naive, device)
        naive_test_probs = load_probs(naive_dir, 'test_probs.npy', fallback=naive_fallback)
        auc_naive = mia_from_probs(naive_train_probs, naive_test_probs)
        rows.append({'model': 'naive_retrain', 'mia_auc': float(auc_naive)})

    # SISA-unlearned runs
    for path in args.unlearned:
        fallback_dir = fallback_from_meta(path)
        ens = load_ensemble(path, cfg['K'], model_ctor, device, fallback_dir=fallback_dir)
        unlearn_deleted = deleted_indices_for(path)
        train_loader_ul = train_subset_loader(train_set, unlearn_deleted, batch_size, num_workers)
        train_probs = collect_probs(ens, train_loader_ul, device)
        test_probs = load_probs(path, 'test_probs.npy', fallback=fallback_dir)
        auc = mia_from_probs(train_probs, test_probs)
        rows.append({'model': f'unlearned:{os.path.basename(path)}', 'mia_auc': float(auc)})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote MIA AUCs to {args.out}")

if __name__ == '__main__':
    main()
