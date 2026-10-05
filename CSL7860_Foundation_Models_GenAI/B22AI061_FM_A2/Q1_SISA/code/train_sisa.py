import os, argparse, yaml, math, time, json, shutil
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .utils import Timer, set_seed, pick_device, save_checkpoint, load_checkpoint, count_parameters
from .models import build_model
from .data import get_datasets, sisa_partition
from .aggregator import LogitAveragingEnsemble, predict_loader
from sklearn.metrics import f1_score

def _f(x):
    return float(x) if isinstance(x, str) else x

def make_optimizer(params, cfg):
    if cfg['optimizer'] == 'sgd':
        return optim.SGD(params, lr=_f(cfg['lr']), momentum=_f(cfg.get('momentum', 0.0)), weight_decay=_f(cfg.get('weight_decay', 0.0)))
    else:
        return optim.Adam(params, lr=_f(cfg['lr']), weight_decay=_f(cfg.get('weight_decay', 0.0)))

def train_one_slice(model, loader, device, cfg):
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = make_optimizer(model.parameters(), cfg)
    for epoch in range(cfg['epochs_per_slice']):
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

def build_dataloaders(train_set, test_set, indices, batch_size, num_workers):
    train_loader = DataLoader(Subset(train_set, indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def gather_shard_models(base_dir, K, device, model_ctor):
    models = []
    for k in range(K):
        ckpt = os.path.join(base_dir, f"checkpoints/shard{k}_final.pt")
        state = load_checkpoint(ckpt, map_location=device)
        m = model_ctor()
        m.load_state_dict(state['model'])
        m.to(device)
        m.eval()
        models.append(m)
    return models

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--exp_name', type=str, required=True)
    p.add_argument('--simulate_delete', type=float, default=0.0, help='fraction to delete from training set')
    p.add_argument('--from_baseline', type=str, default=None, help='path to baseline results for reuse')
    p.add_argument('--naive_delete_only', type=int, default=0, help='if 1, do not retrain; only write deletion layout')
    p.add_argument('--delete_strategy', type=str, default='uniform', choices=['uniform', 'cluster_shards'],
                   help='uniform=sample deletions across dataset; cluster_shards=concentrate on a subset of shards')
    p.add_argument('--delete_shards', type=int, default=None, help='number of shards to target when delete_strategy=cluster_shards')
    p.add_argument('--tiny', type=int, default=0, help='use tiny subset for smoke test')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    exp_dir = os.path.join('results', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'used_config.yaml'), 'w') as _f:
        yaml.safe_dump(cfg, _f)

    set_seed(cfg['seed'])
    device = pick_device(cfg.get('device', 'auto'))
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    os.makedirs('results', exist_ok=True)
    out_dir = os.path.join('results', args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)

    # Data
    train_set, test_set, num_classes = get_datasets(cfg['dataset'], cfg['data_root'], cfg.get('purchase_path', None))
    param_count = count_parameters(build_model(cfg['model'], channels=cfg.get('channels', (32,64,128)), dropout=cfg.get('dropout', 0.2), num_classes=num_classes))

    # Tiny smoke subset
    train_indices = np.arange(len(train_set))
    test_indices = np.arange(len(test_set))
    if args.tiny:
        train_indices = train_indices[:2000]
        test_indices = test_indices[:1000]
        # Monkey-patch the datasets by creating Subset wrappers at loader time

    # SISA partitioning
    mapping, per_shard_slice, per_shard_cum = sisa_partition(train_set, cfg['K'], cfg['S'], seed=cfg['seed'])

    # Save layout
    layout_rows = []
    for (k, s), idxs in per_shard_slice.items():
        layout_rows.append({'shard': k, 'slice': s, 'n': len(idxs)})
    pd.DataFrame(layout_rows).to_csv(os.path.join(out_dir, 'layout.csv'), index=False)

    # Determine deletions
    delete_count = int(args.simulate_delete * len(train_indices))
    deleted_indices = set()
    targeted_shards = set()
    if delete_count > 0:
        rng = np.random.default_rng(cfg['seed'] + 42)
        if args.delete_strategy == 'uniform':
            chosen = rng.choice(train_indices, size=delete_count, replace=False)
        else:
            # Concentrate deletions into a subset of shards to highlight selective retraining
            target_shards = args.delete_shards or max(1, math.ceil(0.25 * cfg['K']))
            shard_choices = rng.choice(np.arange(cfg['K']), size=min(target_shards, cfg['K']), replace=False)
            candidate = np.concatenate([per_shard_slice[(k, s)] for k in shard_choices for s in range(cfg['S'])])
            if len(candidate) == 0:
                chosen = np.array([], dtype=int)
            else:
                chosen = rng.choice(candidate, size=min(delete_count, len(candidate)), replace=False)
                # If we still need more (rare), top-up uniformly
                remaining = delete_count - len(chosen)
                if remaining > 0:
                    extra_pool = np.setdiff1d(train_indices, chosen, assume_unique=True)
                    extra = rng.choice(extra_pool, size=min(remaining, len(extra_pool)), replace=False)
                    chosen = np.concatenate([chosen, extra])
            targeted_shards = set(int(k) for k in shard_choices.tolist())
        deleted_indices = set(int(x) for x in chosen.tolist())
        # Track actually impacted shards
        for idx in deleted_indices:
            if int(idx) in mapping:
                targeted_shards.add(mapping[int(idx)][0])

    pd.DataFrame({'deleted_idx': sorted(list(deleted_indices))}).to_csv(os.path.join(out_dir, 'deleted.csv'), index=False)

    # If naive-delete-only: skip training but write minimal metadata, compute agg from baseline if present
    if args.naive_delete_only == 1:
        # Copy baseline checkpoints pointer so MIA/eval can read model
        baseline_dir = args.from_baseline
        if baseline_dir is None or not os.path.isdir(baseline_dir):
            raise ValueError("--naive_delete_only requires --from_baseline to evaluate.")
        meta = {
            'note': 'naive_delete_only; no retraining performed; evaluation reuses baseline checkpoints',
            'from_baseline': baseline_dir,
            'simulate_delete': args.simulate_delete,
            'delete_strategy': args.delete_strategy
        }
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            f.write(json.dumps(meta, indent=2))

        # Reuse baseline artifacts so downstream eval/MIA can load this run
        for k in range(cfg['K']):
            src = os.path.join(baseline_dir, f'checkpoints/shard{k}_final.pt')
            dst = os.path.join(out_dir, f'checkpoints/shard{k}_final.pt')
            if os.path.isfile(src) and not os.path.isfile(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy(src, dst)
        for fname in ['test_probs.npy', 'test_targets.npy', 'metrics.csv']:
            src = os.path.join(baseline_dir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy(src, dst)

        base_metrics = {}
        if os.path.isfile(os.path.join(out_dir, 'metrics.csv')):
            base_metrics = pd.read_csv(os.path.join(out_dir, 'metrics.csv')).iloc[0].to_dict()
        elif os.path.isfile(os.path.join(baseline_dir, 'metrics.csv')):
            base_metrics = pd.read_csv(os.path.join(baseline_dir, 'metrics.csv')).iloc[0].to_dict()
        metrics = {
            'exp_name': args.exp_name,
            'simulate_delete': args.simulate_delete,
            'retrained_slices': 0,
            'total_slices': cfg['K'] * cfg['S'],
            'fraction_retrained': 0.0,
            'wall_clock_sec': 0.0,
            'test_acc': float(base_metrics.get('test_acc', 0.0)),
            'test_f1_macro': float(base_metrics.get('test_f1_macro', 0.0)),
            'rss_bytes': base_metrics.get('rss_bytes', 0),
            'gpu_max_mem_bytes': base_metrics.get('gpu_max_mem_bytes', 0),
            'params_per_model': base_metrics.get('params_per_model', param_count),
            'note': 'naive_delete_only (reused baseline checkpoints)',
            'delete_strategy': args.delete_strategy,
            'delete_shards_targeted': len(targeted_shards),
            'delete_count': delete_count
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
        print('Wrote naive-delete metadata and reused baseline checkpoints.')
        return

    # Training per shard-slice
    metrics_rows = []
    total_timer = Timer()
    total_timer.__enter__()

    # Model factory for this dataset
    def model_ctor():
        return build_model(cfg['model'], channels=cfg.get('channels', (32,64,128)), dropout=cfg.get('dropout', 0.2), num_classes=num_classes).to(device)

    # If reusing baseline (for unlearning), load checkpoints up to just before impacted slice per shard
    reuse = args.from_baseline is not None and os.path.isdir(args.from_baseline)
    baseline_dir = args.from_baseline
    baseline_metrics = None
    if reuse and os.path.isfile(os.path.join(baseline_dir, 'metrics.csv')):
        baseline_metrics = pd.read_csv(os.path.join(baseline_dir, 'metrics.csv')).iloc[0].to_dict()

    # Compute earliest impacted slice per shard
    earliest_impacted = {k: None for k in range(cfg['K'])}
    if len(deleted_indices) > 0:
        for idx in deleted_indices:
            if int(idx) not in mapping:
                continue
            k, s = mapping[int(idx)]
            if earliest_impacted[k] is None or s < earliest_impacted[k]:
                earliest_impacted[k] = s

    # For time-saved computation, also record how many slices retrained
    retrained_slices = 0
    total_slices = cfg['K'] * cfg['S']

    for k in range(cfg['K']):
        # Init / restore model
        if reuse and os.path.isfile(os.path.join(baseline_dir, f'checkpoints/shard{k}_pre_slice0.pt')):
            m = model_ctor()
            # start from pre-slice0 (initial weights saved by baseline run)
            state = load_checkpoint(os.path.join(baseline_dir, f'checkpoints/shard{k}_pre_slice0.pt'), map_location=device)
            m.load_state_dict(state['model'])
        else:
            m = model_ctor()

        # Find starting slice
        start_slice = 0
        if earliest_impacted[k] is not None and reuse:
            # Restore to checkpoint BEFORE that slice
            s = earliest_impacted[k]
            if s == 0:
                # already at pre_slice0
                start_slice = 0
            else:
                ck = os.path.join(baseline_dir, f'checkpoints/shard{k}_post_slice{s-1}.pt')
                if not os.path.isfile(ck):
                    raise FileNotFoundError(f"Missing checkpoint {ck} for shard {k} unlearning.")
                state = load_checkpoint(ck, map_location=device)
                m.load_state_dict(state['model'])
                start_slice = s
        elif reuse:
            # Nothing impacted in this shard: copy final model from baseline and skip retrain
            final_ckpt = os.path.join(baseline_dir, f'checkpoints/shard{k}_final.pt')
            state = load_checkpoint(final_ckpt, map_location=device)
            save_checkpoint(state, os.path.join(out_dir, f'checkpoints/shard{k}_final.pt'))
            continue

        # Save initial checkpoint for reproducibility
        save_checkpoint({'model': m.state_dict()}, os.path.join(out_dir, f'checkpoints/shard{k}_pre_slice0.pt'))

        # Train through slices
        for s in range(start_slice, cfg['S']):
            slice_indices = per_shard_slice[(k, s)]
            # Filter deletions if any
            if len(deleted_indices) > 0:
                slice_indices = [i for i in slice_indices if i not in deleted_indices]
            if args.tiny:
                # if tiny run, subset per slice to keep quick
                slice_indices = slice_indices[:max(1, len(slice_indices)//10)]
            train_loader = DataLoader(Subset(train_set, slice_indices), batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])

            with Timer() as t:
                train_one_slice(m, train_loader, device, cfg)
            retrained_slices += 1
            # Save post-slice checkpoint
            save_checkpoint({'model': m.state_dict()}, os.path.join(out_dir, f'checkpoints/shard{k}_post_slice{s}.pt'))

        # Save per-shard final
        save_checkpoint({'model': m.state_dict()}, os.path.join(out_dir, f'checkpoints/shard{k}_final.pt'))

    total_timer.__exit__(None, None, None)
    wall_time = total_timer.elapsed

    # Evaluate aggregated model
    models = []
    for k in range(cfg['K']):
        ckpt = os.path.join(out_dir, f'checkpoints/shard{k}_final.pt')
        if not os.path.isfile(ckpt) and reuse:
            ckpt = os.path.join(baseline_dir, f'checkpoints/shard{k}_final.pt')
        state = load_checkpoint(ckpt, map_location=device)
        m = model_ctor()
        m.load_state_dict(state['model'])
        m.to(device).eval()
        models.append(m)
    ensemble = LogitAveragingEnsemble(models)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    acc, y_true, y_prob = predict_loader(ensemble, test_loader, device)
    import numpy as _np
    y_pred = _np.argmax(y_prob, axis=1)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    import os as _os, torch as _torch
    rss_bytes = psutil.Process(_os.getpid()).memory_info().rss
    gpu_max_mem_bytes = int(_torch.cuda.max_memory_allocated()) if _torch.cuda.is_available() else 0

    impacted_shard_count = sum(1 for _, s in earliest_impacted.items() if s is not None)
    delta_acc = None
    time_saved_frac = None
    if baseline_metrics is not None:
        if 'test_acc' in baseline_metrics:
            delta_acc = acc - float(baseline_metrics.get('test_acc', 0.0))
        baseline_wall = float(baseline_metrics.get('wall_clock_sec', 0.0))
        if baseline_wall > 0:
            time_saved_frac = (baseline_wall - wall_time) / baseline_wall

    # Metrics
    metrics = {
        'exp_name': args.exp_name,
        'simulate_delete': args.simulate_delete,
        'retrained_slices': retrained_slices,
        'total_slices': total_slices,
        'fraction_retrained': (retrained_slices / total_slices) if total_slices > 0 else 0.0,
        'wall_clock_sec': wall_time,
        'test_acc': acc,
        'test_f1_macro': f1_macro,
        'rss_bytes': rss_bytes,
        'gpu_max_mem_bytes': gpu_max_mem_bytes,
        'params_per_model': param_count,
        'delete_strategy': args.delete_strategy,
        'delete_shards_targeted': len(targeted_shards),
        'delete_count': delete_count,
        'impacted_shards': impacted_shard_count,
        'delta_acc_vs_baseline': delta_acc if delta_acc is not None else np.nan,
        'time_saved_frac_vs_baseline': time_saved_frac if time_saved_frac is not None else np.nan
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    # Save probs to npy for MIA
    np.save(os.path.join(out_dir, 'test_probs.npy'), y_prob)
    np.save(os.path.join(out_dir, 'test_targets.npy'), y_true)

    print(f"Done {args.exp_name}. Test accuracy={acc:.4f}, wall_clock={wall_time:.1f}s, retrained_slices={retrained_slices}/{total_slices}")

if __name__ == '__main__':
    main()
