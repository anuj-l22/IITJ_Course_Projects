import os, argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def load_metrics(roots):
    # Combine metrics.csv across experiments
    rows = []
    for r in roots:
        mpath = os.path.join(r, 'metrics.csv')
        if os.path.isfile(mpath):
            df = pd.read_csv(mpath)
            df['root'] = r
            rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = load_metrics(args.roots)
    if metrics.empty:
        print('No metrics found.')
        return

    # Fig-1: Test accuracy vs % deleted
    fig1 = os.path.join(args.out_dir, 'fig_accuracy_vs_deleted.png')
    plt.figure()
    x = metrics['simulate_delete'] * 100.0
    y = metrics['test_acc'] * 100.0
    plt.scatter(x, y)
    for i, row in metrics.iterrows():
        plt.annotate(os.path.basename(row['root']), (row['simulate_delete']*100.0, row['test_acc']*100.0))
    plt.xlabel('% deleted')
    plt.ylabel('Test accuracy (%)')
    plt.title('Test accuracy vs % deleted')
    plt.grid(True)
    plt.savefig(fig1, bbox_inches='tight')
    plt.close()

    # Fig-2: Time saved vs % deleted (requires baseline wall-clock for comparison)
    # Prefer simulate_delete==0 as baseline; otherwise use smallest delete
    baseline_candidates = metrics.loc[metrics['simulate_delete'] == 0]
    if baseline_candidates.empty:
        baseline = metrics.loc[metrics['simulate_delete'].idxmin()]
    else:
        baseline = baseline_candidates.iloc[0]
    fig2 = os.path.join(args.out_dir, 'fig_time_saved_vs_deleted.png')
    plt.figure()
    xs, ys = [], []
    for i, row in metrics.iterrows():
        if row['simulate_delete'] <= baseline['simulate_delete'] + 1e-9:
            continue
        xs.append(row['simulate_delete'] * 100.0)
        if 'time_saved_frac_vs_baseline' in row and pd.notnull(row['time_saved_frac_vs_baseline']):
            ys.append(max(0.0, row['time_saved_frac_vs_baseline'] * 100.0))
        else:
            time_saved = max(0.0, baseline['wall_clock_sec'] - row['wall_clock_sec'])
            ys.append(time_saved / baseline['wall_clock_sec'] * 100.0 if baseline['wall_clock_sec'] > 0 else 0.0)
    if xs:
        plt.plot(xs, ys, marker='o')
    plt.xlabel('% deleted')
    plt.ylabel('Time saved (%)')
    plt.title('Time saved vs % deleted')
    plt.grid(True)
    plt.savefig(fig2, bbox_inches='tight')
    plt.close()

    # Fig-3: MIA AUC comparison
    mia_path = os.path.join(os.path.dirname(args.out_dir), 'mia_auc.csv')
    if os.path.isfile(mia_path):
        mia = pd.read_csv(mia_path)
        fig3 = os.path.join(args.out_dir, 'fig_mia_auc.png')
        plt.figure()
        plt.bar(range(len(mia)), mia['mia_auc'])
        plt.xticks(range(len(mia)), mia['model'], rotation=45, ha='right')
        plt.ylabel('AUC')
        plt.title('Membership Inference AUC')
        plt.grid(True, axis='y')
        plt.savefig(fig3, bbox_inches='tight')
        plt.close()

    # Table-1: KxS layout, retrained slices, wall-clock times
    # Gather from each root's layout.csv and metrics.csv
    table_rows = []
    for r in args.roots:
        layout_path = os.path.join(r, 'layout.csv')
        metrics_path = os.path.join(r, 'metrics.csv')
        if os.path.isfile(layout_path) and os.path.isfile(metrics_path):
            lay = pd.read_csv(layout_path)
            met = pd.read_csv(metrics_path).iloc[0]
            K = lay['shard'].nunique()
            S = lay['slice'].nunique()
            table_rows.append({
                'exp': os.path.basename(r),
                'K': K, 'S': S,
                'retrained_slices': int(met.get('retrained_slices', 0)),
                'total_slices': int(met.get('total_slices', K*S)),
                'fraction_retrained': float(met.get('fraction_retrained', met.get('retrained_slices', 0)/(K*S))),
                'wall_clock_sec': float(met.get('wall_clock_sec', 0.0)),
                'simulate_delete': float(met.get('simulate_delete', 0.0)),
                'delete_strategy': met.get('delete_strategy', 'uniform')
            })
    if table_rows:
        table = os.path.join(args.out_dir, 'table_layout_times.csv')
        pd.DataFrame(table_rows).to_csv(table, index=False)

    print('Wrote plots and table to', args.out_dir)

if __name__ == '__main__':
    main()
