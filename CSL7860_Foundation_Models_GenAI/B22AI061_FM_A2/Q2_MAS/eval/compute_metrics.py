from __future__ import annotations
import os, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

def compute(log_path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(log_path, 'r', encoding='utf-8') if l.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Flatten constraints
    if 'constraints' in df.columns:
        c = pd.json_normalize(df['constraints']).add_prefix('constraint.')
        df = pd.concat([df.drop(columns=['constraints']), c], axis=1)
    else:
        df['constraint.has_takeaways'] = False
        df['constraint.has_readings'] = False
        df['constraint.length_ok'] = False
    # Success: all hard constraints satisfied
    df['success'] = df[['constraint.has_takeaways','constraint.has_readings','constraint.length_ok']].all(axis=1)
    # Violation count per row
    df['violations'] = (~df[['constraint.has_takeaways','constraint.has_readings','constraint.length_ok']]).sum(axis=1)
    # Ensure expected numeric columns exist
    for col in ['latency_ms_total','tool_call_count']:
        if col not in df.columns:
            df[col] = 0
    return df

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='path to results.jsonl')
    ap.add_argument('--csv', required=True, help='path to write metrics.csv')
    ap.add_argument('--plot', default='', help='optional path to write latency histogram PNG')
    args = ap.parse_args()

    df = compute(args.log)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f'Wrote CSV: {args.csv} ({len(df)} rows)')

    # High-level aggregates
    if len(df) > 0:
        print(f"Success rate: {df['success'].mean():.2%}")
        print(f"Avg total latency (ms): {df['latency_ms_total'].mean():.1f}")
        print(f"Avg tool calls: {df['tool_call_count'].mean():.2f}")
        print(f"Total constraint violations: {int(df['violations'].sum())}")

    if args.plot:
        plt.figure()
        df['latency_ms_total'].plot(kind='hist', bins=10)
        plt.xlabel('End-to-end latency (ms)')
        plt.ylabel('Count')
        plt.title('Latency Distribution')
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
        plt.savefig(args.plot, bbox_inches='tight')
        print(f'Saved plot: {args.plot}')

if __name__ == '__main__':
    main()
