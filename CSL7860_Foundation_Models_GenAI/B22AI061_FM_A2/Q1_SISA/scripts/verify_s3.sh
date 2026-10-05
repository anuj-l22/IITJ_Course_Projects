#!/usr/bin/env bash
set -euo pipefail
echo "[*] Baseline used_config.yaml (K,S):"
python - <<'PY'
import yaml
cfg = yaml.safe_load(open('results/baseline_s3/used_config.yaml'))
print("K=", cfg.get('K'), " S=", cfg.get('S'))
PY

echo "[*] Retrained vs total slices:"
python - <<'PY'
import pandas as pd
for exp in ['baseline_s3','del1pct_s3','del5pct_s3']:
    df = pd.read_csv(f'results/{exp}/metrics.csv')
    row = df.iloc[-1].to_dict()
    print(exp, 'retrained/total =', row.get('retrained_slices'), '/', row.get('total_slices'))
PY
