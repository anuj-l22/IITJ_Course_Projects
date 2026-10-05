#!/usr/bin/env bash
set -euo pipefail
CFG=code/config_s3.yaml

echo "[1/6] Baseline (S=3)"
python -m code.train_sisa --config ${CFG} --exp_name baseline_s3

echo "[2/6] 1% deletion (SISA selective retrain from baseline)"
python -m code.train_sisa --config ${CFG} --exp_name del1pct_s3 --simulate_delete 0.01 --from_baseline results/baseline_s3 --delete_strategy cluster_shards --delete_shards 5

echo "[3/6] 5% deletion (SISA selective retrain from baseline)"
python -m code.train_sisa --config ${CFG} --exp_name del5pct_s3 --simulate_delete 0.05 --from_baseline results/baseline_s3 --delete_strategy cluster_shards --delete_shards 5

echo "[4/6] Naive full retrain (gold standard)"
python -m code.train_sisa --config ${CFG} --exp_name del5pct_naive_s3 --simulate_delete 0.05 --delete_strategy cluster_shards --delete_shards 5

echo "[5/6] Membership Inference AUCs"
python -m code.mia --config ${CFG} --baseline results/baseline_s3 --naive results/del5pct_naive_s3 --unlearned results/del1pct_s3 results/del5pct_s3 --out results/mia_auc.csv

echo "[6/6] Plots + PDF (plots -> results/plots)"
python -m code.plot_results --roots results/baseline_s3 results/del1pct_s3 results/del5pct_s3 results/del5pct_naive_s3 --out_dir results/plots
python -m code.make_report --results_dir results --out report_s3.pdf
