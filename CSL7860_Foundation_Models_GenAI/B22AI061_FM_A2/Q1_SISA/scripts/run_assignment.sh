#!/usr/bin/env bash
set -euo pipefail

CFG=code/config.yaml
#CFG=code/config_tiny.yaml
DEL_FLAGS="--delete_strategy cluster_shards --delete_shards 3"

# A. Baseline
python -m code.train_sisa --config ${CFG} --exp_name baseline

# B. Deletions 1% and 5% (SISA unlearning; retrain impacted slices only)
python -m code.train_sisa --config ${CFG} --exp_name sisa_unlearn_1pct --simulate_delete 0.01 --from_baseline results/baseline ${DEL_FLAGS}
python -m code.train_sisa --config ${CFG} --exp_name sisa_unlearn_5pct --simulate_delete 0.05 --from_baseline results/baseline ${DEL_FLAGS}

# Naive full retraining for gold standard
python -m code.train_sisa --config ${CFG} --exp_name naive_retrain_1pct --simulate_delete 0.01 ${DEL_FLAGS}
python -m code.train_sisa --config ${CFG} --exp_name naive_retrain_5pct --simulate_delete 0.05 ${DEL_FLAGS}

# C. Membership inference AUC
python -m code.mia \
  --config ${CFG} \
  --baseline results/baseline \
  --naive results/naive_retrain_5pct \
  --unlearned results/sisa_unlearn_1pct results/sisa_unlearn_5pct \
  --out results/mia_auc.csv

# Plots + Table-1
python -m code.plot_results       --roots results/baseline results/sisa_unlearn_1pct results/naive_retrain_1pct results/sisa_unlearn_5pct results/naive_retrain_5pct       --out_dir results

# Report
python -m code.make_report --results_dir results --plots_dir results --out results/report.pdf
