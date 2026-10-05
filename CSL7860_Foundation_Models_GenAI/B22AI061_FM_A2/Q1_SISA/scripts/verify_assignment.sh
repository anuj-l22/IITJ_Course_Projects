#!/usr/bin/env bash
set -euo pipefail
echo "[Verify] Checking deliverables presence..."
test -f results/report.pdf && echo "✔ report.pdf"
test -d code && echo "✔ code/"
test -d results && echo "✔ results/"
test -f results/fig_accuracy_vs_deleted.png && echo "✔ Fig-1"
test -f results/fig_time_saved_vs_deleted.png && echo "✔ Fig-2"
test -f results/fig_mia_auc.png && echo "✔ Fig-3"
test -f results/table_layout_times.csv && echo "✔ Table-1 CSV"
echo "[Verify] Done."
