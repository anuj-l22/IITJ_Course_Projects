# What this folder contains

Q2_Prompt_Sensitivity.ipynb
Notebook that runs the robustness study:

builds 50 clean QA prompts (state → capital) with gold answers

creates noisy variants (typos, spacing/punctuation, unicode confusables, emoji) at light/heavy levels

evaluates with FLAN-T5 (google/flan-t5-large)

computes Exact Match (EM) accuracy and token-F1

saves figures + per-example results

outputs/ (all generated)

heatmap_full.png – Accuracy grid for the full 50-item baseline (no mitigations).
Rows = noise types; columns = none/light/heavy; values = % accuracy (EM).

heatmap_subset_subset_baseline.png – 20-item subset, baseline per noise/level.

heatmap_subset_subset_prompt.png – 20-item subset, robust prompt per noise/level.

heatmap_subset_subset_preproc_prompt.png – 20-item subset, light preprocessor + robust prompt per noise/level.

results.csv or results.xlsx – Per-example records for the full 50 × (clean + noises) run.
Columns:
id, noise_type, noise_level, prompt_in, gold, pred, correct(0/1), f1, tag

results_subset.csv or results_subset.xlsx – Per-example records for the 20-item subset under three settings (baseline / robust prompt / preproc+prompt).
Columns: same as above; tag ∈ {subset_baseline,subset_prompt,subset_preproc_prompt}.

error_examples.json – One example per error category used in the report (e.g., other_state_capital, extra_verbiage, etc.).

appendix_preprocessor.py – Minimal light preprocessor used in the subset mitigation:
NFKC → remove emoji (Unicode ranges) → unidecode → collapse whitespace.

requirements.txt ( Extracted from the colab notebook )

# How to reproduce (notebook)

Open Q2_Prompt_Sensitivity.ipynb and Run All.

Outputs will be written under outputs/ with the same filenames as above.