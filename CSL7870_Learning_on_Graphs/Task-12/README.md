For reproducibility ,the evaluator can run the colab notebook as attached

What files are produced (used later for t-SNE)

A) Index CSVs (Experiment 2)

These log test accuracy and point to the saved checkpoint for each model/depth.

exp2_index_full.csv — clean graph, full features

exp2_index_full_noisy.csv — noisy graph, full features

Columns (typical):
Tag, Model, Depth, TestAcc, Checkpoint

The TSNE code reads these CSVs to know which checkpoints to load for each model at L=2/4/8/16.

B) Checkpoints

Saved under a checkpoints/ folder (created automatically).
Filename pattern:

checkpoints/{tag}_{model}_L{depth}.pt
# e.g., checkpoints/full_GAT-h1 (full)_L4.pt
# e.g., checkpoints/noisy_SAGE-max (full)_L16.pt


tag = full for clean graph

tag = noisy for noisy graph

So these are the files that will get produced when running in colab (or locally if notebook is run locally) but advise is to run in colab