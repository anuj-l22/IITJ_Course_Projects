# SISA Machine Unlearning

**Course:** CSL7860 – Foundation Models and Generative AI
**Author:** Anuj

Implemented **SISA (Sharded, Isolated, Sliced, Aggregated) training** so that data-deletion requests can be honoured without retraining the whole model. The training data is split into isolated shards, and each shard is trained in chronological slices. When data is deleted, only the shards that contained it are retrained, starting from the earliest affected slice.

## Highlights
- **Saved 75–83% of wall-clock time** compared with full retraining, by retraining only 25% of slices (15 of 60).
- **Accuracy stayed within 0.7 percentage points** of the baseline under 1% and 5% deletion.
- Membership-inference AUC stayed **close to chance (0.45–0.47)**, a basic check for privacy leakage.

## Approach
- **Setup:** CIFAR-10 with a small CNN (288k parameters).
- **SISA layout:** K = 20 shards and S = 3 slices. Weights carry forward across slices within a shard, and shard models stay isolated from one another.
- **Aggregation:** the shard models' logits are averaged.
- **Deletion handling:** 1% and 5% deletion scenarios, compared with a naive "full retrain after delete" control.
- **Metrics:** test accuracy, wall-clock time and time saved, and a confidence-based membership-inference attack AUC.

## Results
| Experiment | Retrained Slices | Time (s) | Time Saved | Accuracy |
|---|---|---|---|---|
| Baseline | 60/60 | 159.57 | — | 27.82% |
| SISA – 1% deleted | 15/60 | 40.31 | **74.74%** | 27.29% |
| SISA – 5% deleted | 15/60 | 27.89 | **82.52%** | 27.13% |
| Naive retrain – 5% | 60/60 | 146.95 | 7.91% | 26.07% |

## Extension
The report maps SISA's shard and slice isolation onto adapter or LoRA fine-tuning of foundation models. The idea is to keep a frozen backbone, store one adapter per shard, and retrain only the affected adapter slices.

## Skills & Tools
Machine Unlearning · Privacy-preserving ML · Membership Inference Attacks · CNNs · CIFAR-10
