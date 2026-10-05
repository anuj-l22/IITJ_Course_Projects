# Interpreting Deepfake Detection on SID-Set

**Course:** MAP7020 – Applied ML Lab
**Author:** Anuj

*Foundation-model baselines and mechanistic analysis with sparse autoencoders*

A two-part interpretability study of deepfake detection on social media, using the **SID-Set** benchmark. SID-Set treats detection as a three-class problem: **real** images, **fully synthetic** images, and **tampered** images (real photos with AI-inpainted regions).
- **Part I** builds transparent baselines on frozen vision foundation models and runs a suite of interpretability analyses on them.
- **Part II** looks inside **SIDA-7B**, a state-of-the-art multimodal detector, using sparse autoencoders.

## Highlights
- **90.0% macro-F1** using only a linear probe on frozen CLIP features. This is about 96% of the performance of the fully fine-tuned 7B-parameter SIDA model (93.5%), with no fine-tuning and no GPU needed at evaluation time.
- **Causal feature steering** inside SIDA-7B raised tampered-class accuracy by up to **+9.4 percentage points**.
- Found a **critical failure mode**: the hardest 20% of tampered images reach only **12.7% accuracy**.
- Mapped **adversarial vulnerability by class**. The most accurate class (synthetic, 99.9%) turned out to be the most fragile to targeted feature perturbations.

## Part I: Frozen Foundation-Model Baselines
- **Backbones:** CLIP ViT-B/32, DINOv2-Base and SigLIP-Base, all frozen. Each was paired with L1 and L2 logistic regression and with decision trees.
- **Localisation:** per-patch classifiers on patch tokens. The best mean IoU for tampered regions was **0.375** (CLIP).
- **Interpretability suite:** L1 sparsity, decision trees, exact SHAP for linear models, Token-CAM, CKA similarity, calibration (ECE), layer-wise probing, pooling ablations, and robustness to JPEG compression, resizing and blur.
- **Findings:**
  - Deepfake cues are class-specific rather than shortcuts.
  - The cues are concentrated in the last quarter of the backbone, and concatenating the last 3 layers gives the best F1 (0.919).
  - The cues are robust to social-media image degradation, with less than a 3-point F1 drop.
  - Detection relies on holistic embedding shifts rather than localising the edited region.

## Part II: Sparse Autoencoders on SIDA-7B
- Extracted the 4096-d classification embedding from SIDA-7B for 60k images and decomposed it with **five SAE variants**: Vanilla, TopK, Gated, JumpReLU and Matryoshka, each with 16,384 features.
- **Reconstruction:** the Vanilla SAE explains 96.7% of the variance. Passing embeddings through the SAE preserves classification accuracy to within 0.05 percentage points.
- **Causal analysis:** feature selectivity, ablation, amplification-based steering, decision-boundary analysis of hard vs. easy samples, and logit attribution.
- **Findings:**
  - Synthetic-class features are highly selective and redundant: synthetic accuracy never drops below 99.8% under any ablation.
  - The top 20 features per class carry the full discriminative signal.

## Key Insights
Both parts reach the same conclusions independently:
- Tampered images are the hard class.
- Synthetic-discriminative features are concentrated and high-amplitude.
- Detectors rely on **globally distributed rather than spatially localised** cues.

## Skills & Tools
CLIP / DINOv2 / SigLIP · Sparse Autoencoders · Mechanistic Interpretability · SHAP · CKA · Multimodal LLMs (LLaVA-based SIDA) · Deepfake Detection
