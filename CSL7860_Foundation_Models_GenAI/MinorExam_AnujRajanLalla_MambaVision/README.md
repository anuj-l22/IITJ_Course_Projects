# MambaVision Analysis & LLM Robustness Experiments

**Course:** CSL7860 – Foundation Models and Generative AI
**Author:** Anuj

A two-part study for the course's minor exam:
- **Part A** analyses and runs experiments on **MambaVision** (CVPR 2025), a hybrid vision backbone that combines Mamba state-space models with Transformer attention.
- **Part B** stress-tests LLMs: their robustness to noisy prompts, and their ability to retrieve a fact from a long context.

## Part A: MambaVision
- **Paper analysis:** a deep dive into MambaVision's design: a hierarchical 4-stage backbone (CNN stages followed by hybrid stages), a non-causal two-branch Mamba mixer, and self-attention placed only in the final layers of the later stages. The analysis is backed by the paper's ablations.
- **Reproduction sanity check:** trained on a class-balanced CIFAR-10 subset (1,000 images, upsampled to 224×224). Optimisation was stable and showed no divergence.
- **Distribution-shift experiment:** linear-probe fine-tuning on **ImageNet-R** (sketches, paintings, cartoons and other renditions), compared against **ViT-B/16** and **ConvNeXt-Tiny**. Comparisons covered parameter count, throughput, throughput per million parameters and top-1 accuracy.
- **Finding:** MambaVision-Tiny matched the baselines' accuracy under distribution shift while delivering **substantially higher inference throughput and throughput per parameter**.

## Part B: LLM Robustness
- **Robustness to messy inputs (`flan-t5-large`):** asked 50 questions about U.S. state capitals under 4 kinds of noise (typos, spacing and punctuation, Unicode look-alike characters, emoji) at two levels each, for 450 prompts in total.
  - Unicode look-alikes were the most damaging, causing up to a **−22 point (−91.7%)** drop from the clean accuracy.
  - A robustness-oriented prompt raised clean accuracy on the 20-item subset from **30% to 45%**. Adding a light Unicode-normalisation preprocessor brought large gains on the look-alike noise (5% → 35% for light noise).
- **Needle-in-a-haystack (`Qwen2.5-1.5B-Instruct`):** built a long-context corpus and inserted a unique code at the start, middle or end, then logged token positions. The model failed at all three positions (0% exact match), so the setup could not show a lost-in-the-middle effect. The report states this outcome plainly.

## Skills & Tools
PyTorch · torchvision · Hugging Face Transformers · State-Space Models (Mamba) · Vision Transformers · Robustness Evaluation · Long-context Evaluation · t-SNE
