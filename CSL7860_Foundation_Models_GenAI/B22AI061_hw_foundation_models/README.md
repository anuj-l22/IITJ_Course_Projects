# Generative Foundation Models: Creation, Evaluation & a Custom Metric

**Course:** CSL7860 – Foundation Models and Generative AI
**Author:** Anuj

A hands-on study of generative foundation models in three parts:
1. Creating artifacts with image and audio generation models.
2. Surveying how generative models are evaluated, both automatically and by humans.
3. Designing a **custom quantitative metric** to score the images generated in part 1.

## Highlights
- Generated **optical-illusion images** in which a forest scene (an owl and a panther) hides a woman's face. This used Stable Diffusion 1.5 (Realistic Vision) with a QR-code-monster ControlNet. Background music for a kids' show was generated with MusicGen.
- Designed **IllusionScore**, a custom metric that combines Canny edge-overlap F2 against the control image with a face-detector penalty for literal portraits and an edge-density penalty for clutter.
- Ranked 8 generated variants and traced how prompt wording and illusion strength changed the score. The best image scored 0.386.

## Approach
- **Creation:** controlled experiments with a fixed seed, guidance scale and sampler, comparing illusion strengths {0.8, 1.5} across 4 prompt variants. All runs were logged for reproducibility.
- **Architecture comparison:** latent diffusion with ControlNet spatial conditioning vs. MusicGen's autoregressive decoder-only Transformer over EnCodec audio tokens.
- **Evaluation survey:** worked examples of perplexity, BLEU and FID. Covered human evaluation through pairwise preference (Bradley–Terry) and rubric ratings, the ARC and GSM-8K benchmarks, and gaps highlighted by HELM: calibration, faithfulness, long context, multilingual coverage and safety.
- **Metric:** IllusionScore = F2 − 0.9 × portrait penalty − 0.25 × edge-density penalty, using 1-pixel dilation tolerance and a Haar-cascade face detector.

## Key Insights
- A higher illusion strength increased edge recall without producing a literal face.
- Explicitly asking for a "hidden face" in the prompt improved alignment but triggered the portrait penalty at high strength.
- Prompts for a minimal background removed clutter but left too little structure to form the illusion.
- The report also outlines a LoRA fine-tuning plan, with evaluation gates and mitigations for its risks.

## Skills & Tools
Stable Diffusion · ControlNet · MusicGen · Hugging Face · Canny Edges & Haar-cascade Face Detection · Generative Model Evaluation · Metric Design · Prompt Engineering
