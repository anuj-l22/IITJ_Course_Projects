# ControlNet from Scratch: Spatial Conditioning for Stable Diffusion

**Course:** AIL7490 – MMLVL
**Author:** Anuj

An end-to-end implementation of **ControlNet** (Zhang et al., ICCV 2023), which adds spatial conditioning control to a frozen Stable Diffusion 1.5 model. The core components were built from scratch, following the paper and the official source code instead of an existing ControlNet library: zero convolutions, the trainable encoder copy, the conditioning embedding network and skip-connection injection. The model was trained on the Fill50K dataset and evaluated with standard generative metrics.

## Highlights
- Built the full ControlNet architecture from scratch on top of a frozen Stable Diffusion 1.5 UNet.
- Found and fixed a subtle `copy.deepcopy` / `requires_grad` bug that had silently frozen the encoder copy. This led to a controlled comparison between **12.6M** and **361M** trainable parameters.
- Reproduced the paper's "sudden convergence" phenomenon. The model began following the conditioning at around 2k–5k steps, and the norms of the zero-convolution weights grew from 0 as ControlNet "turned on".

## Approach
- **Zero convolutions:** 1×1 convolutions with weights and biases initialised to zero, so the pretrained model is undisturbed at step 0.
- **Trainable encoder copy:** a deep copy of the UNet's 12 encoder blocks and middle block.
- **Conditioning embedding:** an 8-layer convolutional hint encoder that maps 512×512 conditions to 64×64 latents. It follows the official code rather than the paper text.
- **Training:** MSE noise-prediction loss, AdamW (lr 1e-5), 50% prompt dropout, fp16, on an NVIDIA A100.
- **Inference:** DDIM (50 steps) with classifier-free guidance (scale 9.0), applying the conditioning to both the conditional and unconditional passes.

## Results
| Metric | V1 (12.6M trainable, 50K steps) | V2 (361M trainable, 25K steps) |
|---|---|---|
| FID ↓ | **119.81** | 134.11 |
| CLIP Score ↑ | 0.326 | **0.3275** |
| PSNR (dB) ↑ | **8.61** | 8.50 |
| Training time | 4.83 h | **2.77 h** |

## Key Insights
- More trainable parameters is not always better. On a simple conditioning task, training only the zero convolutions and the hint encoder beat full encoder training, given limited training steps.
- The fully trainable encoder showed "encoder drift" away from the pretrained features (lower zero-conv norm: 1.44 vs. 3.24).
- The architecture is efficient. Training needs 12–18 GB of VRAM on a single GPU and finishes in a few hours.

## Skills & Tools
PyTorch · Stable Diffusion · Diffusion Models · ControlNet · DDIM Sampling · Classifier-Free Guidance · FID / CLIP Score / PSNR
