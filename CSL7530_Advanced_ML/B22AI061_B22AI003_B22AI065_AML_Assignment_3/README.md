# Variational Autoencoder & Conditional VAE on MNIST

**Course:** CSL7530 – Advanced Machine Learning
**Author:** Anuj

Implemented a **Variational Autoencoder (VAE)** and a **Conditional VAE (CVAE)** in PyTorch for generative modelling of handwritten digits. The VAE learns a smooth latent space to sample from. The CVAE conditions both the encoder and the decoder on the digit label, so it can generate a chosen digit in varied handwriting styles.

## Highlights
- **VAE test ELBO loss: 102.79.** The **CVAE improved this to 98.49**, with nearly all of the gain coming from better reconstruction.
- The CVAE reliably generates the requested digit while the latent vector controls the style (thickness, slant and proportions).
- A latent-dimension ablation showed a clear trade-off between reconstruction and regularisation, with total loss levelling off at d = 20.

## Approach
- **Architecture:** MLP encoder and decoder (784 → 400 → d), with the reparameterisation trick and a log-variance parameterisation.
- **Loss:** negative ELBO, combining binary cross-entropy reconstruction with the closed-form KL divergence to N(0, I).
- **CVAE:** the one-hot label is concatenated to the encoder input (794-d) and to the decoder input (d + 10).
- **Training:** Adam (lr 1e-3), batch size 128, 30 epochs, on a T4 GPU.
- **Analysis:** latent grid traversals, interpolation between digits, t-SNE of latent means, and a direct 2-D latent visualisation.

## Results: latent dimension ablation
| d | Test Loss | Recon (BCE) | KL |
|---|---|---|---|
| 2 | 152.04 | 145.94 | 6.10 |
| 10 | 107.90 | 89.51 | 18.39 |
| 20 | **103.94** | 78.47 | 25.47 |
| 50 | 103.98 | 77.34 | 26.64 |

## Key Insights
- Going from d=20 to d=50 gives almost no benefit, because the small reconstruction gain is cancelled out by a matching rise in KL.
- Conditioning on the label frees latent capacity for style, which improves both reconstruction and control.
- KL regularisation produces a continuous latent space without sharp jumps. Interpolations pass through plausible intermediate digits.

## Skills & Tools
PyTorch · Generative Models · Variational Inference · VAE / CVAE · t-SNE · Latent Space Analysis
