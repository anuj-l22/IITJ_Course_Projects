# Gaussian Process Regression & Classification from Scratch

**Course:** CSL7530 – Advanced Machine Learning
**Author:** Anuj

From-scratch implementation of three algorithms from *Gaussian Processes for Machine Learning* (Rasmussen & Williams), using only NumPy and SciPy for linear algebra:
- Algorithm 2.1: Cholesky-based GP regression.
- Algorithm 3.1: Laplace-approximation mode finding with Newton's method.
- Algorithm 3.2: predictive probabilities for binary GP classification.

## Highlights
- **R² = 0.92 on the 5-D Friedman #1 dataset** with only 300 training points and one closed-form inference step.
- **98.3% accuracy on Two Moons** and **97.8% on Iris (binary)** with the Laplace GP classifier, which converged in 5–7 Newton iterations.
- The averaged predictive probability gave more conservative, better-calibrated outputs than the MAP prediction in regions of high uncertainty.

## Approach
- RBF and Matérn 3/2 kernels, parameterised by length scale and signal variance.
- Logistic likelihood with {−1, +1} labels. The probit approximation was used for the Gaussian–sigmoid integral.
- Visualised the full 2-D predictive covariance, GP prior and posterior samples, and reliability diagrams.
- Ablations over length scale ℓ, noise level σₙ and kernel type. The log marginal likelihood was used for model selection.

## Results
| Task | Dataset | Result |
|---|---|---|
| Regression | Friedman #1 (5-D) | R² 0.9206, MSE 1.91 |
| Classification | Two Moons | Acc 98.33% (5 Newton iterations) |
| Classification | Iris (binary) | Acc 97.78% (7 Newton iterations) |

## Key Insights
- The log marginal likelihood acts as an automatic Occam's razor. It picked the best length scale (ℓ ≈ 0.95) without cross-validation.
- Small ℓ overfits and large ℓ oversmooths. The choice of kernel matters less than tuning the length scale correctly.
- The O(n³) cost of the Cholesky decomposition limits exact GPs to moderate dataset sizes.

## Skills & Tools
Python · NumPy · SciPy · Gaussian Processes · Bayesian Inference · Laplace Approximation · Uncertainty Quantification
