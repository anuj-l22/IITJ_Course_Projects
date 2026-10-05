# Gradient Boosted Trees from Scratch (ESL Algorithms 10.3 & 10.4)

**Course:** CSL7530 – Advanced Machine Learning
**Author:** Anuj

From-scratch implementation of **Gradient Tree Boosting for regression** (Algorithm 10.3) and **Gradient Boosting for K-class classification** (Algorithm 10.4) from *The Elements of Statistical Learning*. The whole boosting loop was written by hand in NumPy: pseudo-residuals, leaf-value optimisation (including the Newton step for multi-class), softmax and shrinkage updates. `DecisionTreeRegressor` was used only as the base learner.

## Highlights
- **97.4% test accuracy on Digits (10 classes)**, compared with 83.3% for a single deep tree and 18.7% for a single stump.
- **R² = 0.83 on California Housing**, compared with 0.62 for a fully grown single tree.
- Robust losses (absolute error, Huber) reduced **test MSE by about 4.5×** compared with squared error when 5% outliers were present.

## Approach
- **Regression:** squared-error, absolute-error and Huber losses, implemented as modular loss classes that share an interface.
- **Classification:** K additive models with softmax probabilities and diagonal-Hessian Newton leaf values (the (K−1)/K factor).
- **Datasets:** California Housing, Friedman #1, Digits (K=10) and Wine (K=3).
- **Ablations:** number of iterations M, tree depth J, learning rate ν, loss function under outliers, and a depth × learning-rate grid.

## Results
| Task | Dataset | Metric | Score |
|---|---|---|---|
| Regression | California Housing | Test R² | 0.8185 |
| Regression | Friedman #1 | Test R² | 0.9376 |
| Classification | Digits (K=10) | Test Acc | 0.9704 |
| Classification | Wine (K=3) | Test Acc | 0.9815 |

## Key Insights
- Increasing M, depth or ν all widen the gap between train and test error. Shrinkage acted as the most effective regulariser.
- Depth and learning rate interact. The best Friedman #1 configuration was depth 3 with ν = 0.1.
- Even boosted stumps (R² 0.70) beat a single deep tree. This shows the value of stagewise ensembles of weak learners.

## Skills & Tools
Python · NumPy · scikit-learn · Gradient Boosting · Ensemble Methods · Bias–Variance Analysis
