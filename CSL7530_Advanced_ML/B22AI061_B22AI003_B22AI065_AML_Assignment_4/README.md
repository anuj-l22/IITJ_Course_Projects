# Proximal Policy Optimization from Scratch: Clipped vs. Unclipped Objectives

**Course:** CSL7530 – Advanced Machine Learning
**Author:** Anuj

From-scratch PyTorch implementation of **Proximal Policy Optimization (PPO)** on Gymnasium's CartPole-v1. The code separates the policy network, value network, rollout generator, GAE advantage estimator, loss and update routine. The study compares PPO's clipped surrogate objective against the unclipped importance-sampled objective, sweeps the main hyperparameters, and visualises what the critic learns.

## Highlights
- Showed that **clipping trades peak return for stability**. The standard deviation across seeds was **10.4 for clipped vs. 175.4 for unclipped**, even though the unclipped variant reached a higher mean return (376.0 vs. 152.8).
- Per-update KL diagnostics explained the difference. Clipping keeps KL in a narrow band, while the unclipped objective allows KL spikes that line up with policy collapses.
- The learned value function peaks at the upright, still state and decays as the pole tilts or spins.

## Approach
- **Actor-critic:** a shared two-layer tanh MLP with a categorical policy head and a scalar value head, using orthogonal initialisation.
- **Rollouts:** T = 1024 steps per rollout, with bootstrapping across episode boundaries.
- **GAE:** γ = 0.99 and λ = 0.95, with per-rollout advantage normalisation.
- **Update:** 10 epochs of mini-batches of size 64 per rollout, value loss and an optional entropy bonus, and gradient-norm clipping at 0.5.
- **Diagnostics:** approximate KL, clip fraction and value MSE, plus rendered GIFs of the agent at several training checkpoints.

## Results
| Metric (60k steps, 3 seeds) | Clipped | Unclipped |
|---|---|---|
| Final mean return | 152.8 | 376.0 |
| Across-seed std (final) | **10.4** | 175.4 |
| Greedy eval return (best seed) | 176 | 500 |

**Hyperparameter sweeps:** learning rate and the number of PPO epochs showed clear monotonic trends; lr = 3e-3 reached the 500-return cap. Clip range, GAE λ and the entropy coefficient did not show clean trends at a single seed.

## Key Insights
- Where reliability matters more than one lucky run, clipped PPO is clearly preferable.
- Single-seed RL sweeps are noisy, so any difference should be judged against the variance between seeds.

## Skills & Tools
PyTorch · Gymnasium · Reinforcement Learning · PPO · Actor-Critic · Generalized Advantage Estimation
