# GRAM Implementation Notes

This branch adds a paper-faithful implementation of **Generative Recursive Reasoning Models (GRAM)** from arXiv:2605.19376v1.

Implemented details:

- Latent state `z=(h,l)` with high-level `h` and low-level `l`.
- Encoder with token embeddings, optional 16-token ARC puzzle embeddings, and RoPE/learned/no positional encodings.
- Separate recursive modules `fL` and `fH`, each configurable as `[Attention + SwiGLU] x 2`.
- Sudoku option `mlp_t=True` for the paper's SwiGLU-only recursive-core exception.
- Fixed learned checkpoint buffers for `h0,l0`, initialized once from `N(0,I)`.
- Stochastic high-level guidance:
  - prior `p_theta(eps_t | u_t) = N(mu_theta(u_t), sigma_theta(u_t)^2 I)`
  - posterior `q_phi(eps_t | u_t,y) = N(mu_phi(u_t,y), sigma_phi(u_t,y)^2 I)`
  - `h_t = u_t + eps_t`
- Training samples from the posterior; inference samples from the prior.
- Truncated GRAM surrogate objective:
  - each supervision step runs `T` high-level stochastic transitions
  - only the final transition of each supervision step carries recursive-core gradients
  - reconstruction and final-step KL are applied at every supervision step
- Deep supervision with `N_sup=16`.
- KL balancing coefficient `0.8`.
- Task KL coefficients:
  - Sudoku-Extreme: `beta=0.1`
  - ARC-AGI-1: `beta=0.04`
- SwiGLU decoder MLP followed by LM projection.
- ACT halt/continue Q head.
- LPRM value head trained with final prediction accuracy targets.
- Explicit training entrypoints:
  - `scripts/train_gram_sudoku_extreme.py`
  - `scripts/train_gram_arc_agi_1.py`

Paper hyperparameters included in the train files:

| Setting | Sudoku-Extreme | ARC-AGI-1 |
|---|---:|---:|
| Epochs | 50K | 200K |
| Global batch size | 768 | 768 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | 1.0 | 1.0 |
| Gradient clipping | 1.0 | 1.0 |
| EMA decay | 0.9999 | 0.9999 |
| Hidden size | 512 | 512 |
| Attention heads | 8 | 8 |
| `fL`, `fH` layers | 2 each | 2 each |
| High-level transitions `T` | 3 | 3 |
| Low-level refinements `K` | 6 | 4 |
| Deep supervision `N_sup` | 16 | 16 |
| KL balance | 0.8 | 0.8 |
| KL beta | 0.1 | 0.04 |
| Puzzle embeddings | no | yes, 16 tokens |

The official GRAM code is not available in this repository. The posterior target-conditioning pathway is implemented as additive conditioning of the deterministic proposal `u_t` with embedded target tokens, which is the minimal architecture consistent with the paper's `q_phi(eps_t | u_t, y)` specification.
