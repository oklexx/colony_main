# Sweep 3 — Detailed Training Analysis

All runs: 10,223,616 steps, 64 envs, PPO. Base = R2_4_R9sale (sweep 2).

## Learning Curves (best_reward over time)

| Run | @25% (2.36M) | @50% (4.98M) | @75% (7.60M) | @100% (10.22M) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R3_0_base10M | 27.81 | 27.81 | 27.81 | 191.94 | Flat until last 25%, then jump |
| R3_1_inc8 | 332.04 | 486.52 | 714.72 | **807.60** | Steady climb throughout |
| R3_2_inc10 | 27.81 | 373.13 | 540.91 | 659.82 | Slow start, steady late climb |
| R3_3_ultra | 334.79 | **1047.36** | 1047.36 | 1047.36 | Explosive by 50%, then plateau |

## Entropy Trajectory

| Run | Entropy @25% | Entropy @100% | Note |
|-----|-----------:|-----------:|------|
| R3_0_base10M | 1.489 | 2.054 | Recovered from early dip |
| R3_1_inc8 | 1.848 | 2.204 | Stable, healthy |
| R3_2_inc10 | 2.115 | 2.197 | Stable throughout |
| R3_3_ultra | 2.111 | 2.178 | Stable, good exploration |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R3_0_base10M | 74.4 | 17.6 | 5.6 | 4.2 |
| R3_1_inc8 | 17.7 | 5.1 | 3.3 | 2.8 |
| R3_2_inc10 | 9.2 | 17.2 | 10.2 | 8.0 |
| R3_3_ultra | 10.9 | 5.8 | 3.6 | 3.8 |

## Top Actions (Final @ 10.22M)

| Run | Top 3 Actions |
|-----|---------------|
| R3_0_base10M | ACTION_44 (17.8%), DAY (12.4%), ACTION_35 (12.0%) |
| R3_1_inc8 | ACTION_38 (12.7%), ACTION_44 (12.2%), DAY (9.2%) |
| R3_2_inc10 | ACTION_44 (12.8%), ACTION_38 (11.4%), ACTION_36 (10.6%) |
| R3_3_ultra | ACTION_44 (13.6%), ACTION_38 (9.3%), ACTION_40 (9.2%) |

## Evaluation (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R3_0_base10M | 365 | 31 | **38** | -1181.1 | **114.21** | FAIL |
| R3_1_inc8 | 365 | 31 | 33 | -1408.2 | 99.21 | FAIL |
| R3_2_inc10 | 365 | 31 | 33 | -1281.1 | 99.21 | FAIL |
| **R3_3_ultra** | **365** | **31** | **38** | **-1366.5** | **114.21** | FAIL |

## Key Observations

### R3_3_ultra: Best Reward
- **1047.36 best reward** — highest of the sweep, hit by 50% mark
- Tied-best eval (38 bases, 114.21 score) with R3_0
- Most efficient: reaches peak reward in half the training time
- Stable entropy (2.11 → 2.18) and low value loss (3.8 final)

### R3_0_base10M: Best Eval (tied)
- **38 bases, 114.21 score** — tied with R3_3
- But only 191.94 reward — 5.5x lower than R3_3
- Flat until last 25% (27.81 → 191.94) — slow learner
- Shows the base config CAN build well given enough time, but inefficiently

### Income Scaling Pattern
- R3_0 (inc=6): 191.94
- R3_1 (inc=8): 807.60 — 4.2x
- R3_2 (inc=10): 659.82 — 3.4x
- R3_3 (inc=12): 1047.36 — 5.5x
- Non-monotonic: R3_1 > R3_2 despite lower income. Debt (0.005 vs 0.003) matters more at mid-range income.

### Failure Modes
- **R3_0**: flat reward for 75% of training — the base config is too conservative to learn fast
- **R3_2**: slow start (27.81 @ 25%) — higher income without proportional debt reduction delays learning

### Recommendations
1. **R3_3_ultra is the new baseline** — best reward, tied-best eval, fastest convergence
2. Income 12 is the sweet spot so far — push to 15-25 (sweep 4)
3. Debt 0.002 works well with high income — try 0.001
4. The 38-base ceiling needs a new lever: survival_bonus or born_bonus (sweep 4-5)
