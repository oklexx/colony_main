# Sweep 2 — Detailed Training Analysis

All runs: 5,242,880 steps, 64 envs, PPO. Base = R9_aggressive (sweep 1).

## Learning Curves (best_reward over time)

| Run | @25% (1.31M) | @50% (2.62M) | @75% (3.93M) | @100% (5.24M) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R2_0_R9x5M | 27.81 | 490.16 | 490.16 | **698.99** | Steady climb, late jump |
| R2_1_R9inc7 | 35.13 | 266.13 | 266.13 | 266.13 | Plateau at 50% |
| R2_2_R9debt | 27.81 | 27.81 | 182.51 | 285.90 | Late gain |
| R2_3_R9max | 249.90 | 307.03 | 307.03 | **691.30** | Fast start, late jump |
| R2_4_R9sale | 34.52 | 34.52 | 34.52 | **402.69** | Slow start, explosive late |

## Entropy Trajectory

| Run | Entropy @25% | Entropy @100% | Note |
|-----|-----------:|-----------:|------|
| R2_0_R9x5M | 2.250 | 2.249 | Stable, healthy exploration |
| R2_1_R9inc7 | 1.103 | 1.786 | Collapsed early, partial recovery |
| R2_2_R9debt | 1.164 | 1.769 | Collapsed early, partial recovery |
| R2_3_R9max | 1.400 | 1.713 | Moderate collapse, recovered |
| R2_4_R9sale | 1.862 | 2.178 | Stable, good exploration |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R2_0_R9x5M | 77.3 | 26.3 | 7.1 | 4.3 |
| R2_1_R9inc7 | 49.7 | 13.9 | 14.1 | 8.6 |
| R2_2_R9debt | 57.5 | 9.6 | 9.3 | 24.7 |
| R2_3_R9max | 51.1 | 22.7 | 25.0 | 14.4 |
| R2_4_R9sale | 84.3 | 10.3 | 8.9 | 5.3 |

## Top Actions (Final @ 5.24M)

| Run | Top 3 Actions |
|-----|---------------|
| R2_0_R9x5M | ACTION_35 (10.6%), ACTION_38 (9.6%), ACTION_44 (8.8%) |
| R2_1_R9inc7 | ACTION_40 (19.4%), ACTION_36 (15.4%), ACTION_35 (14.2%) |
| R2_2_R9debt | ACTION_44 (29.8%), ACTION_34 (18.0%), ACTION_35 (17.3%) |
| R2_3_R9max | ACTION_44 (35.6%), ACTION_40 (16.5%), ACTION_34 (7.9%) |
| R2_4_R9sale | ACTION_44 (15.3%), ACTION_42 (13.2%), ACTION_38 (11.1%) |

## Evaluation (Final @ 5,242,880 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R2_0_R9x5M | 731 | 50 | 27 | -2235.1 | 81.39 | FAIL |
| R2_1_R9inc7 | 365 | 31 | 26 | -1596.3 | 78.21 | FAIL |
| R2_2_R9debt | 365 | 31 | 25 | -1453.0 | 75.21 | FAIL |
| R2_3_R9max | 365 | 31 | 29 | -1070.3 | 87.21 | FAIL |
| **R2_4_R9sale** | **365** | **31** | **35** | **-1147.8** | **105.21** | FAIL |

## Key Observations

### R2_4_R9sale: Best Eval
- **35 bases** — most of any run across all sweeps so far
- **Best eval score** (105.21) and best return-to-bases ratio
- Sale bonus (2.0) incentivizes selling resources → more building capital
- Slow reward start (34.52 @ 25%) but explosive late growth to 402.69
- Stable entropy (1.86 → 2.18) — healthy exploration maintained

### R2_0_R9x5M: Best Reward
- **698.99 best reward** — highest of the sweep
- Steady, reliable learning curve (27 → 490 → 699)
- Lowest final value loss (4.3) — best value function convergence
- But fewer bases (27) and higher negative return (-2235) than R2_4

### Failure Modes
- **R2_1 (income=7)**: entropy collapsed to 1.10 by 25%, never fully recovered. Income alone without sale/debt tuning destabilizes policy.
- **R2_2 (debt=0.005)**: similar early entropy collapse (1.16). Debt reduction alone insufficient.

### Recommendations
1. **R2_4 is the new baseline** — best eval, most bases, stable training
2. Push sale_bonus to 3.0-5.0 (tested in sweep 3-4)
3. Combine R2_4 sale + R2_3 low debt for best of both
4. Survival_bonus is the next lever (tested in sweep 4)
