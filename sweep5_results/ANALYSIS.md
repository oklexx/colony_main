# Sweep 5 — Detailed Training Analysis

All runs: 10,223,616 steps, 64 envs, PPO. Base = R4_3_survival (sweep 4).

## Learning Curves (best_reward over time)

| Run | @25% (2.36M) | @50% (4.98M) | @75% (7.60M) | @100% (10.22M) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R5_0_s2_base | 1671.14 | 2966.61 | 3934.20 | **4056.27** | Steady climb, late plateau |
| R5_1_s2_born10 | 2440.98 | 3582.82 | 3811.87 | 3811.87 | Fast start, plateau at 75% |
| R5_2_s2_pop | 1683.73 | 3587.64 | **5758.16** | 5758.16 | Steady late climb |
| R5_3_s1_pop | 1885.50 | 6862.96 | 8568.47 | **9537.12** | Explosive mid-to-late |

## Entropy Trajectory

| Run | Entropy @25% | Entropy @100% | Note |
|-----|-----------:|-----------:|------|
| R5_0_s2_base | 1.899 | 1.732 | Slight decline, ok |
| R5_1_s2_born10 | 2.214 | 1.709 | Decline, some collapse |
| R5_2_s2_pop | 2.321 | 2.067 | Stable, healthy |
| R5_3_s1_pop | 2.397 | 2.027 | Stable, good exploration |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R5_0_s2_base | 82.8 | 12.6 | 3.2 | 2.1 |
| R5_1_s2_born10 | 20.2 | 7.1 | 3.3 | 2.4 |
| R5_2_s2_pop | 7.2 | 10.5 | 4.6 | 3.5 |
| R5_3_s1_pop | 7.3 | 9.2 | 9.2 | 3.9 |

## Top Actions (Final @ 10.22M)

| Run | Top 3 Actions |
|-----|---------------|
| R5_0_s2_base | ACTION_40 (20.1%), WEEK (12.3%), BUILD_GOLDMINE (9.6%) |
| R5_1_s2_born10 | ACTION_40 (18.8%), WEEK (10.7%), ACTION_38 (10.6%) |
| R5_2_s2_pop | ACTION_40 (17.2%), BUILD_GOLDMINE (10.2%), ACTION_44 (10.0%) |
| R5_3_s1_pop | ACTION_40 (15.9%), ACTION_44 (9.5%), ACTION_42 (8.4%) |

## Evaluation (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| **R5_0_s2_base** | **365** | **31.5** | **39** | **-1119.3** | **117.21** | FAIL |
| R5_1_s2_born10 | 731 | **50** | 33 | -2445.7 | 99.39 | FAIL |
| R5_2_s2_pop | 548 | 42 | 30 | -1500.3 | 90.30 | FAIL |
| R5_3_s1_pop | 365 | 31 | 36 | -1005.2 | 108.21 | FAIL |

## Key Observations

### R5_3_s1_pop: Best Reward
- **9537.12 best reward** — highest of the sweep, edges out R4_3 (8897.89)
- Explosive mid-to-late (1885 → 6862 → 8568 → 9537)
- Born=15 + income=20 + death=0 + stage=1
- But eval people is only 31 — no population growth in eval despite high reward
- Stable entropy (2.40 → 2.03) and good value loss (3.9)

### R5_0_s2_base: Best Eval
- **117.21 score** — highest eval score of the sweep
- 39 bases, 31.5 people
- Stage 2 + inc=15 builds efficiently
- But only 4056 reward — 2.4x lower than R5_3

### Curriculum Stage Effect
- Stage 1 (R5_3): 9537 reward, 36 bases, 31 people
- Stage 2 (R5_0): 4056 reward, 39 bases, 31.5 people
- Stage 2 is harder — restricts early actions, slows learning
- Stage 1 allows more freedom → faster reward growth
- **Stage 1 is the winner** for reward, stage 2 for eval bases

### Born Bonus Effect
- R5_1 (born=10): 50 people in eval, 3811 reward
- R5_2 (born=15): 42 people in eval, 5758 reward
- Born bonus grows population in eval but doesn't translate to higher reward
- R5_3 (born=15, stage=1): 31 people in eval, 9537 reward — stage matters more than born

### Failure Modes
- **R5_1**: entropy collapsed to 1.71 by end — born=10 + stage=2 is too constrained
- **R5_2**: good reward but lowest eval score (90.30) — population growth doesn't help eval
- **R5_3**: best reward but no population growth in eval (31 people) — reward and population are decoupled

### Recommendations
1. **R5_3_s1_pop is the best reward config** — use as baseline
2. **Stage 1 > Stage 2** for reward — stage 2 is too hard
3. **Born bonus grows eval population** but doesn't improve reward — different objective
4. **Combine R5_3 + R4_3 survival** — survival=10 + born=15 + stage=1 for max population AND reward
5. The reward-vs-population tradeoff needs resolution: survival_bonus is the bridge (sweep 4 showed 59.5 people)
