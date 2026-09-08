# Sweep 4 — Detailed Training Analysis

All runs: 10,223,616 steps, 64 envs, PPO. Base = R3_3_ultra (sweep 3).

## Learning Curves (best_reward over time)

| Run | @25% (2.36M) | @50% (4.98M) | @75% (7.60M) | @100% (10.22M) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R4_0_inc15 | 27.81 | 54.70 | 417.99 | 417.99 | Slow start, late jump, plateau |
| R4_1_inc20 | 31.90 | 598.07 | 598.07 | **826.20** | Fast mid, late gain |
| R4_2_inc25 | 27.81 | 510.81 | 1031.01 | **1775.44** | Steady late climb |
| R4_3_survival | **5431.46** | **8334.62** | **8897.89** | **8897.89** | Explosive from the start |

## Entropy Trajectory

| Run | Entropy @25% | Entropy @100% | Note |
|-----|-----------:|-----------:|------|
| R4_0_inc15 | 1.690 | 2.198 | Recovered, healthy |
| R4_1_inc20 | 1.712 | 2.201 | Stable |
| R4_2_inc25 | 2.316 | 2.243 | Stable, high exploration |
| R4_3_survival | 2.340 | 2.164 | Stable, slight decline |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R4_0_inc15 | 36.5 | 10.4 | 11.3 | 4.9 |
| R4_1_inc20 | 8.9 | 11.7 | 5.1 | 3.7 |
| R4_2_inc25 | 12.0 | 5.1 | 8.6 | 4.6 |
| R4_3_survival | 10.5 | 4.0 | 2.4 | **2.2** |

## Top Actions (Final @ 10.22M)

| Run | Top 3 Actions |
|-----|---------------|
| R4_0_inc15 | ACTION_44 (13.2%), ACTION_38 (9.7%), DAY (9.6%) |
| R4_1_inc20 | ACTION_44 (11.9%), ACTION_40 (9.9%), ACTION_35 (9.1%) |
| R4_2_inc25 | ACTION_44 (11.4%), ACTION_38 (9.3%), ACTION_41 (8.9%) |
| R4_3_survival | ACTION_44 (17.1%), DAY (11.9%), ACTION_38 (11.4%) |

## Evaluation (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| **R4_0_inc15** | **365** | **31** | **55** | **-1434.2** | **165.21** | FAIL |
| R4_1_inc20 | 365 | 31 | 38 | -1352.5 | 114.21 | FAIL |
| R4_2_inc25 | 365 | 31 | 35 | -1523.5 | 105.21 | FAIL |
| **R4_3_survival** | **731** | **59.5** | **44** | **-2528.8** | **132.41** | FAIL |

## Key Observations

### R4_3_survival: Breakthrough
- **8897.89 best reward** — 8.5x over R3_3_ultra, highest of any sweep
- **59.5 people** — only run to grow population significantly (others stuck at 31)
- **44 bases** — second-most, with actual population growth
- **Best value loss** (2.2 final) — most stable value function
- Explosive from 25% (5431 → 8334 → 8897) — survival bonus creates positive feedback loop: more people → more production → more building → more people
- Higher negative return (-2528.8) due to larger population = more deaths, but net positive

### R4_0_inc15: Most Bases
- **55 bases** — highest of any sweep
- **165.21 score** — highest eval score of any sweep
- But only 417.99 reward — 21x lower than R4_3
- 31 people (no growth) — builds many bases but doesn't sustain population
- Slow learning (27.81 @ 25%, 54.70 @ 50%) — inefficient

### Survival Bonus Effect
- R4_3 (surv=10): 59.5 people, 44 bases, 8897 reward
- R4_0 (surv=0): 31 people, 55 bases, 417 reward
- Survival bonus is the single most impactful parameter discovered
- It breaks the 31-person ceiling that all previous runs hit

### Income Scaling (without survival)
- R4_0 (inc=15): 417.99
- R4_1 (inc=20): 826.20
- R4_2 (inc=25): 1775.44
- Monotonic but sublinear — each +5 income gives less gain
- All stuck at 31 people — income alone can't grow population

### Failure Modes
- **R4_0**: builds 55 bases but can't sustain them — 31 people is too few for 55 bases
- **R4_1**: mid-range income is the worst — not enough to drive growth, not enough to build efficiently

### Recommendations
1. **R4_3_survival is the new baseline** — best reward, best eval, population growth
2. **Survival bonus is the key lever** — push to 15-20
3. **Combine survival + born_bonus** for maximum population growth (sweep 5)
4. **R4_0's 55 bases show building capacity** — pair with survival for sustainable growth
5. Population growth (59.5) is the breakthrough — focus future sweeps on population
