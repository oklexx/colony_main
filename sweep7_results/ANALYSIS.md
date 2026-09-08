# Sweep 7 — Detailed Training Analysis

All runs: 10,223,616 steps, 64 envs, PPO. Hypothesis: survival_coeff + reduced penalties → better population.

## Learning Curves (best_reward over time)

| Run | @25% (2.36M) | @50% (4.98M) | @75% (7.60M) | @100% (10.22M) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R7_0_survival | 27.81 | 27.81 | 27.81 | **97.07** | Flat until last 25%, then jump |
| R7_1_nw_heavy | 34.52 | 34.52 | 34.52 | 34.52 | Completely flat — no learning |
| R7_2_nw_max | 27.81 | 27.81 | 27.81 | 27.81 | Completely flat — no learning |
| R7_3_s2_nw | 35.13 | 35.13 | 35.13 | 35.13 | Completely flat — no learning |

## Entropy Trajectory

| Run | Entropy @25% | Entropy @100% | Note |
|-----|-----------:|-----------:|------|
| R7_0_survival | **0.480** | 2.006 | Severe early collapse, recovered |
| R7_1_nw_heavy | 1.564 | 1.589 | Stable but low — no learning |
| R7_2_nw_max | 0.984 | 2.090 | Early collapse, recovered |
| R7_3_s2_nw | 1.060 | 0.968 | Collapsed and stayed low |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R7_0_survival | 133.6 | 19.2 | 2.4 | 2.9 |
| R7_1_nw_heavy | **973.0** | 371.9 | 226.1 | 63.7 |
| R7_2_nw_max | **1048.9** | 330.8 | 39.2 | 2.7 |
| R7_3_s2_nw | **1020.9** | **1047.6** | 453.9 | 99.2 |

## Top Actions (Final @ 10.22M)

| Run | Top 3 Actions |
|-----|---------------|
| R7_0_survival | ACTION_36 (20.5%), ACTION_34 (15.3%), ACTION_43 (13.5%) |
| R7_1_nw_heavy | ACTION_34 (36.8%), ACTION_44 (19.8%), ACTION_41 (12.8%) |
| R7_2_nw_max | ACTION_44 (23.9%), ACTION_36 (22.4%), ACTION_40 (8.1%) |
| R7_3_s2_nw | ACTION_40 (54.5%), ACTION_42 (29.1%), ACTION_43 (5.9%) |

## Evaluation (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| **R7_0_survival** | **365** | **31** | **35** | **-1539.9** | **105.21** | FAIL |
| R7_1_nw_heavy | 731 | 50 | 5 | -3093.8 | 15.39 | FAIL |
| R7_2_nw_max | 731 | **65.5** | 17 | -2426.3 | 51.42 | FAIL |
| R7_3_s2_nw | 365 | 31 | 7 | -1744.8 | 21.21 | FAIL |

## Key Observations

### R7_0_survival: Only Learner
- **97.07 best reward** — only run that showed genuine learning
- Flat until last 25% (27.81 → 97.07) — very late bloom
- surv_coeff=0.1 + game_over=5 + death=1 + debt=0.005 — moderate penalties
- 35 bases, 31 people, 105.21 eval — decent but far below R5_3/R4_3
- Severe early entropy collapse (0.480) then recovery — unstable start

### R7_2_nw_max: Most People, No Learning
- **65.5 people** — most of any sweep
- But **27.81 reward** — zero learning
- surv_coeff=0.5 + game_over=1 + death=0 — agent over-optimizes survival
- 17 bases — builds some but can't sustain
- Value loss exploded early (1048.9) then stabilized (2.7)

### R7_1_nw_heavy: Frozen
- **34.52 reward** — flat, no learning
- ACTION_34 dominant (36.8%) — policy stuck
- Value loss stayed high (63.7 final) — never converged
- surv_coeff=0.2 + game_over=3 — too much survival pressure

### R7_3_s2_nw: Double Failure
- **35.13 reward** — flat, no learning
- **Value loss exploded** (1020.9 → 1047.6 → 453.9 → 99.2) — worst of all
- Stage 2 + surv_coeff=0.2 + born=10 — too many constraints
- ACTION_40 dominant (54.5%) — policy frozen

### Why This Sweep Failed
- **survival_coeff > 0.1 breaks learning** — the agent over-optimizes survival at the cost of building
- **Reduced penalties (game_over, death) without survival_bonus** creates a survival-focused but build-starved agent
- **survival_coeff is the wrong lever** — survival_bonus (sweep 4) is better because it rewards survival directly, not via need-weight multiplier
- **Stage 2 + high surv_coeff (R7_3) is the worst combination** — too many constraints

### Comparison: survival_bonus vs survival_coeff
| Metric | R4_3 (surv_bonus=10) | R7_0 (surv_coeff=0.1) |
|--------|---------------------|----------------------|
| Best Reward | **8897.89** | 97.07 |
| Eval People | **59.5** | 31 |
| Eval Bases | 44 | 35 |
| Eval Score | 132.41 | 105.21 |
| Value Loss (final) | **2.2** | 2.9 |

**survival_bonus is 91x better on reward and 2x better on population.**

### Recommendations
1. **Abandon survival_coeff approach** — survival_bonus (sweep 4) is far superior
2. **R7_0 is salvageable** — 97.07 reward, 105.21 eval. But far below R5_3/R4_3
3. **Return to R5_3/R4_3 baseline** — build=25, survival_bonus=10, born=15
4. **If combining**: R4_3 survival_bonus=10 + R7_0 surv_coeff=0.1 for dual survival pressure
5. **Sweep 8 should test**: survival_bonus=15-20 + born=15 + income=20 for max population
