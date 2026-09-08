# Sweep 6 — Detailed Training Analysis

All runs: 10,223,616 steps, 64 envs, PPO. Hypothesis: low build + high population.

## Learning Curves (best_reward over time)

| Run | @25% (2.36M) | @50% (4.98M) | @75% (7.60M) | @100% (10.22M) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R6_0_pop_focus | 4.90 | 4.90 | 5.90 | 5.90 | Flat — no learning |
| R6_1_s2_pop | 4.90 | 4.90 | 426.92 | **1230.75** | Flat until 75%, then jump |
| R6_2_min_build | 1.30 | 1.30 | 1.30 | 1.30 | Completely flat — no learning |
| R6_3_inc30 | 2.90 | 2.90 | 2.90 | 2.90 | Completely flat — no learning |

## Entropy Trajectory

| Run | Entropy @25% | Entropy @100% | Note |
|-----|-----------:|-----------:|------|
| R6_0_pop_focus | 1.777 | 2.184 | Recovered but no learning |
| R6_1_s2_pop | 1.164 | 1.592 | Collapsed early, partial recovery |
| R6_2_min_build | 1.762 | **0.789** | Severe collapse — policy frozen |
| R6_3_inc30 | 1.691 | 2.111 | Recovered but no learning |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R6_0_pop_focus | 48.6 | 7.0 | 6.6 | 7.0 |
| R6_1_s2_pop | **576.4** | 74.2 | 80.7 | 69.4 |
| R6_2_min_build | **668.9** | 128.6 | 94.6 | **506.6** |
| R6_3_inc30 | **348.9** | 8.0 | 5.0 | 5.1 |

## Top Actions (Final @ 10.22M)

| Run | Top 3 Actions |
|-----|---------------|
| R6_0_pop_focus | ACTION_44 (14.4%), DAY (14.3%), ACTION_37 (11.1%) |
| R6_1_s2_pop | ACTION_36 (25.5%), ACTION_37 (17.5%), WEEK (13.7%) |
| R6_2_min_build | **ACTION_41 (66.4%)**, ACTION_34 (20.7%), ACTION_35 (5.3%) |
| R6_3_inc30 | ACTION_44 (16.7%), DAY (12.3%), ACTION_40 (10.1%) |

## Evaluation (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R6_0_pop_focus | 3897 | 50 | 26 | -14388.3 | 79.66 | FAIL |
| R6_1_s2_pop | 365 | 31.5 | 8 | -1400.0 | 24.21 | FAIL |
| R6_2_min_build | 365 | 31 | 4 | -1764.0 | 12.21 | FAIL |
| **R6_3_inc30** | **731** | **50** | **28** | **-2180.6** | **84.39** | FAIL |

## Key Observations

### R6_1_s2_pop: Only Learner
- **1230.75 best reward** — only run that showed genuine learning
- Flat until 75% (4.90 → 426.92 → 1230.75) — very late bloom
- Stage 2 + build=5 + born=25 — the stage restriction actually helped focus learning
- But only 8 bases in eval — builds few, builds slow
- High value loss (69.4 final) — unstable value function

### R6_2_min_build: Total Failure
- **1.30 reward** — completely flat, zero learning
- **Entropy collapsed to 0.789** — policy frozen on ACTION_41 (66.4%)
- Value loss exploded to 506.6 — value function diverged
- build=2 is too low — no learning signal

### R6_3_inc30: Eval Winner, Training Loser
- **84.39 eval score** — best of the sweep, 28 bases, 50 people
- But **2.90 reward** — agent barely learned
- High value loss early (348.9) then stabilized (5.1)
- income=30 + build=3 + born=20 — income without build is wasted

### R6_0_pop_focus: Time-Spammer
- 5.90 reward, 3897 days — agent learned to fast-forward time
- 26 bases but 50 people at 3897 days — unsustainable
- Slowest run (1908s) — inefficient
- surv_coeff=0.05 didn't help

### Why This Sweep Failed
- **build_bonus < 5 starves the learning signal** — the agent can't get enough reward to learn
- **High born_bonus without build creates population without infrastructure** — 50 people, 4-8 bases
- **Stage 2 helped R6_1** by restricting actions, but made learning slower
- **The fundamental issue**: build_bonus is the primary learning signal. Removing it kills learning.

### Recommendations
1. **Abandon low-build approach** — build_bonus must be ≥ 15 for learning
2. **R6_1 is the only salvageable config** — but 1230 reward is far below R5_3 (9537)
3. **Return to R5_3/R4_3 baseline** — build=25, survival=10, born=15
4. **Population growth needs survival_bonus, not born_bonus alone** — R4_3 showed 59.5 people with survival=10
5. **Sweep 7 should test survival_coeff variants** — the right lever for population
