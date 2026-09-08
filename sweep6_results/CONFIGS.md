# Sweep 6 — Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 10.22M steps
**Hypothesis:** Lower build_bonus + higher born_bonus/income → population-focused growth

## Reward Parameters

| Parameter | R6_0 | R6_1 | R6_2 | R6_3 |
|-----------|-----:|-----:|-----:|-----:|
| build_bonus | **5** | **5** | **2** | **3** |
| chain_bonus | 1 | 1 | 1 | 1 |
| novelty | **10** | **10** | **10** | **10** |
| daily_income | **20** | **20** | **20** | **30** |
| sale_bonus | **5** | **5** | **5** | **5** |
| game_over_penalty | 10 | 10 | 10 | 10 |
| diversity_bonus | 8 | 8 | 8 | 8 |
| death_penalty | 2 | 2 | 2 | 2 |
| base_lost_penalty | 30 | 30 | 30 | 30 |
| idle_build_penalty | -2 | -2 | -2 | -2 |
| housing_need_bonus | **15** | **15** | **15** | **15** |
| food_need_bonus | **10** | **10** | **10** | **10** |
| water_need_bonus | **10** | **10** | **10** | **10** |
| milestone_base_bonus | 30 | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 |
| debt_coeff | **0.001** | **0.001** | **0.001** | **0.0005** |
| error_penalty | -1 | -1 | -1 | -1 |
| build_cost_penalty | **0.001** | **0.001** | **0.01** | **0.001** |
| tax_daily_bonus | 0.3 | 0.3 | 0.3 | 0.3 |
| born_bonus | **20** | **25** | **25** | **20** |
| survival_bonus | 0 | 0 | 0 | 0 |
| curriculum_stage | **1** | **2** | **1** | **1** |

**Bold** = differs from R5_3 baseline (build=25, income=20, born=15, survival=5)

## What Changed Per Run

| Run | Key Changes vs R5_3 baseline |
|-----|------------------------------|
| R6_0_pop_focus | build 25→5, income 20→20, born 15→20, survival 5→0, housing 8→15, food 6→10, water 6→10, novelty 15→10, build_cost 0.0001→0.001 |
| R6_1_s2_pop | build 25→5, born 15→25, survival 5→0, housing 8→15, food 6→10, water 6→10, novelty 15→10, build_cost 0.0001→0.001, **stage 1→2** |
| R6_2_min_build | build 25→2, born 15→25, survival 5→0, housing 8→15, food 6→10, water 6→10, novelty 15→10, **build_cost 0.0001→0.01** |
| R6_3_inc30 | build 25→3, income 20→30, born 15→20, survival 5→0, housing 8→15, food 6→10, water 6→10, novelty 15→10, debt 0.001→0.0005, build_cost 0.0001→0.001 |

## Evaluation Results (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R6_0_pop_focus | 3897 | 50 | 26 | -14388.3 | 79.66 | FAIL |
| R6_1_s2_pop | 365 | 31.5 | 8 | -1400.0 | 24.21 | FAIL |
| R6_2_min_build | 365 | 31 | 4 | -1764.0 | 12.21 | FAIL |
| **R6_3_inc30** | **731** | **50** | **28** | **-2180.6** | **84.39** | FAIL |

**Note:** All runs FAIL. R6_1 is the only run that learned (1230 reward). R6_3 has the best eval (84.39) but barely learned (2.90 reward). Low build_bonus is a dead end.
