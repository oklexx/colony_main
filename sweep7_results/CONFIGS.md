# Sweep 7 — Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 10.22M steps
**Hypothesis:** survival_coeff (need-weight multiplier) + reduced penalties → better population survival

## Reward Parameters

| Parameter | R7_0 | R7_1 | R7_2 | R7_3 |
|-----------|-----:|-----:|-----:|-----:|
| build_bonus | 25 | 25 | 25 | 25 |
| chain_bonus | 1 | 1 | 1 | 1 |
| novelty | 15 | 15 | 15 | 15 |
| daily_income | **6** | **6** | **6** | **15** |
| sale_bonus | 2.0 | 2.0 | 2.0 | 2.0 |
| game_over_penalty | **5** | **3** | **1** | **3** |
| diversity_bonus | 8 | 8 | 8 | 8 |
| death_penalty | **1** | **0** | **0** | **0** |
| base_lost_penalty | 30 | 30 | 30 | 30 |
| idle_build_penalty | -2 | -2 | -2 | -2 |
| housing_need_bonus | 8 | 8 | 8 | 8 |
| food_need_bonus | 6 | 6 | 6 | 6 |
| water_need_bonus | 6 | 6 | 6 | 6 |
| milestone_base_bonus | 30 | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 |
| debt_coeff | **0.005** | **0.003** | **0.001** | **0.003** |
| error_penalty | -1 | -1 | -1 | -1 |
| build_cost_penalty | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| tax_daily_bonus | 0.3 | 0.3 | 0.3 | 0.3 |
| born_bonus | 1 | 1 | 1 | **10** |
| survival_bonus | 0 | 0 | 0 | 0 |
| curriculum_stage | **1** | **1** | **1** | **2** |
| survival_coeff | **0.1** | **0.2** | **0.5** | **0.2** |

**Bold** = differs from R5_3 baseline (build=25, income=20, born=15, survival=5)

## What Changed Per Run

| Run | Key Changes vs R5_3 baseline |
|-----|------------------------------|
| R7_0_survival | income 20→6, game_over 10→5, death 2→1, debt 0.001→0.005, survival 5→0, **surv_coeff 0→0.1** |
| R7_1_nw_heavy | income 20→6, game_over 10→3, death 2→0, debt 0.001→0.003, survival 5→0, **surv_coeff 0→0.2** |
| R7_2_nw_max | income 20→6, game_over 10→1, death 2→0, debt 0.001→0.001, survival 5→0, **surv_coeff 0→0.5** |
| R7_3_s2_nw | income 20→15, game_over 10→3, death 2→0, debt 0.001→0.003, survival 5→0, born 15→10, **surv_coeff 0→0.2**, **stage 1→2** |

## Evaluation Results (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| **R7_0_survival** | **365** | **31** | **35** | **-1539.9** | **105.21** | FAIL |
| R7_1_nw_heavy | 731 | 50 | 5 | -3093.8 | 15.39 | FAIL |
| R7_2_nw_max | 731 | **65.5** | 17 | -2426.3 | 51.42 | FAIL |
| R7_3_s2_nw | 365 | 31 | 7 | -1744.8 | 21.21 | FAIL |

**Note:** All runs FAIL. R7_0 is the only run that learned (97.07 reward). survival_coeff > 0.1 breaks learning. survival_bonus (sweep 4) is far superior to survival_coeff.
