# Sweep 5 — Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 10.22M steps
**Base:** R4_3_survival (sweep 4) — build=25, income=15, sale=2.0, debt=0.001, survival=10

## Reward Parameters

| Parameter | R5_0 | R5_1 | R5_2 | R5_3 |
|-----------|-----:|-----:|-----:|-----:|
| build_bonus | 25 | 25 | 25 | 25 |
| chain_bonus | 1 | 1 | 1 | 1 |
| novelty | 15 | 15 | 15 | 15 |
| daily_income | **15** | **15** | **15** | **20** |
| sale_bonus | 2.0 | 2.0 | 2.0 | 2.0 |
| game_over_penalty | 10 | 10 | 10 | 10 |
| diversity_bonus | 8 | 8 | 8 | 8 |
| death_penalty | 2 | 2 | 2 | **0** |
| base_lost_penalty | 30 | 30 | 30 | 30 |
| idle_build_penalty | -2 | -2 | -2 | -2 |
| housing_need_bonus | 8 | 8 | 8 | 8 |
| food_need_bonus | 6 | 6 | 6 | 6 |
| water_need_bonus | 6 | 6 | 6 | 6 |
| milestone_base_bonus | 30 | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 |
| debt_coeff | 0.001 | 0.001 | 0.001 | 0.001 |
| error_penalty | -1 | -1 | -1 | -1 |
| build_cost_penalty | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| tax_daily_bonus | 0.3 | 0.3 | 0.3 | 0.3 |
| born_bonus | **1** | **10** | **15** | **15** |
| survival_bonus | **0** | **0** | **5** | **5** |
| curriculum_stage | **2** | **2** | **2** | **1** |

**Bold** = differs from R4_3 base

## What Changed Per Run

| Run | Key Changes vs R4_3 base |
|-----|--------------------------|
| R5_0_s2_base | survival 10→0, **stage 1→2** |
| R5_1_s2_born10 | survival 10→0, **born 1→10**, **stage 1→2** |
| R5_2_s2_pop | survival 10→5, **born 1→15**, **stage 1→2** |
| R5_3_s1_pop | survival 10→5, **born 1→15**, income 15→20, **death 2→0**, **stage stays 1** |

## Evaluation Results (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| **R5_0_s2_base** | **365** | **31.5** | **39** | **-1119.3** | **117.21** | FAIL |
| R5_1_s2_born10 | 731 | **50** | 33 | -2445.7 | 99.39 | FAIL |
| R5_2_s2_pop | 548 | 42 | 30 | -1500.3 | 90.30 | FAIL |
| R5_3_s1_pop | 365 | 31 | 36 | -1005.2 | 108.21 | FAIL |

**Note:** All runs FAIL the evaluation thresholds. R5_0 has the best eval score (117.21). R5_3 has the best reward (9537.12). R5_1 has the most people (50).
