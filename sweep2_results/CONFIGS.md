# Sweep 2 — Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 5.24M steps
**Base:** R9_aggressive (sweep 1) — build=25, income=4, death=2, needs 8/6/6

## Reward Parameters

| Parameter | R2_0 | R2_1 | R2_2 | R2_3 | R2_4 |
|-----------|-----:|-----:|-----:|-----:|-----:|
| build_bonus | 25 | 25 | 25 | 25 | 25 |
| chain_bonus | 1 | 1 | 1 | 1 | 1 |
| novelty | 15 | 15 | 15 | 15 | 15 |
| daily_income | **4** | **7** | **5** | **10** | **6** |
| sale_bonus | **0.2** | **0.2** | **0.2** | **0.2** | **2.0** |
| game_over_penalty | 10 | 10 | 10 | 10 | 10 |
| diversity_bonus | 8 | 8 | 8 | 8 | 8 |
| death_penalty | 2 | 2 | 2 | **1** | 2 |
| base_lost_penalty | 30 | 30 | 30 | 30 | 30 |
| idle_build_penalty | -2 | -2 | -2 | -2 | -2 |
| housing_need_bonus | 8 | 8 | 8 | 8 | 8 |
| food_need_bonus | 6 | 6 | 6 | 6 | 6 |
| water_need_bonus | 6 | 6 | 6 | 6 | 6 |
| milestone_base_bonus | 30 | 30 | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 | 2 |
| debt_coeff | 0.02 | 0.02 | **0.005** | **0.003** | 0.02 |
| error_penalty | -1 | -1 | -1 | -1 | -1 |
| build_cost_penalty | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| tax_daily_bonus | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| born_bonus | 1 | 1 | 1 | 1 | 1 |
| survival_bonus | 0 | 0 | 0 | 0 | 0 |

**Bold** = differs from R2_0 (R9 base)

## What Changed Per Run

| Run | Key Changes vs R9 base |
|-----|------------------------|
| R2_0_R9x5M | (none — R9 exact, just 5M steps) |
| R2_1_R9inc7 | income 4→7 |
| R2_2_R9debt | income 4→5, debt 0.02→0.005 |
| R2_3_R9max | income 4→10, death 2→1, debt 0.02→0.003 |
| R2_4_R9sale | income 4→6, sale 0.2→2.0 |

## Evaluation Results (Final @ 5,242,880 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R2_0_R9x5M | 731 | 50 | 27 | -2235.1 | 81.39 | FAIL |
| R2_1_R9inc7 | 365 | 31 | 26 | -1596.3 | 78.21 | FAIL |
| R2_2_R9debt | 365 | 31 | 25 | -1453.0 | 75.21 | FAIL |
| R2_3_R9max | 365 | 31 | 29 | -1070.3 | 87.21 | FAIL |
| **R2_4_R9sale** | **365** | **31** | **35** | **-1147.8** | **105.21** | FAIL |

**Note:** All runs FAIL the evaluation thresholds. R2_4_R9sale has the best eval score (105.21) with 35 bases.
