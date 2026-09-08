# Sweep 3 — Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 10.22M steps
**Base:** R2_4_R9sale (sweep 2) — build=25, income=6, sale=2.0, debt=0.02

## Reward Parameters

| Parameter | R3_0 | R3_1 | R3_2 | R3_3 |
|-----------|-----:|-----:|-----:|-----:|
| build_bonus | 25 | 25 | 25 | 25 |
| chain_bonus | 1 | 1 | 1 | 1 |
| novelty | 15 | 15 | 15 | 15 |
| daily_income | **6** | **8** | **10** | **12** |
| sale_bonus | **2.0** | **2.0** | **2.0** | **3.0** |
| game_over_penalty | 10 | 10 | 10 | 10 |
| diversity_bonus | 8 | 8 | 8 | 8 |
| death_penalty | 2 | 2 | 2 | 2 |
| base_lost_penalty | 30 | 30 | 30 | 30 |
| idle_build_penalty | -2 | -2 | -2 | -2 |
| housing_need_bonus | 8 | 8 | 8 | 8 |
| food_need_bonus | 6 | 6 | 6 | 6 |
| water_need_bonus | 6 | 6 | 6 | 6 |
| milestone_base_bonus | 30 | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 |
| debt_coeff | **0.02** | **0.005** | **0.003** | **0.002** |
| error_penalty | -1 | -1 | -1 | -1 |
| build_cost_penalty | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| tax_daily_bonus | 0.3 | 0.3 | 0.3 | 0.3 |
| born_bonus | 1 | 1 | 1 | 1 |
| survival_bonus | 0 | 0 | 0 | 0 |

**Bold** = differs from R3_0 (base)

## What Changed Per Run

| Run | Key Changes vs R2_4 base |
|-----|--------------------------|
| R3_0_base10M | (none — R2_4 exact, just 10M steps) |
| R3_1_inc8 | income 6→8, debt 0.02→0.005 |
| R3_2_inc10 | income 6→10, debt 0.02→0.003 |
| R3_3_ultra | income 6→12, sale 2.0→3.0, debt 0.02→0.002 |

## Evaluation Results (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R3_0_base10M | 365 | 31 | **38** | -1181.1 | **114.21** | FAIL |
| R3_1_inc8 | 365 | 31 | 33 | -1408.2 | 99.21 | FAIL |
| R3_2_inc10 | 365 | 31 | 33 | -1281.1 | 99.21 | FAIL |
| **R3_3_ultra** | **365** | **31** | **38** | **-1366.5** | **114.21** | FAIL |

**Note:** All runs FAIL the evaluation thresholds. R3_0 and R3_3 tie on eval score (114.21, 38 bases). R3_3 has 5.5x higher best reward.
