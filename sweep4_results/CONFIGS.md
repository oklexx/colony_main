# Sweep 4 — Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 10.22M steps
**Base:** R3_3_ultra (sweep 3) — build=25, income=12, sale=3.0, debt=0.002

## Reward Parameters

| Parameter | R4_0 | R4_1 | R4_2 | R4_3 |
|-----------|-----:|-----:|-----:|-----:|
| build_bonus | 25 | 25 | 25 | 25 |
| chain_bonus | 1 | 1 | 1 | 1 |
| novelty | 15 | 15 | 15 | 15 |
| daily_income | **15** | **20** | **25** | **15** |
| sale_bonus | **2.0** | **5.0** | **5.0** | **2.0** |
| game_over_penalty | 10 | 10 | 10 | 10 |
| diversity_bonus | 8 | 8 | 8 | 8 |
| death_penalty | 2 | 2 | **1** | 2 |
| base_lost_penalty | 30 | 30 | 30 | 30 |
| idle_build_penalty | -2 | -2 | -2 | -2 |
| housing_need_bonus | 8 | 8 | 8 | 8 |
| food_need_bonus | 6 | 6 | 6 | 6 |
| water_need_bonus | 6 | 6 | 6 | 6 |
| milestone_base_bonus | 30 | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 |
| debt_coeff | **0.001** | **0.001** | **0.001** | **0.001** |
| error_penalty | -1 | -1 | -1 | -1 |
| build_cost_penalty | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| tax_daily_bonus | 0.3 | 0.3 | 0.3 | 0.3 |
| born_bonus | 1 | 1 | 1 | 1 |
| survival_bonus | **0** | **0** | **0** | **10** |

**Bold** = differs from R3_3 base

## What Changed Per Run

| Run | Key Changes vs R3_3 base |
|-----|--------------------------|
| R4_0_inc15 | income 12→15, sale 3→2, debt 0.002→0.001 |
| R4_1_inc20 | income 12→20, sale 3→5, debt 0.002→0.001 |
| R4_2_inc25 | income 12→25, sale 3→5, debt 0.002→0.001, death 2→1 |
| R4_3_survival | income 12→15, sale 3→2, debt 0.002→0.001, **survival 0→10** |

## Evaluation Results (Final @ 10,223,616 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| **R4_0_inc15** | **365** | **31** | **55** | **-1434.2** | **165.21** | FAIL |
| R4_1_inc20 | 365 | 31 | 38 | -1352.5 | 114.21 | FAIL |
| R4_2_inc25 | 365 | 31 | 35 | -1523.5 | 105.21 | FAIL |
| **R4_3_survival** | **731** | **59.5** | **44** | **-2528.8** | **132.41** | FAIL |

**Note:** All runs FAIL the evaluation thresholds. R4_0 has the most bases (55) and highest score (165.21). R4_3 has the best reward (8897.89) and only population growth (59.5 people).
