# Reward Configuration Comparison

All runs share: PPO, lr=3e-4, gamma=0.99, gae=0.98, clip=0.2, net=[256,256], 64 envs, 1M steps

## Reward Parameters

| Parameter | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 |
|-----------|----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|----:|----:|
| build_bonus | 16 | 16 | 16 | **30** | 10 | 10 | 20 | 20 | 10 | 25 | 20 | 15 |
| chain_bonus | 1 | 1 | 1 | 1 | 1 | **5** | 1 | 1 | 1 | 1 | 1 | 1 |
| novelty | 15 | 15 | 15 | 15 | 15 | **25** | 15 | 15 | 15 | 15 | 15 | 15 |
| daily_income | 1 | **5** | 2 | 2 | 3 | 2 | 3 | 3 | 2 | 4 | 3 | **5** |
| game_over_penalty | 10 | 10 | **5** | 10 | 10 | 10 | 10 | **2** | 10 | 10 | **2** | 10 |
| diversity_bonus | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| death_penalty | 5 | 5 | **2** | 3 | 3 | 3 | 3 | **0** | 3 | **2** | **1** | 3 |
| base_lost_penalty | 30 | 30 | **10** | 30 | 30 | 30 | 30 | **5** | 30 | 30 | **5** | 30 |
| idle_build_penalty | -2 | -2 | **-1** | -2 | -2 | -2 | -2 | -2 | -2 | -2 | **0** | -2 |
| housing_need_bonus | 6 | 4 | 6 | 5 | **10** | 6 | 5 | 6 | 6 | **8** | 6 | 5 |
| food_need_bonus | 4 | 3 | 4 | 3 | **8** | 4 | 5 | 4 | 4 | **6** | 4 | 4 |
| water_need_bonus | 4 | 3 | 4 | 3 | **8** | 4 | 5 | 4 | 4 | **6** | 4 | 4 |
| milestone_base_bonus | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | **50** | 30 | 30 | 30 |
| milestone_people_bonus | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **5** | 2 | 2 | 2 |
| debt_coeff | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | **0.005** |
| error_penalty | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | -1 | **0** | -1 |

**Bold** = differs from R0_baseline

## What Changed Per Run

| Run | Key Changes vs R0 |
|-----|-------------------|
| R1_income | income 1→5, housing 6→4, food 4→3, water 4→3 |
| R2_low_pen | income 1→2, game_over 10→5, idle -2→-1, death 5→2, base_lost 30→10 |
| R3_high_build | build 16→30, income 1→2, death 5→3, housing 6→5, food 4→3, water 4→3 |
| R4_production | build 16→10, income 1→3, death 5→3, housing 6→10, food 4→8, water 4→8 |
| R5_chain | build 16→10, chain 1→5, novelty 15→25, income 1→2, death 5→3 |
| R6_balanced | build 16→20, income 1→3, death 5→3, housing 6→5, food 4→5, water 4→5 |
| R7_no_death | build 16→20, income 1→3, game_over 10→2, death 5→0, base_lost 30→5 |
| R8_milestones | build 16→10, income 1→2, death 5→3, milestone_base 30→50, milestone_people 2→5 |
| R9_aggressive | build 16→25, income 1→4, death 5→2, housing 6→8, food 4→6, water 4→6 |
| R10_minimal | build 16→20, income 1→3, game_over 10→2, idle -2→0, death 5→1, base_lost 30→5, error -1→0 |
| R11_low_debt | build 16→15, income 1→5, death 5→3, debt_coeff 0.02→0.005, housing 6→5 |

## Evaluation Results (Final @ 1,048,576 steps)

| Run | Days | People | Bases | Return | Score | Thresholds |
|-----|-----:|-------:|------:|-------:|------:|------------|
| R0_baseline | 365 | 31 | 8 | -1735 | 24.21 | FAIL |
| R1_income | 365 | 31.5 | 31 | -2317 | 93.21 | FAIL |
| R2_low_pen | 3897 | 50 | 1 | -11670 | 4.66 | FAIL |
| R3_high_build | 365 | 31 | 4 | -1292 | 12.21 | FAIL |
| R4_production | 365 | 31 | 14 | -1682 | 42.21 | FAIL |
| R5_chain | 1096 | 50 | 1.5 | -3659 | 5.04 | FAIL |
| R6_balanced | 3897 | 50 | 1 | -2755 | 4.66 | FAIL |
| R7_no_death | 731 | 67.5 | 6 | -2510 | 18.43 | FAIL |
| R8_milestones | 365 | 34 | 4.5 | -1770 | 13.71 | FAIL |
| **R9_aggressive** | **365** | **32.5** | **27** | **-856** | **81.21** | FAIL |
| R10_minimal | 3897 | 50 | 1 | -10368 | 4.66 | FAIL |
| R11_low_debt | 3897 | 50 | 1 | -12274 | 4.66 | FAIL |

**Note:** All runs FAIL the evaluation thresholds. R9_aggressive has the best eval score (81.21) with 27 bases and lowest negative return (-856).
