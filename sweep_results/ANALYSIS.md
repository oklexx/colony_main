# Detailed Training Analysis

## Learning Curves (best_reward over time)

| Run | @25% (262k) | @50% (524k) | @75% (786k) | @100% (1048k) | Trend |
|-----|-----------:|-----------:|-----------:|-------------:|-------|
| R0_baseline | 20.59 | 20.59 | 20.59 | 20.59 | Flat |
| R1_income | 17.64 | 17.64 | 17.64 | 17.64 | Flat |
| R2_low_pen | 18.81 | 18.81 | 18.81 | 18.81 | Flat |
| R3_high_build | 32.13 | 32.13 | 32.81 | 32.81 | Slight gain |
| R4_production | 12.13 | 12.13 | 12.81 | 12.81 | Slight gain |
| R5_chain | 12.81 | 12.81 | 12.81 | 12.81 | Flat |
| R6_balanced | 26.90 | 26.90 | 26.90 | 26.90 | Flat |
| R7_no_death | 22.81 | 22.81 | 22.81 | 22.81 | Flat |
| R8_milestones | 12.81 | 12.81 | 12.81 | 12.81 | Flat |
| **R9_aggressive** | **34.64** | **34.64** | **228.98** | **496.14** | **Explosive growth** |
| R10_minimal | 22.81 | 22.81 | 22.81 | 22.81 | Flat |
| R11_low_debt | 19.59 | 19.59 | 19.59 | 19.59 | Flat |

## Entropy Collapse

| Run | Entropy Start | Entropy End | Drop |
|-----|-------------:|-----------:|-----:|
| R0_baseline | 2.558 | 1.966 | -23% |
| R1_income | 2.537 | 2.055 | -19% |
| R2_low_pen | 2.550 | 1.681 | -34% |
| R3_high_build | 2.543 | 1.921 | -24% |
| R4_production | 2.538 | 1.682 | -34% |
| R5_chain | 2.562 | 2.050 | -20% |
| R6_balanced | 2.546 | 1.681 | -34% |
| R7_no_death | 2.532 | 1.632 | -35% |
| R8_milestones | 2.541 | 2.108 | -17% |
| **R9_aggressive** | **2.548** | **1.882** | **-26%** |
| R10_minimal | 2.530 | 1.494 | -41% |
| R11_low_debt | 2.548 | 1.736 | -32% |

## Top Actions (Final)

| Run | Top 3 Actions |
|-----|---------------|
| R0_baseline | ACTION_37 (22.9%), ACTION_38 (22.0%), ACTION_43 (11.6%) |
| R1_income | ACTION_42 (22.8%), ACTION_35 (19.4%), DAY (14.9%) |
| R2_low_pen | DAY (48.5%), ACTION_44 (15.3%), ACTION_40 (5.8%) |
| R3_high_build | ACTION_44 (32.1%), ACTION_40 (20.4%), ACTION_42 (8.4%) |
| R4_production | ACTION_36 (35.2%), ACTION_34 (18.0%), ACTION_37 (12.3%) |
| R5_chain | DAY (23.2%), ACTION_37 (17.4%), ACTION_44 (15.9%) |
| R6_balanced | WEEK (39.7%), ACTION_38 (12.6%), ACTION_43 (10.6%) |
| R7_no_death | DAY (45.8%), WEEK (14.4%), ACTION_42 (7.9%) |
| R8_milestones | ACTION_41 (19.4%), ACTION_36 (16.8%), ACTION_40 (14.9%) |
| **R9_aggressive** | **WEEK (26.7%), ACTION_43 (18.2%), ACTION_40 (16.8%)** |
| R10_minimal | ACTION_44 (53.9%), DAY (7.5%), ACTION_36 (7.0%) |
| R11_low_debt | DAY (39.6%), ACTION_40 (8.7%), BUILD_GOLDMINE (7.6%) |

## Value Loss Trajectory

| Run | v_loss @25% | v_loss @50% | v_loss @75% | v_loss @100% |
|-----|-----------:|-----------:|-----------:|------------:|
| R0_baseline | 238.3 | 197.9 | 147.2 | 98.9 |
| R1_income | 244.3 | 203.8 | 174.1 | 137.4 |
| R2_low_pen | 191.8 | 163.8 | 96.5 | 56.5 |
| R3_high_build | 129.1 | 93.8 | 66.9 | 51.3 |
| R4_production | 255.7 | 195.5 | 118.8 | 74.9 |
| R5_chain | 268.3 | 251.3 | 195.1 | 131.7 |
| R6_balanced | 193.4 | 192.1 | 184.5 | 83.7 |
| R7_no_death | 192.3 | 189.7 | 160.5 | 93.1 |
| R8_milestones | 224.4 | 205.6 | 158.9 | 93.8 |
| **R9_aggressive** | **180.1** | **149.2** | **122.6** | **121.3** |
| R10_minimal | 144.8 | 119.6 | 109.4 | 67.3 |
| R11_low_debt | 204.3 | 165.3 | 121.4 | 95.4 |

## Key Observations

### R9_aggressive: Why It Works
- **Explosive reward growth** in last 25% (34→229→496) — suggests the agent discovers a positive feedback loop late in training
- **WEEK action dominance** (26.7%) — the agent learned to fast-forward time to accumulate income
- **BUILD_GOLDMINE** appears in top actions — mining for resources
- **Lowest avg_return** (-1871) among top performers — fewer catastrophic failures
- **Best eval**: 27 bases, 32.5 people, return -856 (best of all runs)

### Failure Modes
- **R2, R6, R10, R11** all collapse to `days=3897, people=50, bases=1` — the agent learns to do nothing and let time pass
- **R5, R8** stuck at 12.81 reward — likely hitting the game_over penalty floor repeatedly
- **R1** builds 31 bases but with 31 people — inefficient resource use

### Recommendations
1. **Start from R9 config** — it's the only one showing genuine learning progress
2. **Add curriculum** — R9's late-stage explosion suggests early stages are too hard
3. **Cap WEEK action** or add time-cost to prevent time-spamming
4. **Increase training to 5M-10M steps** for R9 to see full potential
5. **Combine R9 + R3**: build=30, income=4, needs=8, death=2
