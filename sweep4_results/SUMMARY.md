# Sweep 4 — High Income & Survival Bonus (10M steps)

**Total runs:** 4 | **All completed successfully** (returncode=0)
**Training:** 10,223,616 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)
**Base:** R3_3_ultra (sweep 3) — build=25, income=12, sale=3.0, debt=0.002

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | Eval Score |
|------|-----|-------------|------------:|---------:|---------:|-----------:|
| 1 | **R4_3_survival** | income=15, survival=10, debt=0.001 | **8897.89** | 10000 | 1056.9 | **132.41** |
| 2 | R4_2_inc25 | income=25, sale=5, debt=0.001, death=1 | 1775.44 | 10000 | 1100.2 | 105.21 |
| 3 | R4_1_inc20 | income=20, sale=5, debt=0.001 | 826.20 | 10000 | 1753.6 | 114.21 |
| 4 | R4_0_inc15 | income=15, debt=0.001 | 417.99 | 10000 | 1141.1 | **165.21** |

## Key Findings

1. **R4_3_survival is a breakthrough** — 8897.89 reward (8.5x over R3_3), best eval (132.41), 44 bases, 59.5 people. Survival bonus is the missing lever.
2. **R4_0 wins on bases** — 55 bases, score 165.21 (highest eval score of any sweep). But only 417.99 reward — inefficient.
3. **Survival bonus > income scaling** — R4_3 (surv=10, inc=15) crushes R4_2 (inc=25, no surv). Survival drives population growth which compounds.
4. **Income alone has diminishing returns** — R4_0→R4_2 (15→25) only 4x reward, while adding survival (R4_3) gives 8.5x.
5. **R4_1 (inc=20) is the weakest** — 826.20, slower than both neighbors. Mid-range income without survival is suboptimal.

## Recommended Next Steps

- **Base on R4_3_survival** — best reward AND best eval
- Push survival_bonus higher (15-20) and pair with born_bonus (tested in sweep 5)
- Combine R4_0's 55-base building with R4_3's survival: income=15, survival=10, sale=2
- Population growth is the key — born_bonus and survival_bonus should be the primary levers going forward
