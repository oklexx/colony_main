# Hyperparameter Sweep Results

**Total runs:** 12 | **All completed successfully** (returncode=0)
**Training:** 1,048,576 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | FPS |
|------|-----|-------------|------------:|---------:|---------:|----:|
| 1 | **R9_aggressive** | High everything positive | **496.14** | 3667 | 93.8 | 11,245 |
| 2 | **R3_high_build** | Maximum build incentive | **32.81** | 2915 | 152.8 | 7,015 |
| 3 | **R6_balanced** | Everything moderate | **26.90** | 3815 | 110.9 | 9,749 |
| 4 | R7_no_death | Remove death fear | 22.81 | 3669 | 110.8 | 9,759 |
| 4 | R10_minimal | Almost no penalties | 22.81 | 2850 | 260.4 | 4,079 |
| 6 | **R0_baseline** | Current best config | **20.59** | 2943 | 96.5 | 11,245 |
| 7 | R11_low_debt | High income, low debt | 19.59 | 3031 | 248.3 | 4,280 |
| 8 | R2_low_pen | Reduce fear of loss | 18.81 | 2928 | 211.8 | 5,030 |
| 9 | R1_income | High income, lower production | 17.64 | 2700 | 140.6 | 7,640 |
| 10 | R4_production | Food/housing/water production | 12.81 | 2942 | 96.7 | 11,229 |
| 10 | R5_chain | Reward diversity | 12.81 | 3180 | 127.0 | 8,477 |
| 10 | R8_milestones | Milestone chasing | 12.81 | 3188 | 112.7 | 9,579 |

## Key Findings

1. **R9_aggressive dominates** — 24x better than baseline. High positive rewards + low death penalty (2.0) + high need bonuses drive aggressive building.
2. **R3_high_build** (build_bonus=30) is the best "moderate" config at 32.81.
3. **R6_balanced** (build=20, needs=5) performs well at 26.90.
4. **Penalty removal alone is insufficient** — R2, R7, R10 all land in the 18-23 range, barely above baseline.
5. **Production-focused configs underperform** — R4, R5, R8 all stuck at 12.81 (likely the game_over floor).
6. **Income focus hurts** — R1 (income=5) drops to 17.64, below baseline.

## Recommended Next Steps

- **Base on R9_aggressive** and tune down to avoid reward hacking
- Try: build_bonus=25, death_penalty=1.0, housing/food/water_need=8, income=4
- Add curriculum stages to prevent early collapse
- Increase total_timesteps for R9 to see if reward keeps climbing
