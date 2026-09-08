# Sweep 7 — Survival Coefficient & Need Weights (10M steps)

**Total runs:** 4 | **All completed successfully** (returncode=0)
**Training:** 10,223,616 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)
**Hypothesis:** survival_coeff (need-weight multiplier) + reduced penalties → better population survival

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | Eval Score |
|------|-----|-------------|------------:|---------:|---------:|-----------:|
| 1 | **R7_0_survival** | surv_coeff=0.1, game_over=5, death=1, debt=0.005 | **97.07** | 10000 | 2080.8 | 105.21 |
| 2 | R7_3_s2_nw | stage=2, surv_coeff=0.2, inc=15, born=10 | 35.13 | 10000 | 1231.2 | 21.21 |
| 3 | R7_1_nw_heavy | surv_coeff=0.2, game_over=3, death=0, debt=0.003 | 34.52 | 10000 | 1275.7 | 15.39 |
| 4 | R7_2_nw_max | surv_coeff=0.5, game_over=1, death=0, debt=0.001 | 27.81 | 10000 | 1514.7 | 51.42 |

## Key Findings

1. **R7_0 is the only run that learned** — 97.07 reward, 35 bases, 105.21 eval. surv_coeff=0.1 + reduced penalties works.
2. **R7_1, R7_2, R7_3 all collapsed** — reward < 36, entropy collapsed, value loss exploded. Aggressive survival coefficients break learning.
3. **R7_0 wins on eval** — 105.21 score, 35 bases. Moderate survival coefficient (0.1) is the sweet spot.
4. **Higher survival_coeff = worse** — 0.1 (R7_0) > 0.2 (R7_1, R7_3) > 0.5 (R7_2). The agent over-optimizes survival at the cost of building.
5. **Stage 2 (R7_3) didn't help** — 35.13 reward, 7 bases. Stage restriction + high surv_coeff is too constrained.

## Recommended Next Steps

- **R7_0 is the best of this sweep** — but 97.07 reward is far below R5_3 (9537) and R4_3 (8897)
- **survival_coeff=0.1 is the max** — higher values break learning
- **Return to R5_3/R4_3 baseline** — build=25, survival_bonus=10, born=15 is still the best
- **Sweep 7 confirms**: survival_bonus (sweep 4) > survival_coeff (sweep 7) for population growth
- **Next**: combine R4_3 survival_bonus=10 + R7_0 surv_coeff=0.1 for dual survival pressure
