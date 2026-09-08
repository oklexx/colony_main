# Sweep 2 — R9 Refined (5M steps)

**Total runs:** 5 | **All completed successfully** (returncode=0)
**Training:** 5,242,880 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)
**Base:** R9_aggressive from sweep 1 (build=25, income=4, death=2, needs 8/6/6)

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | Eval Score |
|------|-----|-------------|------------:|---------:|---------:|-----------:|
| 1 | **R2_0_R9x5M** | R9 exact on 5M | **698.99** | 10000 | 527.1 | 81.39 |
| 2 | **R2_4_R9sale** | R9 + sale=2.0, income=6 | **402.69** | 10000 | 508.7 | **105.21** |
| 3 | R2_2_R9debt | R9 + debt_coeff=0.005 | 285.90 | 10000 | 661.2 | 75.21 |
| 4 | R2_1_R9inc7 | R9 + income=7 | 266.13 | 10000 | 527.4 | 78.21 |
| 5 | **R2_3_R9max** | R9 + income=10, death=1, debt=0.003 | **691.30** | 10000 | 554.5 | 87.21 |

## Key Findings

1. **R2_0 (R9 exact) is the most reliable** — 698.99 best reward, stable across training.
2. **R2_4 (sale bonus) wins on eval** — 35 bases, score 105.21, best return-to-bases ratio. Sale bonus drives more building.
3. **R2_3 (max) matches R2_0 on reward** (691.30) but with higher income/death/debt tweaks — more aggressive, fewer bases (29).
4. **Raising income alone (R2_1) hurts** — 266.13, well below baseline R9. Income without sale/debt tuning destabilizes.
5. **Debt reduction alone (R2_2) is neutral** — 285.90, modest gain over R9 base.

## Recommended Next Steps

- **Base on R2_4 (sale=2.0, income=6)** — best eval score, most bases (35)
- Push sale_bonus higher (3.0-5.0) and pair with income=10-15
- Keep debt_coeff low (0.003-0.005) as in R2_3
- Consider survival_bonus next (tested in sweep 4)
