# Sweep 3 — Income & Debt Scaling (10M steps)

**Total runs:** 4 | **All completed successfully** (returncode=0)
**Training:** 10,223,616 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)
**Base:** R2_4_R9sale (sweep 2) — build=25, income=6, sale=2.0, debt=0.02

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | Eval Score |
|------|-----|-------------|------------:|---------:|---------:|-----------:|
| 1 | **R3_3_ultra** | income=12, sale=3, debt=0.002 | **1047.36** | 10000 | 1238.0 | **114.21** |
| 2 | R3_1_inc8 | income=8, debt=0.005 | 807.60 | 10000 | 995.9 | 99.21 |
| 3 | R3_2_inc10 | income=10, debt=0.003 | 659.82 | 10000 | 1010.5 | 99.21 |
| 4 | R3_0_base10M | R2_4 exact on 10M | 191.94 | 10000 | 1378.3 | **114.21** |

## Key Findings

1. **R3_3_ultra dominates** — 1047.36 best reward, 5.5x over base. Higher income + sale + low debt compounds.
2. **R3_0 and R3_3 tie on eval** (114.21, 38 bases) — but R3_3 has 5.5x higher reward, meaning it's more efficient per step.
3. **Income scaling works** — 6→12 (R3_0→R3_3) nearly doubles best reward.
4. **Debt reduction amplifies income** — R3_1 (debt=0.005) beats R3_2 (debt=0.003) despite lower income, suggesting debt is a stronger lever than income in the 8-10 range.
5. **R3_0 (base) underperforms at 10M** — only 191.94 reward. The R2_4 config doesn't scale well; it needs the income/sale/debt boosts.

## Recommended Next Steps

- **Base on R3_3_ultra** — best reward and tied-best eval
- Push income to 15-25 (tested in sweep 4)
- Consider survival_bonus to break the 38-base ceiling (tested in sweep 4)
- Lower debt further (0.001) to see if it helps at higher income
