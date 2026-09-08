# Sweep 5 — Population & Curriculum (10M steps)

**Total runs:** 4 | **All completed successfully** (returncode=0)
**Training:** 10,223,616 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)
**Base:** R4_3_survival (sweep 4) — build=25, income=15, sale=2.0, debt=0.001, survival=10

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | Eval Score |
|------|-----|-------------|------------:|---------:|---------:|-----------:|
| 1 | **R5_3_s1_pop** | stage=1, born=15, inc=20, death=0 | **9537.12** | 10000 | 1094.4 | 108.21 |
| 2 | R5_2_s2_pop | stage=2, born=15, survival=5 | 5758.16 | 10000 | 1193.4 | 90.30 |
| 3 | R5_0_s2_base | stage=2, inc=15 | 4056.27 | 10000 | 1067.9 | 117.21 |
| 4 | R5_1_s2_born10 | stage=2, born=10 | 3811.87 | 10000 | 1378.8 | 99.39 |

## Key Findings

1. **R5_3_s1_pop is the best reward** — 9537.12, slight edge over R4_3 (8897.89). Born=15 + inc=20 + death=0 + stage=1 works.
2. **Curriculum stage matters** — R5_0/R5_1/R5_2 (stage=2) all underperform R5_3 (stage=1). Stage 2 is too hard.
3. **Born_bonus drives population** — R5_1 (born=10): 50 people, R5_2 (born=15): 42 people. But neither beats R5_3 on reward.
4. **R5_0 wins on eval** — 117.21 score, 39 bases. Stage 2 + inc=15 builds well but doesn't grow population.
5. **Death=0 (R5_3) helps reward** — no death penalty means population compounds faster, but eval people is only 31 (no growth in eval).

## Recommended Next Steps

- **Base on R5_3_s1_pop** — best reward, stage=1 is easier and works better
- Keep born_bonus=15, income=20, death=0
- Try survival_bonus=10 (from R4_3) combined with born=15 for max population
- Stage 1 is the sweet spot — stage 2 is too hard and hurts learning
- Population growth in eval is still weak (31-50) — need stronger survival + born combo
