# Sweep 6 — Low Build, High Population (10M steps)

**Total runs:** 4 | **All completed successfully** (returncode=0)
**Training:** 10,223,616 steps each | 64 envs | PPO (lr=3e-4, gamma=0.99)
**Hypothesis:** Lower build_bonus + higher born_bonus/income → population-focused growth

## Leaderboard (by best_reward)

| Rank | Run | Description | Best Reward | Episodes | Time (s) | Eval Score |
|------|-----|-------------|------------:|---------:|---------:|-----------:|
| 1 | **R6_1_s2_pop** | Same + stage=2 | **1230.75** | 10000 | 1235.6 | 24.21 |
| 2 | R6_3_inc30 | income=30, build=3, born=20, debt=0.0005 | 2.90 | 10000 | 1262.7 | **84.39** |
| 3 | R6_0_pop_focus | Low build, high pop, high income, surv_coeff=0.05 | 5.90 | 10000 | 1908.6 | 79.66 |
| 4 | R6_2_min_build | build=2, build_cost=0.01, born=25 | 1.30 | 10000 | 1019.1 | 12.21 |

## Key Findings

1. **R6_1 is the only run that learned** — 1230.75 reward, 8 bases. Stage 2 + low build + high born works.
2. **R6_0, R6_2, R6_3 all collapsed** — reward < 6, entropy collapsed, value loss exploded. These configs are broken.
3. **R6_3 wins on eval** — 84.39 score, 28 bases, 50 people. But reward is 2.90 — the agent barely learned anything.
4. **R6_0 is the worst** — 5.90 reward, 1908s (slowest), 26 bases but 3897 days (time-spamming).
5. **Low build_bonus is a dead end** — build=2-3 starves the agent of learning signal. Only R6_1 (build=5, stage=2) survived.

## Recommended Next Steps

- **Abandon low-build approach** — build_bonus < 5 is too low for learning
- **R6_1 is salvageable** — 1230 reward with build=5, born=25, stage=2. Push born higher.
- **R6_3's 28 bases + 50 people** show the eval can work — but training signal is too weak
- **Return to R5_3/R4_3 baseline** — build=25 is essential for learning
- **surv_coeff (R6_0) didn't help** — 0.05 survival coefficient is insufficient
