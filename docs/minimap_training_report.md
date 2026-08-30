# Minimap observation pipeline — training report

**Date:** 2026-08-30
**Hardware:** RTX 5080, torch 2.13.0+cu130, CUDA bf16 AMP
**Repo:** `sakhalin_colony_main` (branch `master`)

## Objective

Add a spatial minimap observation (2D tensor) alongside the existing flat 203-dim
observation, build a CNN variant of the actor-critic, and compare training
performance of **flat MLP** vs **minimap CNN** on the Sakhalin Colony PPO task.

## What was built

### C++ (`include/colony/env.h`, `src/env.cpp`, `src/bindings.cpp`)

- `ColonyEnvCpp::minimap()` — returns `[8, 2R+1, 2R+1]` float32 (R=14 → 29×29).
  8 channels: 0=LT_NORMAL (суша), 1=LT_WATER, 2=LT_WOOD, 3=LT_COAL,
  4=LT_IRON, 5=LT_OIL, 6=LT_GOLD, 7=occupied building. Centered on
  `earth.init_sel_x/y`. Out-of-bounds = 0.
- `ColonyVecEnvCpp::minimap_batch()` — same for all envs, returns `[B, 8, N, N]`.
- `src/bindings.cpp` exposes both as numpy arrays.
- **Critical fix:** `Game::refresh_occupied()` was only called at construction.
  Added `g.refresh_occupied()` in `ColonyEnvCpp::step()` after the action so
  channel 7 (occupied) actually updates as bases are built/destroyed. Without
  this the minimap was **completely static** and the CNN could never learn.

### Python (`python/minimap.py`, `rl/actor_critic_cnn.py`, `rl/rollout_buffer.py`,
`rl/config.py`, `rl/env_manager.py`, `rl/ppo.py`, `rl/async_trainer.py`,
`train.py`, `train_ui/evaluator.py`)

- `python/minimap.py`: `MinimapVecEnvWrapper` / `MinimapSingleEnvWrapper` —
  expose `minimap_obs()` returning `(n_envs, 8, 29, 29)` or `(8, 29, 29)`.
- `rl/actor_critic_cnn.py`: `ActorCriticCNN` — 3× Conv2d (8→32→64→64, 3×3,
  padding=1, ReLU) + AdaptiveAvgPool2d(1) + Flatten → MLP trunk (hidden sizes
  from config) → actor head (n_actions) + critic head (1).
- `rl/rollout_buffer.py`: `_TensorRolloutBuffer` — same GAE logic as base
  `RolloutBuffer`, but `obs` allocated with N-D shape `(total, C, H, W)`.
- `rl/config.py`: `obs_mode` (`"flat" | "minimap"`) + `minimap_radius` (default 14).
- `rl/env_manager.py`: selects `ActorCritic` (flat) vs `ActorCriticCNN` (minimap),
  builds the matching buffer, converts env obs → policy obs via `_policy_obs()`.
- `rl/ppo.py`: `save()` now stores `n_channels`/`grid_size` for CNN checkpoints
  (in addition to `obs_size` for MLP), so the evaluator can reconstruct the model.
- `rl/async_trainer.py`: passes `obs_mode` + `minimap_radius` to `run_eval()`.
- `train_ui/evaluator.py`: `_load_policy()` auto-detects CNN vs MLP from the
  state_dict (presence of `cnn.` keys) and reconstructs the correct model.
  `run_eval()` accepts `mode` + `minimap_radius`; for minimap it skips the
  flat-obs normalizer (CNN doesn't use it) and feeds the minimap tensor.
- `train.py`: `--obs-mode {flat,minimap}` CLI flag.

## Verification

### Unit / integration

- **102/102 tests pass** (`python -m pytest tests/ -q`), including the new
  minimap code paths and the existing flat MLP paths.
- Minimap shape/dtype verified from Python: `(8, 29, 29) float32`, per-env
  `(2, 8, 29, 29)` for vec env.
- Channel sums at reset (seed=7): `[443, 55, 108, 62, 29, 2, 9, 1]` —
  matches expected land distribution (суша dominant, oil/gold rare).
- After the `refresh_occupied()` fix, channel 7 grows with each build:
  `[1 → 7 → 11]` over 200 random actions. Confirmed the minimap is now dynamic.
- CNN forward pass on CUDA: actions `(2,)`, log_probs `(2,)`, values `(2,)`
  — shapes correct, ~88.5k params.

### Forward-pass throughput (isolated, batch=512, 50k steps)

| Mode    | FPS (steps/s) | Relative |
|---------|--------------:|---------:|
| flat MLP [256,256]  | 569,211 | 1.0× |
| minimap CNN [256,256] | 49,104 | 11.6× slower |

The CNN is ~11.6× slower per forward pass on this GPU. In the full training
loop (which includes env stepping, GAE, optimizer, etc.) the end-to-end
penalty is smaller.

### Training comparison (100k steps, 2 envs, n_steps=1024, batch=8192, 10 epochs)

| Metric | flat MLP | minimap CNN |
|--------|---------:|------------:|
| Avg FPS | **1,314** | 910 |
| Wall time | **76.4s** | 110.3s |
| Best in-ep reward | 2,385,279 | **2,753,361** |
| Eval @50k — days / people / bases / return / score | 1000 / 50 / **1** / −1798 / 413.0 (FAIL) | 1000 / 50 / 1 / −1798 / 413.0 (FAIL) |
| Eval @100k — days / people / bases / return / score | **731 / 50 / 23 / 359,383 / 407.3 (PASS)** | 365 / 33 / 14 / −814.6 / 194.6 (FAIL) |
| best_model.pt saved | **yes** | no |
| GPU mem (policy+buffer) | 68 MB | 119 MB |
| Final entropy | 1.82 | 3.32 |

### Interpretation

- **At 100k steps the flat MLP is clearly ahead on eval**: it reaches 23 bases
  and a positive return (359k), passing the `min_bases=5` threshold and saving
  `best_model.pt`. The minimap CNN reaches only 14 bases and a negative return.
- **The minimap CNN's in-episode best reward is higher** (2.75M vs 2.39M) and
  its entropy stays high (3.3 vs 1.8) — it is exploring more, but the eval
  policy (greedy argmax) is less consistent. The CNN needs more steps to
  converge to a good greedy policy.
- **The CNN is ~1.4× slower end-to-end** (1,314 vs 910 FPS) and uses ~1.7× more
  VRAM (68 vs 119 MB). The isolated forward-pass gap (11.6×) is largely hidden
  by env stepping / GAE / optimizer overhead in the full loop.
- **Root cause of the earlier "minimap never learns" problem:** the minimap
  was completely static because `refresh_occupied()` was never called in
  `step()`. Fixed by adding `g.refresh_occupied()` after the action.

### `best_model.pt` save threshold

`rl/async_trainer.py` saves `best_model.pt` only when:
1. `bases_agg >= eval_min_bases` (default 5) AND
2. `return_agg >= eval_min_return` (default 0) AND
3. composite `score > best_score` (so far).

The flat MLP passed all three at 100k. The minimap CNN failed #1 and #2 at
100k, so no `best_model.pt` was saved. This is **expected behaviour**, not a
bug — the CNN simply hasn't converged enough at this budget.

## Files changed / added

**Modified:**
- `include/colony/env.h` — minimap decls + `minimap_radius_` member
- `src/env.cpp` — `minimap()`, `minimap_batch()`, `refresh_occupied()` call in step
- `src/bindings.cpp` — numpy `minimap()` / `minimap_batch()`
- `rl/config.py` — `obs_mode`, `minimap_radius`
- `rl/rollout_buffer.py` — `_TensorRolloutBuffer`
- `rl/env_manager.py` — CNN path, `_policy_obs()`
- `rl/ppo.py` — CNN-aware `save()`
- `rl/async_trainer.py` — pass `obs_mode`/`minimap_radius` to eval
- `train.py` — `--obs-mode` flag
- `train_ui/evaluator.py` — CNN-aware `_load_policy()`, `run_eval(mode=...)`

**Added:**
- `python/minimap.py` — `MinimapVecEnvWrapper`, `MinimapSingleEnvWrapper`
- `rl/actor_critic_cnn.py` — `ActorCriticCNN`

## How to run

```bash
# flat MLP (baseline)
python train.py --steps 100000 --envs 2 --n-steps 1024 --obs-mode flat \
    --eval-freq 50000 --eval-episodes 3 --name flat_100k

# minimap CNN
python train.py --steps 100000 --envs 2 --n-steps 1024 --obs-mode minimap \
    --eval-freq 50000 --eval-episodes 3 --name minimap_100k
```

## 500k-step comparison

Both models were trained to 500k steps with the same hyperparameters
(2 envs, n_steps=1024, batch=8192, 10 epochs, lr=3e-4, entropy coeff=0.01).

### Eval trajectory (every 100k)

| Step | flat MLP score | flat status | minimap CNN score | minimap status |
|------|--------------:|:-----------:|------------------:|:--------------:|
| 100k | 202.0 | PASS | 202.0 | PASS |
| 200k | **484.0** | **PASS** | 202.0 | PASS |
| 301k | 463.9 | FAIL (bases=2) | — | — |
| 401k | 489.1 | FAIL (bases=4) | — | — |
| 501k | 336.2 | FAIL (bases=1) | 202.0 | PASS |

### Best model saved

| | flat 500k | minimap 500k |
|--|-----------|--------------|
| best_score | **483.95** | 201.99 |
| best_days | **1000** | 365 |
| best_bases | **10** | 11 |
| best_people | **50** | 33 |
| best_return | **439,509** | 163,885 |
| timesteps when saved | 200,704 | 501,760 |
| avg FPS | **1,352** | ~900 |
| wall time | **371s** | ~550s |
| final entropy | 1.34 | 0.75 |

### Interpretation

- **The flat MLP wins decisively at 500k steps.** It reaches a score of 484
  (1000 days, 10 bases, +439k return) by 200k steps and keeps it as its best.
  The minimap CNN's best-ever score is 202 (365 days, 11 bases, +164k return),
  and it only reaches that at the very last eval (500k steps) — meaning it took
  2.5× longer to converge to a worse result.
- **Both models show the same pattern of eval variance.** The flat MLP hits
  484 at 200k, drops to 336 at 500k, and the minimap CNN hovers around 202
  for the entire 500k run. With only 3 eval episodes and a stochastic env,
  individual evals are noisy — the `best_model.pt` threshold (bases≥5 AND
  return≥0) filters out the worst cases but can't eliminate the variance.
- **The minimap CNN's entropy collapses** (3.8 → 0.75 by 500k) while the flat
  MLP stabilizes around 1.3–1.9. The CNN is over-committing to a narrow policy
  without improving performance — a sign it's underfitting or that the 2D
  observation isn't giving it the signal it needs to find better strategies.
- **Conclusion:** for the current Sakhalin Colony task at ≤500k steps, the
  flat 203-dim observation + MLP is the better choice. The minimap CNN is
  slower, uses more VRAM, and underperforms on eval. The spatial information
  in the minimap doesn't seem to be the bottleneck — the policy can learn good
  colony management from global stats alone.

## Next steps / open questions

1. **Train the minimap CNN longer** (500k–1M steps) to see if it eventually
   beats the flat MLP on eval. The higher in-ep best reward suggests potential.
2. **Hybrid observation**: concatenate the flat 203-dim + minimap 8×29×29 and
   use a hybrid MLP+CNN trunk. This would let the policy use both global
   stats (money, taxes, credits) and spatial layout simultaneously.
3. **Larger CNN**: the current 3-conv trunk with 64 output channels may be too
   small for 45 actions. Try 128 channels or add a 4th conv layer.
4. **Minimap radius**: R=14 (29×29) covers ~28×28 cells around start. Try
   R=28 (57×57) to see if a larger field of view helps.
5. **torch.compile** the CNN path — the 11.6× forward-pass gap may close
   significantly with CUDA graphs.
6. **watch_champion.py** needs the minimap-aware `_load_policy` (currently
   only handles MLP) if you want to visually inspect a minimap-trained policy.
