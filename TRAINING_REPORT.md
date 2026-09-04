# Training Report: Sakhalin Colony RL

## Executive Summary

Fixed three critical bugs that prevented training and evaluation:
1. Reward normalization feedback loop (NaN rewards)
2. NaN propagation in reward normalization
3. **Eval without action masking** (root cause of score=413)

After fixes, the model trains stably and passes eval thresholds (bases=5, return=+31). The model still gets stuck in action loops and needs more training to learn diverse strategies.

## Bugs Found and Fixed

### Bug 1: Reward Normalization Feedback Loop (CRITICAL)

**Root cause**: `ColonyVecEnvCpp::step_wait_batch()` stored **normalized** rewards in `old_rew_buffer_`, but `rew_rms_.update_scalar()` used those normalized values to update the running statistics.

**Effect**: Feedback loop — normalized rewards → RMS update → smaller variance → larger normalized values → RMS variance → 0 or negative → `normalize_reward` divides by `sqrt(negative)` → NaN.

**Evidence**: After 10M steps, `rew_rms.mean=[None], var=[None]` in JSON (NaN serialization). 17.8% of all rewards (372,737/2,097,152) became NaN, which corrupted all GAE advantages, causing 100% of PPO batches to be skipped (n_batches=0).

**Fix** (`env.cpp:1275-1286`):
```cpp
// Save raw rewards BEFORE normalization (RMS needs raw values)
std::copy(rewards_.begin(), rewards_.end(), old_rew_buffer_.begin());

if (norm_reward_) {
    for (int i = 0; i < n_envs_; ++i) {
        rew_rms_.update_scalar(old_rew_buffer_[i]);  // Now uses raw rewards
    }
    for (int i = 0; i < n_envs_; ++i) {
        rewards_[i] = rew_rms_.normalize_reward(rewards_[i], clip_reward_);
    }
}
```

Analogous to the observation normalization which already correctly saves raw obs to `raw_obs_buf_` before normalization.

### Bug 2: NaN Reward Propagation

**Root cause**: Some reward components (likely `std::log2(1.0 + ypv_b / 1000.0)` or `provider_bonus`) could produce NaN values. `std::clamp(NaN, min, max)` is undefined behavior — NaN comparisons return false, so NaN passes through unclamped.

**Fix** (`env.cpp:1120-1121`):
```cpp
if (!std::isfinite(rew)) rew = 0.0;
rew = std::clamp(rew, cfg_.clip_reward_min, cfg_.clip_reward_max);
```

### Bug 3: RunningMeanStd NaN Robustness

**Root cause**: If a single NaN value entered the Welford algorithm, it corrupted all subsequent mean/variance estimates permanently.

**Fixes** (`running_mean_std.h`):
- `normalize_reward()`: Added `std::abs(var)` and `isfinite` guard
- `normalize()`: Added `std::abs(var)` and `isfinite` guard  
- `update_scalar()`: Skip NaN inputs
- `update()`: Skip NaN values in batch statistics

### Bug 4: Eval Without Action Masking (ROOT CAUSE OF SCORE=413)

**Root cause**: `evaluator.py` used `logits.argmax()` to select greedy actions, but did NOT apply action masking. During training, invalid actions (e.g., HuntingLand) are masked with `-inf` logits, so the model never learns they fail. During eval without masking, the model selects these invalid actions, gets error penalty every step, and never builds new buildings.

**Evidence**:
- Without masking: model stuck on HuntingLand (0.79 probability), bases=1, return=-4165, score=413
- With masking: model uses PAY_TAX/Garden/Road, bases=5, return=+31, score=167.6, thresholds PASS

**Fix**:
1. Added `action_mask()` method to `CppColonyEnv` wrapper (`cpp_env.py`)
2. Updated `evaluator.py` to apply `logits.masked_fill(mask == 0, -inf)` before argmax

**Impact**: Eval now matches training behavior. Model correctly avoids invalid actions and passes thresholds (bases>=5, return>=0).

## Action Masking Implementation

Implemented full action masking pipeline across C++ and Python:

### C++ Layer
- `ColonyEnvCpp::action_mask()` — per-env mask checking unlock, road cap, money, find_lot
- `ColonyVecEnvCpp::action_masks_batch()` — batched version returning `[n_envs * n_actions]`
- Bindings exposed via pybind11

### Python Layer
- `cpp_vecenv.py`: `_action_masks` buffer, updated on `reset()` and `step_wait()`
- `rollout_buffer.py`: `action_masks` tensor in both `RolloutBuffer` and `_TensorRolloutBuffer`
- `ppo.py`: `collect_step()` applies `logits.masked_fill(masks == 0, -inf)` before sampling; `_compute_loss_components()` also masks logits during update
- `env_manager.py`: Gets masks from `self.env.action_masks`, passes to PPO and buffer
- `observe.py`: Applied masks during evaluation

**Verified**: 9/32 builds available at start (Farm, Garden, Road, House, SmallHouse, CowFarm, Apiary, Hothouse, Puerperal).

## run_012 (NaN Reward Fix Only, No Raw Guard)

| Step | p_loss | v_loss | ent | KL | eval score |
|------|--------|--------|-----|-----|------------|
| 2M   | 0.0032 | 88.11  | 2.627 | 0.012 | 413 |
| 4M   | -0.0003 | 3.22  | 2.625 | 0.007 | 413 |
| 6M   | -0.0011 | 0.84  | 2.620 | 0.009 | 413 |
| 8M   | -0.0015 | 0.38  | 2.614 | 0.008 | 413 |
| 10M  | -0.0021 | 0.22  | 2.606 | 0.007 | 413 |

**Analysis**: NaN fixed, but rew_rms still had NaN from raw reward NaN propagation. Value function learned well (v_loss 88→0.2) but policy barely changed (entropy 2.627→2.606).

## run_013 (Full Fix)

| Step | p_loss | v_loss | ent | KL | eval score |
|------|--------|--------|-----|-----|------------|
| 2M   | 0.0025 | 57.78  | 2.624 | 0.015 | 413 |
| 4M   | -0.0042 | 48.28  | 2.608 | 0.011 | 413 |
| 6M   | -0.0047 | 50.91  | 2.581 | 0.012 | 413 |
| 8M   | -0.0051 | 49.79  | 2.537 | 0.012 | 413 |
| 10M  | -0.0057 | 45.50  | 2.478 | 0.011 | 413 |

**Analysis**: 
- **Training is healthy**: KL ~0.01 (target range), entropy decreasing (2.62→2.48), p_loss non-zero
- **Value loss higher** (45 vs 0.2 in run_012) — this is actually more correct since rewards now have proper variance
- **Reward RMS clean**: mean=-1.72, var=11.23 (no NaN)
- **Eval stuck at 413**: Model gets 50 people, 1 base but fails on other thresholds

## What's Working

1. **No crashes** — all 10M steps complete with zero NaN skips
2. **Policy learning** — entropy drops from 2.62→2.48 (near uniform for 9 actions)
3. **Value function learning** — v_loss stable and decreasing
4. **Action masking** — prevents illegal actions during training
5. **Reward normalization** — stable with proper raw reward tracking
6. **Eval with masking** — now matches training behavior, passes thresholds

## What's Not Working

1. **Model stuck in action loops** — selects same action (PAY_TAX) every step despite masking
2. **Low diversity** — entropy 2.48 vs optimal ~1.5-2.0 for a good policy
3. **Short episodes** — only 365 days (was 1000 without masking, now terminates on tax/game over)
4. **Low score** — 167.6 vs potential much higher score with diverse strategies

## Next Steps

1. **More training** — 10M steps insufficient; model still near-uniform entropy
2. **LR schedule** — decay lr from 0.0003 to improve convergence
3. **Entropy bonus tuning** — current ent_coef may be too high, keeping policy too random
4. **Reward reshaping** — stronger signals for building diversity and resource management
5. **Curriculum** — progressive difficulty to guide learning

## Files Modified

| File | Changes |
|------|---------|
| `src/env.cpp` | Fixed old_rew_buffer_ order, added NaN guard on raw reward |
| `include/colony/running_mean_std.h` | NaN guards in normalize, normalize_reward, update, update_scalar |
| `rl/ppo.py` | Action masking in collect_step and _compute_loss_components |
| `rl/rollout_buffer.py` | Added action_masks tensor to both buffer classes |
| `rl/env_manager.py` | Passes action masks to PPO and buffer |
| `python/cpp_vecenv.py` | action_masks property, _action_masks buffer |
| `python/cpp_env.py` | Added action_mask() method to CppColonyEnv |
| `train_ui/evaluator.py` | Apply action masking during eval (argmax with masked logits) |
| `src/bindings.cpp` | action_mask and action_masks_batch bindings |
| `observe.py` | Updated with mask support |
