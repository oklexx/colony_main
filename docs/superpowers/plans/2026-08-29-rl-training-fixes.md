# RL Training System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the RL training evaluation system: per-env episode tracking, normalization save/load, in-training eval, and best-model saving.

**Architecture:** 4 independent tasks. Task 1 fixes the broken `_current_ep_return` accumulator in `async_trainer.py` by reading `info["episode"]` from the C++ vec env. Task 2 adds a `Normalizer` wrapper in `cpp_env.py` and integrates `save_normalization`/`load_normalization` into training and eval. Task 3 wires the existing `run_eval` into the training loop at `eval_freq` intervals with TensorBoard logging and best-model saving. Task 4 (optional) changes the tax reward from a one-shot +250 bonus to a daily +0.77 bonus.

**Tech Stack:** Python 3.11+, PyTorch 2.x, C++17 (pybind11), Gymnasium, Stable-Baselines3 VecEnv, TensorBoard.

**Spec:** This plan implements the decisions made in the pre-mortem review of the RL training system (see conversation history).

## Global Constraints

- Python: 3.11+ (uses `list[float]` type hints)
- PyTorch: 2.x with AMP (bfloat16/float16)
- C++: 17 standard, pybind11 bindings
- No new dependencies
- All existing tests must pass after each task
- Working directory: `C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main`

## File Map

| File | Task | Change |
|------|------|--------|
| `rl/async_trainer.py` | 1, 3 | Fix episode tracking, add eval + best-model save |
| `rl/env_manager.py` | 1 | Expose `infos` from `collect_step` |
| `python/cpp_env.py` | 2 | Add `Normalizer` class, integrate into `CppColonyEnv` |
| `train_ui/evaluator.py` | 2, 3 | Load normalization, accept policy object |
| `rl/ppo.py` | 2 | Add `save_normalization`/`load_normalization` |
| `train.py` | 2, 3 | Save/load normalization, pass writer to trainer |
| `tests/test_async_trainer.py` | 1 | New: test per-env episode tracking |
| `tests/test_normalizer.py` | 2 | New: test Normalizer class |
| `tests/test_evaluator.py` | 2, 3 | Update: test normalization loading |
| `src/env.cpp` | 4 | Change tax reward to daily, remove `tax_bonus` (optional) |
| `include/colony/env.h` | 4 | Replace `tax_bonus` with `tax_daily_bonus` (optional) |
| `rl/config.py` | 4 | Replace `tax_bonus` with `tax_daily_bonus` (optional) |
| `python/cpp_env.py` | 4 | Replace `tax_bonus` with `tax_daily_bonus` in `_REWARD_KEYS` (optional) |
| `python/cpp_vecenv.py` | 4 | Replace `tax_bonus` with `tax_daily_bonus` in `_REWARD_KEYS` (optional) |
| `src/bindings.cpp` | 4 | Replace `tax_bonus` with `tax_daily_bonus` (optional) |
| `configs/reward.json` | 4 | Replace `tax_bonus` with `tax_daily_bonus` (optional) |
| `tests/test_integration.py` | 3 | Integration smoke test |

## Task Files

- [Task 1: Per-env episode tracking](task-1-episode-tracking.md)
- [Task 2: Normalization save/load](task-2-normalization.md)
- [Task 3: In-training eval + best model](task-3-eval-best-model.md)
- [Task 4: Daily tax bonus (optional)](task-4-daily-tax.md)
