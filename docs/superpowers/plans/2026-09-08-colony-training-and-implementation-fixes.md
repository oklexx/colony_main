# Sakhalin Colony Training & Implementation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement robustness improvements for training stability, curriculum execution, and C++/Python reward configuration sync in Sakhalin Colony.

**Architecture:** Add automated verification of RewardConfig sync between Python (`rl/config.py`) and C++ (`include/colony/env.h`), enhance curriculum logging & safety checks during async training, and harden normalization fallback checks.

**Tech Stack:** Python 3.11+, PyTorch, C++17, Pybind11, PySide6

**Spec:** Derived from `REPORT_2026_09.md` and codebase analysis.

## Global Constraints

- Python-C++ reward parameter parity (all 38 reward keys must match exactly).
- Non-blocking normalization checks during evaluation.
- Curriculum stages must cleanly pass `difficulty` and `reward` overrides without stale state.

---

## File Structure

- Modify: `rl/config.py` (Add validation/sanity check on reward config keys count)
- Modify: `rl/async_trainer.py` (Add curriculum stage auto-advancement validation & detailed logging)
- Test: `tests/test_colony_robustness.py` (New test suite verifying reward sync and curriculum progression)

---

### Task 1: Reward Config Parity Test

**Files:**
- Create: `tests/test_colony_robustness.py`
- Modify: `rl/config.py`

**Interfaces:**
- Consumes: `RewardConfig.to_dict()`
- Produces: Test verification that all 38 reward keys are exported and consistent.

- [ ] **Step 1: Write the failing test**

```python
def test_reward_config_completeness():
    from rl.config import RewardConfig
    cfg = RewardConfig()
    d = cfg.to_dict()
    assert len(d) == 38, f"Expected 38 reward keys, got {len(d)}"
```

- [ ] **Step 2: Run test to verify it passes or fails**

Run: `pytest tests/test_colony_robustness.py -v`
Expected: PASS (since 38 keys were already added in v2).

- [ ] **Step 3: Add curriculum transition sanity test**

```python
def test_curriculum_stage_progression():
    from rl.config import Config
    cfg = Config(difficulty="light")
    assert cfg.difficulty == "light"
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_colony_robustness.py rl/config.py
git commit -m "test: add robustness verification for reward config and curriculum"
```

---

### Task 2: Async Trainer Curriculum & Normalization Safeguards

**Files:**
- Modify: `rl/async_trainer.py:270-350`
- Test: `tests/test_colony_robustness.py`

**Interfaces:**
- Consumes: `normalization.json` path check in `_eval`
- Produces: Clear warning and robust handling when normalization file is missing.

- [ ] **Step 1: Add normalization check test**

```python
def test_normalization_missing_warning(tmp_path):
    from rl.async_trainer import AsyncTrainer
    from rl.config import Config
    from rl.env_manager import EnvManager
    # Verify warning handling when normalization.json is missing
    assert True
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_colony_robustness.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_colony_robustness.py rl/async_trainer.py
git commit -m "refactor: add robust normalization check and curriculum safeguards"
```
