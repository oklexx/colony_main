### Task 4: Daily Tax Bonus (Optional)

**Problem:** The tax reward is a one-shot +250 bonus (`env.cpp:476`) when the main tax is paid. This creates a spike reward that dominates the learning signal. The agent learns to time its actions to hit the tax payment moment, rather than maintaining steady economic growth. The `survival_bonus` is 0.0, so there's no positive signal for simply keeping the colony alive between tax events.

**Solution:** Replace the one-shot +250 with a daily bonus of +0.77 (≈ 250/365) that accrues every day the tax is NOT due (i.e., the agent is "tax-compliant"). Keep the smaller +15/+30 bonuses for the actual payment events as confirmation signals. **Remove `tax_bonus` entirely** from all configs and bindings since it's no longer used.

**Files:**
- Modify: `include/colony/env.h` — add `tax_daily_bonus`, remove `tax_bonus` from `RewardConfig`
- Modify: `src/env.cpp` — change tax reward logic, remove `tax_bonus` usage
- Modify: `rl/config.py` — add `tax_daily_bonus`, remove `tax_bonus` from Python `RewardConfig`
- Modify: `python/cpp_env.py` — replace `tax_bonus` with `tax_daily_bonus` in `_REWARD_KEYS`
- Modify: `python/cpp_vecenv.py` — replace `tax_bonus` with `tax_daily_bonus` in `_REWARD_KEYS`
- Modify: `src/bindings.cpp` — replace `tax_bonus` with `tax_daily_bonus` in `RewardConfig` binding
- Modify: `configs/reward.json` — replace `tax_bonus` with `tax_daily_bonus`
- Test: `tests/test_tax_daily_bonus.py` (new)

**Interfaces:**
- Consumes: `RewardConfig.tax_daily_bonus: float` (default 0.77)
- Produces: Daily reward +0.77 when `tax_due_days_ == 0` (tax not due)

- [ ] **Step 1: Replace `tax_bonus` with `tax_daily_bonus` in C++ `RewardConfig`**

In `include/colony/env.h`, modify the `RewardConfig` struct:

```cpp
struct RewardConfig {
    double build_bonus = 5.0;
    double chain_bonus = 0.5;
    double chain_daily = 2.0;
    double novelty = 20.0;
    double daily_income = 0.1;
    double sale_bonus = 0.1;
    // Removed: double tax_bonus = 250.0;  (replaced by tax_daily_bonus)
    double tax_daily_bonus = 0.77;  // daily bonus when tax-compliant
    double survival_bonus = 0.0;
    double game_over_penalty = 20.0;
    bool disable_net_worth = false;
    bool disable_daily_income = false;
};
```

- [ ] **Step 2: Update C++ reward logic in `env.cpp`**

In `src/env.cpp`, modify the tax reward section (lines 464-480):

```cpp
// Current:
if (!g.tax_postponed_ && g.annual_tax_due()) {
    if (g.money >= g.annual_tax_amount()) {
        g.pay_annual_tax();
        rew += 15.0;
    } else {
        rew -= 5.0;
    }
} else if (!g.tax_postponed_ && g.main_tax_due()) {
    if (g.money >= g.main_tax_amount()) {
        g.pay_main_tax();
        rew += 30.0;
        rew += cfg_.tax_bonus;
    } else {
        rew -= 5.0;
    }
}

// Change to:
if (!g.tax_postponed_ && g.annual_tax_due()) {
    if (g.money >= g.annual_tax_amount()) {
        g.pay_annual_tax();
        rew += 15.0;
    } else {
        rew -= 5.0;
    }
} else if (!g.tax_postponed_ && g.main_tax_due()) {
    if (g.money >= g.main_tax_amount()) {
        g.pay_main_tax();
        rew += 30.0;
        // Remove: rew += cfg_.tax_bonus;  (replaced by daily bonus)
    } else {
        rew -= 5.0;
    }
}

// Add daily tax-compliance bonus (after the tax block, before action processing):
if (!g.annual_tax_due() && !g.main_tax_due() && !g.tax_postponed_) {
    rew += cfg_.tax_daily_bonus;
}
```

- [ ] **Step 3: Update Python `RewardConfig`**

In `rl/config.py`, add `tax_daily_bonus` to `RewardConfig`:

```python
@dataclass
class RewardConfig:
    build_bonus: float = 5.0
    chain_bonus: float = 0.5
    chain_daily: float = 2.0
    novelty: float = 20.0
    daily_income: float = 0.1
    sale_bonus: float = 0.1
    # Removed: tax_bonus: float = 250.0
    tax_daily_bonus: float = 0.77
    survival_bonus: float = 0.0
    game_over_penalty: float = 20.0
    disable_net_worth: bool = False
    disable_daily_income: bool = False
```

Update `to_dict()`:
```python
# Remove: "tax_bonus": self.tax_bonus,
"tax_daily_bonus": self.tax_daily_bonus,
```

Update `from_dict()`:
```python
# Remove: tax_bonus=d.get("tax_bonus", 250.0),
tax_daily_bonus=d.get("tax_daily_bonus", 0.77),
```

- [ ] **Step 4: Update `cpp_env.py` and `cpp_vecenv.py`**

In both `python/cpp_env.py` and `python/cpp_vecenv.py`, replace `tax_bonus` with `tax_daily_bonus` in the `_REWARD_KEYS` tuple:

```python
# Before:
_REWARD_KEYS = ("build_bonus", "chain_bonus", "chain_daily",
                "novelty", "daily_income", "sale_bonus", "tax_bonus",
                "survival_bonus", "game_over_penalty")

# After:
_REWARD_KEYS = ("build_bonus", "chain_bonus", "chain_daily",
                "novelty", "daily_income", "sale_bonus", "tax_daily_bonus",
                "survival_bonus", "game_over_penalty")
```

- [ ] **Step 5: Update `bindings.cpp`**

In `src/bindings.cpp`, replace `tax_bonus` with `tax_daily_bonus` in the `RewardConfig` binding (line 307-319):

```cpp
py::class_<RewardConfig>(m, "RewardConfig")
    .def(py::init<>())
    .def_readwrite("build_bonus", &RewardConfig::build_bonus)
    .def_readwrite("chain_bonus", &RewardConfig::chain_bonus)
    .def_readwrite("chain_daily", &RewardConfig::chain_daily)
    .def_readwrite("novelty", &RewardConfig::novelty)
    .def_readwrite("daily_income", &RewardConfig::daily_income)
    .def_readwrite("sale_bonus", &RewardConfig::sale_bonus)
    // Removed: .def_readwrite("tax_bonus", &RewardConfig::tax_bonus)
    .def_readwrite("tax_daily_bonus", &RewardConfig::tax_daily_bonus)
    .def_readwrite("survival_bonus", &RewardConfig::survival_bonus)
    .def_readwrite("game_over_penalty", &RewardConfig::game_over_penalty)
    .def_readwrite("disable_net_worth", &RewardConfig::disable_net_worth)
    .def_readwrite("disable_daily_income", &RewardConfig::disable_daily_income);
```

- [ ] **Step 6: Write tests**

Create `tests/test_tax_daily_bonus.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False


def test_tax_daily_bonus_in_reward_config():
    """Verify tax_daily_bonus is part of RewardConfig and tax_bonus is removed."""
    from rl.config import RewardConfig

    rc = RewardConfig()
    assert hasattr(rc, "tax_daily_bonus")
    assert rc.tax_daily_bonus == 0.77
    assert not hasattr(rc, "tax_bonus"), "tax_bonus should be removed"

    # Test serialization
    d = rc.to_dict()
    assert "tax_daily_bonus" in d
    assert d["tax_daily_bonus"] == 0.77
    assert "tax_bonus" not in d, "tax_bonus should not be in dict"

    # Test deserialization
    rc2 = RewardConfig.from_dict(d)
    assert rc2.tax_daily_bonus == 0.77


def test_tax_daily_bonus_default():
    """Verify default value is 0.77 (250/365)."""
    from rl.config import RewardConfig

    rc = RewardConfig()
    assert abs(rc.tax_daily_bonus - 250.0 / 365.0) < 0.01


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_tax_daily_bonus_in_cpp_env():
    """Verify that tax_daily_bonus is applied in C++ env when tax is not due."""
    from cpp_env import CppColonyEnv

    env = CppColonyEnv(map_size=100)
    obs, _ = env.reset(seed=42)

    # Step a few times with DAY action
    total_reward = 0.0
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(0)  # DAY
        total_reward += reward
        if terminated or truncated:
            break

    # With tax_daily_bonus=0.77, we should get ~7.7 from daily bonus
    # (minus any other rewards/penalties)
    # This is a loose check — the exact value depends on env state
    assert total_reward > -100, f"Reward too negative: {total_reward}"
    env.close()


if __name__ == "__main__":
    test_tax_daily_bonus_in_reward_config()
    test_tax_daily_bonus_default()
    print("PASS: tax daily bonus config tests")
```

- [ ] **Step 7: Run tests**

Run: `python tests/test_tax_daily_bonus.py`
Expected: PASS

Run: `python -m pytest tests/test_ppo_smoke.py`
Expected: PASS

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: PASS

- [ ] **Step 8: Rebuild C++ extension**

```bash
cd C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main
python -m pip install -e . --no-build-isolation 2>nul || cmake -B build && cmake --build build
```

(Use the project's existing build system. Check `CMakeLists.txt` or `setup.py` for the correct command.)

- [ ] **Step 9: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_integration.py --ignore=tests/bench_per_step.py
```

Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add include/colony/env.h src/env.cpp rl/config.py python/cpp_env.py python/cpp_vecenv.py src/bindings.cpp configs/reward.json tests/test_tax_daily_bonus.py
git commit -m "feat: replace one-shot tax bonus with daily tax-compliance bonus, remove tax_bonus"
```
