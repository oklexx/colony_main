import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False


def test_error_penalty_config():
    """Verify error_penalty field exists with correct default."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert hasattr(rc, "error_penalty")
    assert rc.error_penalty == -1.0

    d = rc.to_dict()
    assert d["error_penalty"] == -1.0
    rc2 = RewardConfig.from_dict(d)
    assert rc2.error_penalty == -1.0


def test_preserve_penalty_config():
    """Verify preserve_penalty field exists with correct default."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert hasattr(rc, "preserve_penalty")
    assert rc.preserve_penalty == 0.0


def test_demolish_penalty_config():
    """Verify demolish_penalty field exists with correct default."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert hasattr(rc, "demolish_penalty")
    assert rc.demolish_penalty == -3.0


def test_manual_tax_penalty_config():
    """Verify manual_tax_penalty field exists with correct default."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert hasattr(rc, "manual_tax_penalty")
    assert rc.manual_tax_penalty == -0.5


def test_error_penalty_cpp():
    """Verify error_penalty works in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.error_penalty == -1.0
    rc.error_penalty = -10.0
    assert rc.error_penalty == -10.0


def test_preserve_penalty_cpp():
    """Verify preserve_penalty works in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.preserve_penalty == 0.0
    rc.preserve_penalty = -2.0
    assert rc.preserve_penalty == -2.0


def test_demolish_penalty_cpp():
    """Verify demolish_penalty works in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.demolish_penalty == -3.0
    rc.demolish_penalty = -10.0
    assert rc.demolish_penalty == -10.0


def test_manual_tax_penalty_cpp():
    """Verify manual_tax_penalty works in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.manual_tax_penalty == -0.5
    rc.manual_tax_penalty = -1.5
    assert rc.manual_tax_penalty == -1.5


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_error_penalty_integration():
    """Run a short episode with custom error_penalty and verify env works."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.error_penalty = -99.0  # extreme value — if used, reward will be very negative

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(20):
            _, reward, done, _, _ = env.step(0)  # A_DAY
            rewards.append(reward)
            if done:
                break
        # Just verify env runs without crashing
        assert len(rewards) > 0
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_no_legacy_penalties():
    """Verify that legacy -0.1 penalties are gone."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.error_penalty = 0.0  # zero error penalty
    cfg.build_bonus = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.daily_income = 0.0
    cfg.sale_bonus = 0.0
    cfg.tax_daily_bonus = 0.0
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.preserve_penalty = 0.0
    cfg.manual_tax_penalty = 0.0

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        for _ in range(50):
            _, reward, done, _, _ = env.step(0)  # A_DAY
            # With everything zeroed, reward should only come from:
            # - net_worth change (0.005 * delta)
            # - people born/arrived (+1 each)
            # - death (-20), base_lost (-30), overflow (-2)
            # - credit penalty (-0.005 * credit / 1000)
            # None of these should produce exactly -0.1 or -1.0
            assert abs(reward - (-0.1)) > 0.01, \
                f"Legacy -0.1 penalty found: {reward}"
            assert abs(reward - (-1.0)) > 0.01, \
                f"Legacy -1.0 penalty found: {reward}"
            if done:
                break
    finally:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
