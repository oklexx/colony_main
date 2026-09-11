from __future__ import annotations

import pytest
from pathlib import Path
from rl.config import RewardConfig, Config


def test_reward_config_completeness():
    cfg = RewardConfig()
    d = cfg.to_dict()
    assert len(d) == 42, f"Expected 42 reward keys, got {len(d)}"


def test_curriculum_stage_progression():
    cfg = Config(difficulty="light", curriculum_stage=1)
    assert cfg.difficulty == "light"
    assert cfg.curriculum_stage == 1


def test_normalization_path_handling(tmp_path):
    norm_path = tmp_path / "normalization.json"
    assert not norm_path.exists()
    # Simulating eval check
    norm_str = str(norm_path) if norm_path.exists() else None
    assert norm_str is None
