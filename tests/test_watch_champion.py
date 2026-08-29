"""Tests for watch_champion.py --curriculum-stage handling."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def test_curriculum_stage_arg_accepted():
    """--curriculum-stage should be accepted by the argument parser."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "watch_champion.py"), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "--curriculum-stage" in result.stdout


def test_curriculum_stage_applied_to_env():
    """When --curriculum-stage is set, env.cpp_env.set_curriculum_stage is called."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()

    watch_champion.apply_curriculum_stage(mock_env, 2)

    mock_env.cpp_env.set_curriculum_stage.assert_called_once_with(2)


def test_curriculum_stage_none_skips():
    """When stage is None, set_curriculum_stage is NOT called."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()

    watch_champion.apply_curriculum_stage(mock_env, None)

    mock_env.cpp_env.set_curriculum_stage.assert_not_called()


def test_curriculum_stage_zero_disables():
    """When stage is 0, set_curriculum_stage(0) is called (all buildings)."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()

    watch_champion.apply_curriculum_stage(mock_env, 0)

    mock_env.cpp_env.set_curriculum_stage.assert_called_once_with(0)


def test_curriculum_stage_from_meta(tmp_path):
    """When stage is None and meta exists, read stage from meta."""
    import json
    import watch_champion
    importlib.reload(watch_champion)

    meta_path = tmp_path / "best_model.meta.json"
    meta_path.write_text(json.dumps({"curriculum_stage_at_best": 3}))

    stage = watch_champion.read_stage_from_meta(tmp_path)

    assert stage == 3


def test_curriculum_stage_from_meta_missing(tmp_path):
    """When meta does not exist, read_stage_from_meta returns None."""
    import watch_champion
    importlib.reload(watch_champion)

    stage = watch_champion.read_stage_from_meta(tmp_path)

    assert stage is None
