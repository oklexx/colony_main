import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_ui2.models import ModelRegistry, ModelInfo


def _make_model(root: Path, name: str, meta: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "final_model.pt").write_bytes(b"fake")
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_scan_empty_dir(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.scan() == []


def test_scan_missing_dir(tmp_path):
    reg = ModelRegistry(tmp_path / "nope")
    assert reg.scan() == []


def test_scan_finds_final_model(tmp_path):
    _make_model(tmp_path, "run_a")
    models = ModelRegistry(tmp_path).scan()
    assert [m.name for m in models] == ["run_a"]
    assert models[0].model_file == tmp_path / "run_a" / "final_model.pt"


def test_scan_ignores_dir_without_model(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("hi")
    assert ModelRegistry(tmp_path).scan() == []


def test_scan_finds_checkpoint_only(tmp_path):
    d = tmp_path / "ckpt_run"
    d.mkdir()
    (d / "checkpoint_500000_steps.pt").write_bytes(b"fake")
    models = ModelRegistry(tmp_path).scan()
    assert [m.name for m in models] == ["ckpt_run"]


def test_scan_ignores_files(tmp_path):
    (tmp_path / "stray.pt").write_bytes(b"x")
    assert ModelRegistry(tmp_path).scan() == []


def test_meta_read(tmp_path):
    meta = {"created": "2026-08-28T10:00:00", "steps": 123,
            "best_reward": 4.5, "episodes": 7, "train_time_sec": 9.0}
    _make_model(tmp_path, "run_b", meta)
    m = ModelRegistry(tmp_path).scan()[0]
    assert m.steps == 123
    assert m.best_reward == 4.5
    assert m.episodes == 7
    assert m.train_time_sec == 9.0
    assert m.created == datetime(2026, 8, 28, 10, 0, 0)


def test_meta_bad_json_ignored(tmp_path):
    d = _make_model(tmp_path, "run_c")
    (d / "meta.json").write_text("{broken", encoding="utf-8")
    m = ModelRegistry(tmp_path).scan()[0]
    assert m.steps == 0


def test_save_meta_roundtrip(tmp_path):
    _make_model(tmp_path, "run_d")
    reg = ModelRegistry(tmp_path)
    m = reg.scan()[0]
    m.steps = 500
    m.best_reward = 1.5
    m.episodes = 3
    m.train_time_sec = 12.0
    m.created = datetime(2026, 1, 2, 3, 4, 5)
    p = reg.save_meta(m)
    assert p.exists()
    m2 = reg.get("run_d")
    assert m2.steps == 500
    assert m2.best_reward == 1.5
    assert m2.created == datetime(2026, 1, 2, 3, 4, 5)


def test_save_eval(tmp_path):
    _make_model(tmp_path, "run_e")
    reg = ModelRegistry(tmp_path)
    reg.save_eval("run_e", {"days": 100, "people": 50, "bases": 20})
    m = reg.get("run_e")
    assert m.eval == {"days": 100, "people": 50, "bases": 20}


def test_save_eval_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path).save_eval("nope", {"days": 1})


def test_delete(tmp_path):
    _make_model(tmp_path, "run_f")
    reg = ModelRegistry(tmp_path)
    reg.delete("run_f")
    assert not (tmp_path / "run_f").exists()


def test_delete_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path).delete("nope")


def test_delete_many(tmp_path):
    _make_model(tmp_path, "a")
    _make_model(tmp_path, "b")
    removed = ModelRegistry(tmp_path).delete_many(["a", "b"])
    assert removed == ["a", "b"]
    assert ModelRegistry(tmp_path).scan() == []


def test_get_missing(tmp_path):
    assert ModelRegistry(tmp_path).get("nope") is None


def test_default_models_dir():
    from train_ui2.models import default_models_dir
    p = default_models_dir()
    assert p.name == "models"
    assert "colony_runs" in str(p)
