from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_models_dir() -> Path:
    return Path.home() / "colony_runs" / "models"


@dataclass
class ModelInfo:
    name: str
    path: Path
    created: Optional[datetime] = None
    steps: int = 0
    best_reward: float = float("-inf")
    episodes: int = 0
    train_time_sec: float = 0.0
    eval: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_file(self) -> Path:
        return self.path / "final_model.pt"


def _parse_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _load_meta(path: Path) -> Dict[str, Any]:
    p = path / "meta.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _build_info(name: str, path: Path, meta: Dict[str, Any]) -> ModelInfo:
    created = _parse_dt(meta.get("created", ""))
    if created is None:
        try:
            created = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            created = None
    return ModelInfo(
        name=name,
        path=path,
        created=created,
        steps=int(meta.get("steps", 0) or 0),
        best_reward=float(meta.get("best_reward", float("-inf"))),
        episodes=int(meta.get("episodes", 0) or 0),
        train_time_sec=float(meta.get("train_time_sec", 0.0) or 0.0),
        eval=meta.get("eval", {}) if isinstance(meta.get("eval", {}), dict) else {},
        meta=meta,
    )


class ModelRegistry:
    """Scans a models directory for trained model folders."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_models_dir()

    def scan(self) -> List[ModelInfo]:
        if not self.root.exists():
            return []
        out: List[ModelInfo] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / "final_model.pt").exists() and not list(entry.glob("checkpoint_*.pt")):
                continue
            out.append(_build_info(entry.name, entry, _load_meta(entry)))
        out.sort(key=lambda m: (m.created is None, m.created or datetime.min), reverse=True)
        return out

    def get(self, name: str) -> Optional[ModelInfo]:
        p = self.root / name
        if not p.is_dir():
            return None
        return _build_info(name, p, _load_meta(p))

    def save_meta(self, info: ModelInfo) -> Path:
        meta = dict(info.meta)
        meta["steps"] = info.steps
        meta["best_reward"] = info.best_reward
        meta["episodes"] = info.episodes
        meta["train_time_sec"] = info.train_time_sec
        if info.eval:
            meta["eval"] = info.eval
        if info.created:
            meta["created"] = info.created.isoformat()
        p = info.path / "meta.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return p

    def save_eval(self, name: str, eval_result: Dict[str, Any]) -> Path:
        info = self.get(name)
        if info is None:
            raise FileNotFoundError(f"model not found: {name}")
        info.eval = eval_result
        return self.save_meta(info)

    def delete(self, name: str) -> None:
        p = self.root / name
        if not p.is_dir():
            raise FileNotFoundError(f"model not found: {name}")
        shutil.rmtree(p)

    def delete_many(self, names: List[str]) -> List[str]:
        removed = []
        for n in names:
            self.delete(n)
            removed.append(n)
        return removed
