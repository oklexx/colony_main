from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MsgType(str, Enum):
    READY = "ready"
    LOG = "log"
    PROGRESS = "progress"
    SAVED = "saved"
    DONE = "done"
    ERROR = "error"


@dataclass
class ReadyMsg:
    type: MsgType = field(default=MsgType.READY, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "ready"}


@dataclass
class LogMsg:
    level: str
    message: str
    type: MsgType = field(default=MsgType.LOG, init=False)

    def __post_init__(self):
        if self.level not in ("info", "warn", "error"):
            raise ValueError(f"invalid log level: {self.level!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "log", "level": self.level, "message": self.message}


def _safe_float(x: float, default: float = 0.0) -> float:
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return float(x)


@dataclass
class ProgressMsg:
    done: int
    total: int
    fps: float = 0.0
    best_reward: float = 0.0
    episodes: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl: float = 0.0
    type: MsgType = field(default=MsgType.PROGRESS, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "progress",
            "done": int(self.done),
            "total": int(self.total),
            "fps": _safe_float(self.fps),
            "best_reward": _safe_float(self.best_reward),
            "episodes": int(self.episodes),
            "policy_loss": _safe_float(self.policy_loss),
            "value_loss": _safe_float(self.value_loss),
            "entropy": _safe_float(self.entropy),
            "kl": _safe_float(self.kl),
        }


@dataclass
class SavedMsg:
    path: str
    type: MsgType = field(default=MsgType.SAVED, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "saved", "path": self.path}


@dataclass
class DoneMsg:
    total: int
    time_s: float
    best_reward: float
    episodes: int
    type: MsgType = field(default=MsgType.DONE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "done",
            "total": int(self.total),
            "time_s": _safe_float(self.time_s),
            "best_reward": _safe_float(self.best_reward),
            "episodes": int(self.episodes),
        }


@dataclass
class ErrorMsg:
    message: str
    type: MsgType = field(default=MsgType.ERROR, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "error", "message": self.message}


Msg = ReadyMsg | LogMsg | ProgressMsg | SavedMsg | DoneMsg | ErrorMsg

_REQUIRED: Dict[MsgType, tuple] = {
    MsgType.READY: (),
    MsgType.LOG: ("level", "message"),
    MsgType.PROGRESS: ("done", "total"),
    MsgType.SAVED: ("path",),
    MsgType.DONE: ("total", "time_s", "best_reward", "episodes"),
    MsgType.ERROR: ("message",),
}


def encode(msg: Msg) -> str:
    """Serialize a message to a single JSON line (no trailing newline)."""
    d = msg.to_dict()
    return json.dumps(d, ensure_ascii=False, allow_nan=False)


def decode(line: str) -> Msg:
    """Parse one JSON line into a typed message.

    Raises ValueError on malformed input or unknown type.
    """
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(d, dict):
        raise ValueError("message must be a JSON object")
    t = d.get("type")
    try:
        mt = MsgType(t)
    except ValueError as e:
        raise ValueError(f"unknown message type: {t!r}") from e
    missing = [k for k in _REQUIRED[mt] if k not in d]
    if missing:
        raise ValueError(f"missing fields for {mt.value}: {missing}")
    if mt is MsgType.READY:
        return ReadyMsg()
    if mt is MsgType.LOG:
        return LogMsg(level=d["level"], message=d["message"])
    if mt is MsgType.PROGRESS:
        return ProgressMsg(
            done=d["done"],
            total=d["total"],
            fps=d.get("fps", 0.0),
            best_reward=d.get("best_reward", float("-inf")),
            episodes=d.get("episodes", 0),
            policy_loss=d.get("policy_loss", 0.0),
            value_loss=d.get("value_loss", 0.0),
            entropy=d.get("entropy", 0.0),
            kl=d.get("kl", 0.0),
        )
    if mt is MsgType.SAVED:
        return SavedMsg(path=d["path"])
    if mt is MsgType.DONE:
        return DoneMsg(
            total=d["total"],
            time_s=d["time_s"],
            best_reward=d["best_reward"],
            episodes=d["episodes"],
        )
    return ErrorMsg(message=d["message"])


def encode_stop() -> str:
    return json.dumps({"cmd": "stop"}, ensure_ascii=False)
