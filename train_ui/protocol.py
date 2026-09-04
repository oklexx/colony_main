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
    top_actions: Dict[str, float] = field(default_factory=dict)
    loop_detected: bool = False
    loop_action_name: Optional[str] = None
    envs_with_loops: int = 0
    curriculum_stage_active: int = 0
    curriculum_next_at_step: Optional[int] = None
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
            "top_actions": self.top_actions,
            "loop_detected": self.loop_detected,
            "loop_action_name": self.loop_action_name,
            "envs_with_loops": int(self.envs_with_loops),
            "curriculum_stage_active": int(self.curriculum_stage_active),
            "curriculum_next_at_step": int(self.curriculum_next_at_step) if self.curriculum_next_at_step else None,
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


class CommandType(str, Enum):
    START_TRAINING = "start"
    PAUSE_TRAINING = "pause"
    RESUME_TRAINING = "resume"
    STOP_TRAINING = "stop"
    BOOST_ENTROPY = "boost_entropy"
    RESET_CURRICULUM = "reset_curriculum"


@dataclass
class CommandMsg:
    cmd: str
    payload: Optional[Dict[str, Any]] = None
    type: MsgType = field(default=MsgType.LOG, init=False)  # Reuse LOG type for backward compatibility

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "command",
            "cmd": str(self.cmd),
            "payload": self.payload or {},
        }


def encode_command(cmd: str, payload: Optional[Dict[str, Any]] = None) -> str:
    """Encode a command to JSON."""
    msg = CommandMsg(cmd=cmd, payload=payload or {})
    return json.dumps(msg.to_dict(), ensure_ascii=False, allow_nan=False)


def decode_command(line: str) -> Dict[str, Any]:
    """Parse a command from JSON line.
    
    Returns dict with 'cmd' and 'payload' keys.
    """
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    
    if not isinstance(d, dict):
        raise ValueError("command must be a JSON object")
    
    t = d.get("type")
    if t != "command":
        raise ValueError(f"expected command type, got: {t}")
    
    cmd = d.get("cmd")
    if cmd is None:
        raise ValueError("missing 'cmd' field")
    
    payload = d.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("'payload' must be an object")
    
    return {"cmd": cmd, "payload": payload}


Msg = ReadyMsg | LogMsg | ProgressMsg | SavedMsg | DoneMsg | ErrorMsg | CommandMsg

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
    
    # Handle command messages separately
    if t == "command":
        return CommandMsg(
            cmd=d["cmd"],
            payload=d.get("payload", {}),
        )
    
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
            top_actions={k: _safe_float(v) for k, v in d.get("top_actions", {}).items()},
            loop_detected=d.get("loop_detected", False),
            loop_action_name=d.get("loop_action_name"),
            envs_with_loops=d.get("envs_with_loops", 0),
            curriculum_stage_active=d.get("curriculum_stage_active", 0),
            curriculum_next_at_step=d.get("curriculum_next_at_step"),
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
    """Encode stop command with optional final save flag."""
    payload = {"final_save": True}
    msg = CommandMsg(cmd="stop", payload=payload)
    return json.dumps(msg.to_dict(), ensure_ascii=False, allow_nan=False)
