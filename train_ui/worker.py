from __future__ import annotations
import builtins
import queue as _queue
#!/usr/bin/env python3
"""Worker process for train_ui: runs training or eval in a separate process.

Protocol: JSONL messages written to a file (--output). Commands on stdin.

Usage:
  python train_ui/worker.py --config <config.json> --name <run_name> --output <msg.jsonl>
  python train_ui/worker.py --eval-model <model.pt> --output <msg.jsonl>
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from train_ui import protocol as P


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sakhalin Colony train/eval worker")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=str, help="path to config JSON (train mode)")
    mode.add_argument("--eval-model", type=str, dest="eval_model", help="model path (eval mode)")
    p.add_argument("--name", type=str, default="", help="run name (train mode)")
    p.add_argument("--output", type=str, default="", help="path to JSONL message file")
    p.add_argument("--command-file", type=str, default="", dest="command_file",
                   help="path to JSONL command file (optional)")
    p.add_argument("--resume-model", type=str, default="", help="path to model .pt to load weights from (fine-tuning)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-days", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--normalization", type=str, default="", dest="normalization",
                   help="path to normalization.json (eval mode)")
    p.add_argument("--command-queue-size", type=int, default=100, help="max size of command queue")
    return p


def _read_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("config must be a JSON object")
    return d


class MsgFile:
    """Thread-safe writer that appends JSONL messages to a file."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._fh: Optional[TextIO] = None

    def open(self) -> None:
        self._fh = open(self._path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, msg: P.Msg) -> None:
        if self._fh is None:
            return
        line = P.encode(msg) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def write_raw(self, text: str) -> None:
        if self._fh is None:
            return
        with self._lock:
            self._fh.write(text)
            self._fh.flush()
            os.fsync(self._fh.fileno())


def _watch_stdin(stop_event: threading.Event, command_queue: Optional[_queue.Queue] = None) -> None:
    """Watch stdin for commands and either set stop event or queue the command."""
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                import logging
                logging.getLogger("Worker").warning(f"Invalid JSON: {e}")
                continue

            cmd = d.get("cmd")
            payload = d.get("payload", {})

            if command_queue is not None and cmd in ("pause", "resume", "boost_entropy", "reset_curriculum"):
                try:
                    command_queue.put(("command", {"cmd": cmd, "payload": payload}), timeout=1)
                except _queue.Full:
                    import logging
                    logging.getLogger("Worker").warning(f"Command queue full, dropping command: {cmd}")
                    continue
            elif cmd == "stop":
                stop_event.set()
                return
    except (ValueError, OSError) as e:
        import logging
        logging.getLogger("Worker").error(f"Stdin error: {e}")


def _watch_commands(command_file: str, stop_event: threading.Event,
                    command_queue: Optional[_queue.Queue] = None) -> None:
    """Watch a JSONL command file for commands from the UI."""
    offset = 0
    while not stop_event.is_set():
        try:
            size = os.path.getsize(command_file)
        except OSError:
            time.sleep(0.5)
            continue
        if size <= offset:
            time.sleep(0.5)
            continue
        try:
            with open(command_file, "r", encoding="utf-8") as f:
                f.seek(offset)
                data = f.read()
                offset = f.tell()
        except (OSError, UnicodeDecodeError):
            time.sleep(0.5)
            continue
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                cmd = d.get("cmd")
                payload = d.get("payload", {})
                if cmd == "stop":
                    stop_event.set()
                    return
                if command_queue is not None:
                    command_queue.put(("command", {"cmd": cmd, "payload": payload}), timeout=1)
            except (json.JSONDecodeError, _queue.Full, Exception):
                pass
        time.sleep(0.5)


def run_train(cfg_dict: Dict[str, Any], run_name: str, mf: MsgFile, stop_event: threading.Event,
              resume_model: str = "", command_queue: Optional[_queue.Queue] = None) -> int:
    # import queue already at top
    _orig_print = builtins.print

    def _print(*a, **kw):
        sep = kw.get("sep", " ")
        end = kw.get("end", "\n")
        text = sep.join(str(x) for x in a) + end
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                mf.write(P.LogMsg(level="info", message=line))
            except Exception:
                pass

    builtins.print = _print  # type: ignore[assignment]

    try:
        return _run_train_inner(cfg_dict, run_name, mf, stop_event, resume_model, command_queue)
    finally:
        builtins.print = _orig_print  # type: ignore[assignment]


def _run_train_inner(cfg_dict: Dict[str, Any], run_name: str, mf: MsgFile, stop_event: threading.Event,
                      resume_model: str = "", command_queue: Optional[_queue.Queue] = None) -> int:
    import torch
    from rl.config import Config, RewardConfig
    from rl.env_manager import EnvManager
    from rl.async_trainer import AsyncTrainer

    reward = RewardConfig()
    if isinstance(cfg_dict.get("reward"), dict):
        reward = RewardConfig.from_dict(cfg_dict["reward"])
    else:
        reward = RewardConfig.from_dict(cfg_dict)

    net_arch = cfg_dict.get("net_arch", [256, 256])
    if isinstance(net_arch, int):
        net_arch = [net_arch, net_arch]

    home = Path.home()
    model_dir = str(home / "colony_runs" / "models" / run_name)
    log_dir = str(home / "colony_runs" / "logs" / run_name)

    cfg = Config(
        map_size=int(cfg_dict.get("map_size", 280)),
        n_envs=int(cfg_dict.get("n_envs", 8)),
        seed=int(cfg_dict.get("seed", 42)),
        learning_rate=float(cfg_dict.get("learning_rate", 3e-4)),
        n_steps=int(cfg_dict.get("n_steps", 4096)),
        batch_size=int(cfg_dict.get("batch_size", 8192)),
        n_epochs=int(cfg_dict.get("n_epochs", 10)),
        gamma=float(cfg_dict.get("gamma", 0.995)),
        gae_lambda=float(cfg_dict.get("gae_lambda", 0.98)),
        clip_range=float(cfg_dict.get("clip_range", 0.2)),
        ent_coef=float(cfg_dict.get("ent_coef", 0.005)),
        vf_coef=float(cfg_dict.get("vf_coef", 0.5)),
        max_grad_norm=float(cfg_dict.get("max_grad_norm", 0.5)),
        net_arch=[int(x) for x in net_arch],
        total_timesteps=int(cfg_dict.get("total_timesteps", 1_000_000)),
        save_freq=int(cfg_dict.get("save_freq", 500_000)),
        eval_freq=int(cfg_dict.get("eval_freq", 100_000)),
        eval_episodes=int(cfg_dict.get("eval_episodes", 10)),
        device=str(cfg_dict.get("device", "cuda")),
        use_amp=bool(cfg_dict.get("use_amp", True)),
        amp_dtype=str(cfg_dict.get("amp_dtype", "bfloat16")),
        torch_compile=bool(cfg_dict.get("torch_compile", False)),
        cpp_threads=int(cfg_dict.get("cpp_threads", 0)),
        torch_threads=int(cfg_dict.get("torch_threads", 0)),
        async_train=bool(cfg_dict.get("async_train", False)),
        queue_size=int(cfg_dict.get("queue_size", 2)),
        curriculum_stage=int(cfg_dict.get("curriculum_stage", 0)),
        curriculum_schedule=cfg_dict.get("curriculum_schedule", []),
        log_dir=log_dir,
        model_dir=model_dir,
        reward=reward,
    )

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    def log(level: str, message: str) -> None:
        mf.write(P.LogMsg(level=level, message=message))

    def progress_cb(metrics, extra_data=None) -> None:
        try:
            mf.write(P.ProgressMsg(
                done=metrics.total_timesteps,
                total=cfg.total_timesteps,
                fps=float(getattr(metrics, 'fps', 0.0)),
                best_reward=float(getattr(metrics, 'best_reward', 0.0)) if getattr(metrics, 'best_reward', float("-inf")) != float("-inf") else 0.0,
                episodes=int(getattr(metrics, 'n_episodes', 0)),
                policy_loss=float(getattr(metrics, 'policy_loss', 0.0)),
                value_loss=float(getattr(metrics, 'value_loss', 0.0)),
                entropy=float(getattr(metrics, 'entropy', 0.0)),
                kl=float(getattr(metrics, 'approx_kl', 0.0)),
                ent_coef=float(getattr(metrics, 'ent_coef', cfg.ent_coef)),
                top_actions=dict(getattr(metrics, 'top_actions', {})),
                loop_detected=bool(getattr(metrics, 'loop_detected', False)),
                loop_action_name=getattr(metrics, 'loop_action_name', None),
                envs_with_loops=int(getattr(metrics, 'envs_with_loops', 0)),
                curriculum_stage_active=int(getattr(metrics, 'curriculum_stage', 0)),
                curriculum_next_at_step=getattr(metrics, 'curriculum_next_at_step', None),
                curriculum_stage=int(getattr(metrics, 'curriculum_stage', 0)),
                curriculum_progress_percent=float(getattr(metrics, 'curriculum_progress_percent', 0.0)),
                curriculum_available_actions=str(getattr(metrics, 'curriculum_available_actions', '')),
                curriculum_upcoming_stages=list(getattr(metrics, 'curriculum_upcoming_stages', [])),
                avg_return=float(getattr(metrics, 'avg_return', 0.0)),
                median_return=float(getattr(metrics, 'median_return', 0.0)),
                max_return=float(getattr(metrics, 'max_return', 0.0)),
                min_return=float(getattr(metrics, 'min_return', 0.0)),
                n_episodes_for_stats=int(getattr(metrics, 'n_episodes_for_stats', 0)),
            ))
        except Exception as e:
            try:
                mf.write(P.LogMsg(level="warn", message=f"[Worker] progress_cb error: {type(e).__name__}: {e}"))
            except Exception:
                pass

    log("info", f"[Worker] run={run_name} steps={cfg.total_timesteps:,} "
                f"envs={cfg.n_envs} device={device}")
    log("info", f"[Worker] model_dir={cfg.model_dir}")

    t0 = time.perf_counter()
    em = EnvManager(cfg, device)
    log("info", f"[Worker] env ready: obs={em.obs_size} actions={em.n_actions}")

    if resume_model and resume_model.strip():
        ckpt = torch.load(resume_model, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
        clean = {}
        for k, v in state.items():
            ck = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            clean[ck] = v
        em.model.load_state_dict(clean)
        log("info", f"[Worker] loaded weights from {resume_model}")

    trainer = AsyncTrainer(
        cfg=cfg,
        env_manager=em,
        logger=None,
        progress_callback=progress_cb,
        stop_check=lambda: stop_event.is_set(),
    )

    metrics = trainer.train(command_queue=command_queue)
    em.close()

    elapsed = time.perf_counter() - t0
    run_dir = Path(cfg.model_dir)
    final_path = run_dir / "final_model.pt"

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": int(metrics.total_timesteps),
        "best_reward": float(metrics.best_reward) if metrics.best_reward != float("-inf") else 0.0,
        "episodes": int(metrics.n_episodes),
        "train_time_sec": float(elapsed),
        "config": cfg.to_dict(),
    }
    meta_path = run_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("info", f"[Worker] meta saved: {meta_path}")

    mf.write(P.SavedMsg(path=str(final_path)))
    mf.write(P.DoneMsg(
        total=int(metrics.total_timesteps),
        time_s=float(elapsed),
        best_reward=float(metrics.best_reward) if metrics.best_reward != float("-inf") else 0.0,
        episodes=int(metrics.n_episodes),
    ))
    return 0


def run_eval(model_path: str, episodes: int, max_days: int, seed: int,
             device: str, mf: MsgFile, normalization_path: str = "") -> int:
    from train_ui.evaluator import run_eval as _run_eval

    def log(level: str, message: str) -> None:
        mf.write(P.LogMsg(level=level, message=message))

    norm = Path(normalization_path) if normalization_path else None
    log("info", f"[Eval] model={model_path} episodes={episodes} max_days={max_days}")

    log_path = Path("logs") / f"eval_{Path(model_path).stem}_{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result = _run_eval(model_path, episodes=episodes, max_days=max_days,
                       seed=seed, device=device, normalization_path=norm,
                       log_path=log_path)
    log("info", f"[Eval] days={result['days']:.1f} people={result['people']:.1f} "
                f"bases={result['bases']:.1f} avg_return={result['avg_return']:.2f}")
    log("info", f"[Eval] step log written to: {log_path}")
    mf.write_raw(json.dumps({"type": "eval_result", **result, "log_path": str(log_path)}, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    args = _build_parser().parse_args()

    mf = MsgFile(args.output) if args.output else None
    if mf:
        mf.open()

    stop_event = threading.Event()

    command_queue = _queue.Queue(maxsize=100)

    stdin_thread = threading.Thread(target=_watch_stdin, args=(stop_event, command_queue), daemon=True)
    stdin_thread.start()

    # Start file-based command watcher if command file is provided
    cmd_file = getattr(args, 'command_file', '')
    if cmd_file:
        cmd_thread = threading.Thread(
            target=_watch_commands, args=(cmd_file, stop_event, command_queue), daemon=True)
        cmd_thread.start()

    if mf:
        mf.write(P.ReadyMsg())

    rc = 1
    try:
        if args.config:
            cfg_dict = _read_config(Path(args.config))
            run_name = args.name or cfg_dict.get("name", "") or f"run_{int(time.time())}"
            rc = run_train(cfg_dict, run_name, mf, stop_event,
                           resume_model=args.resume_model,
                           command_queue=command_queue)
        else:
            rc = run_eval(args.eval_model, args.episodes, args.max_days,
                          args.seed, args.device, mf,
                          normalization_path=args.normalization)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if mf:
            try:
                mf.write(P.ErrorMsg(message=f"{type(e).__name__}: {e}\n{tb}"))
            except Exception:
                pass
        rc = 1
    finally:
        if mf:
            mf.close()

    return rc


if __name__ == "__main__":
    sys.exit(main())

