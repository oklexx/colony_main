#!/usr/bin/env python3
"""Watch a trained champion model play the colony game in real time.

Usage:
  python watch_champion.py --model-dir ~/colony_runs/models/run_003
  python watch_champion.py --model-dir ~/colony_runs/models/run_003 --speed 5
  python watch_champion.py --model-dir ~/colony_runs/models/run_003 --episodes 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from cpp_env import CppColonyEnv
from train_ui.evaluator import _load_policy


class TeeWriter:
    """Write to both stdout and a file (for UI log capture)."""

    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description="Watch champion model play")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model directory (contains best_model.pt)")
    parser.add_argument("--model-file", type=str, default="best_model.pt",
                        help="Model filename (default: best_model.pt)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Steps per second (0=no delay, 1=1 step/sec)")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Also write output to this file (for UI capture)")
    args = parser.parse_args()

    if args.log_file:
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = TeeWriter(sys.__stdout__, log_f)
        import json as _json
        def emit_step(**kw):
            print(_json.dumps({"type": "step", **kw}, ensure_ascii=False))
        def emit_log(msg, level="info"):
            print(_json.dumps({"type": "log", "level": level, "message": msg}, ensure_ascii=False))
        def emit_done(msg):
            print(_json.dumps({"type": "done", "message": msg}, ensure_ascii=False))
    else:
        emit_step = None
        emit_log = None
        emit_done = None

    model_dir = Path(args.model_dir).expanduser()
    model_path = model_dir / args.model_file
    norm_path = model_dir / "normalization.json"
    if not norm_path.exists():
        norm_path = model_path.with_suffix(".norm.json")

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading policy from {model_path} on {dev}")
    policy = _load_policy(model_path, dev)

    print(f"Creating env (map_size={args.map_size})")
    env = CppColonyEnv(map_size=args.map_size)
    if norm_path.exists():
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
        print(f"Loaded normalization from {norm_path}")
    else:
        print("WARNING: no normalization found, using raw observations")

    action_names = env._action_names

    for ep in range(args.episodes):
        print(f"\n{'=' * 70}")
        print(f"EPISODE {ep + 1}/{args.episodes}")
        print(f"{'=' * 70}")

        obs, _info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        step = 0

        for step in range(1, args.max_steps + 1):
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev)
                obs_t = obs_t.reshape(1, -1)
                logits, _ = policy(obs_t)
                action = int(logits.argmax(dim=-1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            action_name = info.get("action_name", str(action))
            days = info.get("days", 0)
            people = info.get("people", 0)
            bases = info.get("bases", 0)
            money = info.get("money", 0)

            if emit_step:
                emit_step(
                    step=step, day=days, action=action_name,
                    reward=round(reward, 4), total_reward=round(total_reward, 2),
                    people=people, bases=bases, money=money,
                )
            else:
                print(
                    f"Step {step:5d} | Day {days:5d} | "
                    f"{action_name:<16s} | "
                    f"R={reward:+8.2f} | "
                    f"TotR={total_reward:+10.1f} | "
                    f"Pop={people:3d} | Bases={bases:2d} | "
                    f"Money={money:10d}"
                )

            if terminated or truncated:
                reason = "TERMINATED" if terminated else "TRUNCATED"
                msg = f"Episode {reason} at step {step}, day {days}, reward={total_reward:.1f}"
                if emit_log:
                    emit_log(msg)
                else:
                    print(f"\n>>> {msg}")
                break

            if args.speed > 0:
                time.sleep(1.0 / args.speed)

        else:
            msg = f"Max steps ({args.max_steps}) reached"
            if emit_log:
                emit_log(msg)
            else:
                print(f"\n>>> {msg}")

    env.close()
    final_msg = f"Done. {args.episodes} episode(s) completed."
    if emit_done:
        emit_done(final_msg)
    else:
        print(f"\n{final_msg}")


if __name__ == "__main__":
    main()
