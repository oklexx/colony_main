#!/usr/bin/env python3
"""Auto-tune PPO hyperparameters with Optuna.

Each trial runs a full training session via train.py, then reads
best_model.meta.json to get the composite score as the objective.

Usage:
  python auto_trainer.py --n-trials 10 --steps 1000000
  python auto_trainer.py --n-trials 5 --steps 500000 --study-name my_tuning
  python auto_trainer.py --n-trials 3 --steps 500000 --resume  # continue existing study
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def objective(trial: optuna.Trial, base_args: dict) -> float:
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [8192, 16384, 32768, 65536])
    ent_coef = trial.suggest_float("ent_coef", 0.005, 0.05, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    gamma = trial.suggest_float("gamma", 0.99, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    net_arch_choice = trial.suggest_categorical("net_arch", ["256x256", "512x512", "512x512x512"])
    net_arch = [int(x) for x in net_arch_choice.split("x")]

    trial_name = f"optuna_{trial.number}"
    model_dir = str(Path(base_args["model_dir"]) / trial_name)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--steps", str(base_args["steps"]),
        "--envs", str(base_args["envs"]),
        "--seed", str(base_args["seed"]),
        "--map-size", str(base_args["map_size"]),
        "--n-steps", str(n_steps),
        "--batch-size", str(batch_size),
        "--n-epochs", str(base_args["n_epochs"]),
        "--lr", str(lr),
        "--gamma", str(gamma),
        "--gae-lambda", str(base_args["gae_lambda"]),
        "--clip-range", str(clip_range),
        "--ent-coef", str(ent_coef),
        "--vf-coef", str(vf_coef),
        "--max-grad-norm", str(base_args["max_grad_norm"]),
        "--net-arch", *[str(x) for x in net_arch],
        "--device", base_args["device"],
        "--amp", base_args["amp"],
        "--save-freq", str(base_args["save_freq"]),
        "--eval-freq", str(base_args["eval_freq"]),
        "--eval-episodes", str(base_args["eval_episodes"]),
        "--name", trial_name,
        "--model-dir", base_args["model_dir"],
        "--log-dir", base_args["log_dir"],
    ]
    if base_args.get("reward_config"):
        cmd += ["--reward-config", base_args["reward_config"]]

    print(f"\n[Optuna Trial {trial.number}] params: lr={lr}, n_steps={n_steps}, "
          f"batch={batch_size}, ent={ent_coef}, vf={vf_coef}, gamma={gamma}, "
          f"clip={clip_range}, net={net_arch}")
    print(f"[Optuna Trial {trial.number}] model_dir={model_dir}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=base_args.get("timeout", 7200))
    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        print(f"[Optuna Trial {trial.number}] FAILED after {elapsed:.0f}s: {stderr_tail}")
        return -1e9

    meta_path = Path(model_dir) / "best_model.meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        score = meta.get("best_score", -1e9)
        print(f"[Optuna Trial {trial.number}] score={score:.2f} ({elapsed:.0f}s)")
        return score

    final_meta = Path(model_dir) / "meta.json"
    if final_meta.exists():
        with open(final_meta) as f:
            meta = json.load(f)
        score = meta.get("best_reward", -1e9)
        print(f"[Optuna Trial {trial.number}] no best_model, using best_reward={score:.2f}")
        return score

    print(f"[Optuna Trial {trial.number}] No meta found, returning -1e9")
    return -1e9


def main():
    parser = argparse.ArgumentParser(description="Auto-tune PPO with Optuna")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1_000_000, help="Steps per trial")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size", type=int, default=200)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=str, default="bfloat16")
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--reward-config", type=str, default="")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--study-name", type=str, default="colony_tuning")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per trial (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URL (default: in-memory)")
    args = parser.parse_args()

    if not args.model_dir:
        from pathlib import Path as P
        args.model_dir = str(P.home() / "colony_runs" / "models")
    if not args.log_dir:
        from pathlib import Path as P
        args.log_dir = str(P.home() / "colony_runs" / "logs")

    base_args = vars(args)
    base_args.pop("n_trials")
    base_args.pop("study_name")
    base_args.pop("resume")
    base_args.pop("storage")
    base_args.pop("timeout", None)
    base_args["timeout"] = args.timeout

    storage = args.storage if args.storage else None
    if args.resume and storage:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
    else:
        study = optuna.create_study(direction="maximize", study_name=args.study_name, storage=storage)

    print(f"{'=' * 60}")
    print(f"Auto-tuning: {args.n_trials} trials, {args.steps:,} steps each")
    print(f"Study: {args.study_name} (storage: {storage or 'in-memory'})")
    print(f"{'=' * 60}")

    t_start = time.time()
    study.optimize(
        lambda trial: objective(trial, base_args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    t_total = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Optimization complete in {t_total:.0f}s")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.2f}")
    print(f"Best params: {json.dumps(study.best_trial.params, indent=2)}")
    print(f"{'=' * 60}")

    best_model_dir = Path(args.model_dir) / f"optuna_{study.best_trial.number}"
    best_meta = best_model_dir / "best_model.meta.json"
    if best_meta.exists():
        print(f"Best model: {best_model_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
