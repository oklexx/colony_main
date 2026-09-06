#!/usr/bin/env python3
"""Automated training cycle: 30 runs x 10M steps with different parameters.
Evaluates each model, observes top performers, generates comprehensive report.
"""
import subprocess
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ============================================================
# 30 EXPERIMENTAL CONFIGURATIONS
# ============================================================
EXPERIMENTS = [
    # --- GROUP 1: Baseline + LR sweep (runs 01-06) ---
    {"id": "exp_01_baseline", "desc": "Baseline defaults",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_02_lr_low", "desc": "Low LR 1e-4",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 1e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_03_lr_high", "desc": "High LR 1e-3",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 1e-3, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_04_lr_very_low", "desc": "Very low LR 3e-5",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-5, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_05_lr_mid", "desc": "Mid LR 5e-4",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 5e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_06_lr_1e5", "desc": "LR 1e-5",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 1e-5, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},

    # --- GROUP 2: Entropy coefficient sweep (runs 07-11) ---
    {"id": "exp_07_ent_low", "desc": "Low entropy 0.001",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.001, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_08_ent_mid", "desc": "Mid entropy 0.005",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.005, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_09_ent_high", "desc": "High entropy 0.03",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.03, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_10_ent_very_high", "desc": "Very high entropy 0.05",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.05, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_11_ent_zero", "desc": "Zero entropy 0.0 (greedy)",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.0, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},

    # --- GROUP 3: Network architecture sweep (runs 12-16) ---
    {"id": "exp_12_net_small", "desc": "Small net [128,128]",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [128, 128], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_13_net_large", "desc": "Large net [512,512]",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [512, 512], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_14_net_xl", "desc": "XL net [512,512,256]",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [512, 512, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_15_net_deep", "desc": "Deep net [256,256,256,256]",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256, 256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_16_net_tiny", "desc": "Tiny net [64,64]",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [64, 64], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},

    # --- GROUP 4: Batch/rollout size sweep (runs 17-20) ---
    {"id": "exp_17_batch_small", "desc": "Small batch 2048, n-steps 2048",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 2048, "batch-size": 2048,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_18_batch_large", "desc": "Large batch 16384, n-steps 8192",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 8192, "batch-size": 16384,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_19_batch_xl", "desc": "XL batch 32768, n-steps 4096",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 32768,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_20_batch_micro", "desc": "Micro batch 1024, n-steps 1024",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 1024, "batch-size": 1024,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},

    # --- GROUP 5: Discount/GAE/Clip sweep (runs 21-24) ---
    {"id": "exp_21_gamma_low", "desc": "Low gamma 0.99",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.99, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_22_gamma_high", "desc": "High gamma 0.9999",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.9999, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_23_clip_low", "desc": "Low clip 0.1",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.1,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_24_clip_high", "desc": "High clip 0.3",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.3,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},

    # --- GROUP 6: Observation mode + minimap (runs 25-27) ---
    {"id": "exp_25_minimap", "desc": "Minimap CNN mode",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "minimap", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_26_hybrid", "desc": "Hybrid (flat+minimap)",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "hybrid", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_27_minimap_lr_low", "desc": "Minimap CNN + low LR",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 1e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "minimap", "seed": 42, "eval-episodes": 10}},

    # --- GROUP 7: Curriculum + VF coef + combined (runs 28-30) ---
    {"id": "exp_28_vf_coef_high", "desc": "High VF coef 1.0",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 1.0, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
    {"id": "exp_29_curriculum", "desc": "Curriculum stages",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 3e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.01, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [256, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10,
              "curriculum-schedule": "2000000:1,4000000:2,6000000:3"}},
    {"id": "exp_30_best_combo", "desc": "Best combo: LR 5e-4 + ent 0.005 + net [512,256]",
     "args": {"steps": 10_000_000, "envs": 32, "lr": 5e-4, "n-steps": 4096, "batch-size": 8192,
              "n-epochs": 10, "ent-coef": 0.005, "vf-coef": 0.5, "gamma": 0.995, "clip-range": 0.2,
              "net-arch": [512, 256], "obs-mode": "flat", "seed": 42, "eval-episodes": 10}},
]


def _estimate_params(net_arch: list, obs_mode: str) -> int:
    """Rough parameter count estimate for MLP/CNN policy."""
    obs_size = 209
    n_actions = 45
    total = 0
    prev = obs_size
    for h in net_arch:
        total += prev * h + h
        prev = h
    total += prev * n_actions + n_actions
    if obs_mode in ("minimap", "hybrid"):
        total += 8 * 3 * 3 * 32 + 32 * 3 * 3 * 64 + 64 * 7 * 7 * 128
    return total


def build_train_cmd(exp: dict, model_dir: str, log_dir: str) -> List[str]:
    """Build train.py CLI command from experiment config."""
    cmd = [PYTHON, str(PROJECT_ROOT / "train.py"), "--name", exp["id"]]
    args = exp["args"]
    for k, v in args.items():
        if isinstance(v, list):
            cmd.append(f"--{k}")
            cmd.extend(str(x) for x in v)
        elif isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    cmd.extend(["--model-dir", model_dir, "--log-dir", log_dir])
    cmd.append("--log-actions")
    return cmd


def run_experiment(exp: dict, model_dir: str, log_dir: str, results: dict) -> dict:
    """Run a single training experiment and return results."""
    exp_id = exp["id"]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"Description: {exp['desc']}")
    print(f"{'='*70}")

    t0 = time.time()
    cmd = build_train_cmd(exp, model_dir, log_dir)
    print(f"CMD: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,  # 2h timeout
            cwd=str(PROJECT_ROOT)
        )
        elapsed = time.time() - t0
        stdout = proc.stdout
        stderr = proc.stderr

        # Parse final metrics from stdout
        metrics = parse_train_output(stdout + stderr)

        # Check for best_model
        best_path = Path(model_dir) / exp_id / "best_model.pt"
        final_path = Path(model_dir) / exp_id / "final_model.pt"
        meta_path = Path(model_dir) / exp_id / "meta.json"

        result = {
            "id": exp_id,
            "desc": exp["desc"],
            "params": exp["args"],
            "elapsed_sec": elapsed,
            "returncode": proc.returncode,
            "best_model_exists": best_path.exists(),
            "final_model_exists": final_path.exists(),
            "metrics": metrics,
            "stdout_tail": stdout[-2000:] if stdout else "",
            "stderr_tail": stderr[-2000:] if stderr else "",
        }

        # Read meta.json if exists
        if meta_path.exists():
            with open(meta_path) as f:
                result["meta"] = json.load(f)

        # Read best_model.meta.json if exists
        best_meta_path = Path(model_dir) / exp_id / "best_model.meta.json"
        if best_meta_path.exists():
            with open(best_meta_path) as f:
                result["best_meta"] = json.load(f)

        results[exp_id] = result
        print(f"  DONE in {elapsed:.1f}s | returncode={proc.returncode}")
        if metrics:
            print(f"  Best reward: {metrics.get('best_reward', 'N/A')}")
            print(f"  Episodes: {metrics.get('n_episodes', 'N/A')}")

        return result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result = {
            "id": exp_id, "desc": exp["desc"], "params": exp["args"],
            "elapsed_sec": elapsed, "returncode": -1, "error": "TIMEOUT",
            "best_model_exists": False, "final_model_exists": False,
        }
        results[exp_id] = result
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "id": exp_id, "desc": exp["desc"], "params": exp["args"],
            "elapsed_sec": elapsed, "returncode": -2, "error": str(e),
            "best_model_exists": False, "final_model_exists": False,
        }
        results[exp_id] = result
        print(f"  ERROR: {e}")
        return result


def parse_train_output(output: str) -> dict:
    """Parse training output for key metrics."""
    metrics = {}
    lines = output.split("\n")
    for line in lines:
        if "best_reward=" in line:
            try:
                part = line.split("best_reward=")[1].split()[0]
                metrics["best_reward"] = float(part)
            except (IndexError, ValueError):
                pass
        if "Episodes completed:" in line:
            try:
                metrics["n_episodes"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Training complete:" in line:
            try:
                metrics["total_timesteps"] = int(line.split("complete:")[1].split("steps")[0].strip().replace(",", ""))
            except (IndexError, ValueError):
                pass
        if "Final entropy:" in line:
            try:
                metrics["final_entropy"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Final policy_loss:" in line:
            try:
                metrics["final_policy_loss"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Final value_loss:" in line:
            try:
                metrics["final_value_loss"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Final approx_kl:" in line:
            try:
                metrics["final_approx_kl"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Average FPS:" in line:
            try:
                metrics["avg_fps"] = float(line.split(":")[-1].strip().replace(",", ""))
            except ValueError:
                pass
        if "[Eval" in line and "days=" in line:
            try:
                import re
                days_m = re.search(r'days=(\d+\.?\d*)', line)
                bases_m = re.search(r'bases=(\d+\.?\d*)', line)
                score_m = re.search(r'score=(\d+\.?\d*)', line)
                if days_m: metrics["eval_days"] = float(days_m.group(1))
                if bases_m: metrics["eval_bases"] = float(bases_m.group(1))
                if score_m: metrics["eval_score"] = float(score_m.group(1))
            except Exception:
                pass
    return metrics


def evaluate_model(model_dir: str, exp_id: str, eval_dir: str) -> dict:
    """Run observe.py on the best model of an experiment."""
    model_path = Path(model_dir) / exp_id / "best_model.pt"
    if not model_path.exists():
        model_path = Path(model_dir) / exp_id / "final_model.pt"
    if not model_path.exists():
        return {"error": "no model found"}

    cmd = [
        PYTHON, str(PROJECT_ROOT / "observe.py"),
        "--model", str(model_path),
        "--episodes", "3",
        "--steps", "2000",
        "--log-dir", str(eval_dir),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              cwd=str(PROJECT_ROOT))
        # Parse output
        result = {"stdout": proc.stdout[-3000:], "stderr": proc.stderr[-1000:]}
        # Look for log files
        log_files = sorted(Path(eval_dir).glob("observe_*.log"), key=os.path.getmtime)
        if log_files:
            result["log_file"] = str(log_files[-1])
        return result
    except Exception as e:
        return {"error": str(e)}


def generate_report(results: dict, report_path: Path):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# Training Cycle Report - Sakhalin Colony")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Experiments:** {len(results)}")
    lines.append(f"**Steps per experiment:** 10,000,000")
    lines.append(f"**GPU:** NVIDIA RTX 5080 (16GB)")
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| # | Experiment | Desc | Score | Days | Bases | Return | Entropy | KL | FPS | Params | Time(s) |")
    lines.append("|---|-----------|------|-------|------|-------|--------|---------|-----|-----|--------|---------|")

    sorted_results = sorted(
        results.values(),
        key=lambda r: r.get("best_meta", {}).get("best_score", 0),
        reverse=True
    )

    for i, r in enumerate(sorted_results, 1):
        meta = r.get("best_meta", {})
        metrics = r.get("metrics", {})
        params = r.get("params", {})
        score = meta.get("best_score", "N/A")
        days = meta.get("best_days", "N/A")
        bases = meta.get("best_bases", "N/A")
        ret = meta.get("best_return", "N/A")
        ent = metrics.get("final_entropy", "N/A")
        kl = metrics.get("final_approx_kl", "N/A")
        fps = metrics.get("avg_fps", "N/A")
        elapsed = r.get("elapsed_sec", 0)

        score_s = f"{score:.1f}" if isinstance(score, (int, float)) else score
        days_s = f"{days:.0f}" if isinstance(days, (int, float)) else days
        bases_s = f"{bases:.1f}" if isinstance(bases, (int, float)) else bases
        ret_s = f"{ret:.1f}" if isinstance(ret, (int, float)) else ret
        ent_s = f"{ent:.4f}" if isinstance(ent, (int, float)) else ent
        kl_s = f"{kl:.5f}" if isinstance(kl, (int, float)) else kl
        fps_s = f"{fps:,.0f}" if isinstance(fps, (int, float)) else fps

        net_arch = params.get("net-arch", [256, 256])
        obs_mode = params.get("obs-mode", "flat")
        n_params = _estimate_params(net_arch, obs_mode)
        params_s = f"{n_params/1e6:.2f}M"

        lines.append(f"| {i} | {r['id']} | {r['desc']} | {score_s} | {days_s} | {bases_s} | {ret_s} | {ent_s} | {kl_s} | {fps_s} | {params_s} | {elapsed:.0f} |")

    lines.append("")

    # Parameter analysis by group
    groups = {
        "Learning Rate Sweep (exp_01-06)": ["exp_01_baseline", "exp_02_lr_low", "exp_03_lr_high",
                                             "exp_04_lr_very_low", "exp_05_lr_mid", "exp_06_lr_1e5"],
        "Entropy Coefficient Sweep (exp_07-11)": ["exp_07_ent_low", "exp_08_ent_mid", "exp_09_ent_high",
                                                   "exp_10_ent_very_high", "exp_11_ent_zero"],
        "Network Architecture Sweep (exp_12-16)": ["exp_12_net_small", "exp_13_net_large", "exp_14_net_xl",
                                                    "exp_15_net_deep", "exp_16_net_tiny"],
        "Batch/Rollout Size Sweep (exp_17-20)": ["exp_17_batch_small", "exp_18_batch_large",
                                                  "exp_19_batch_xl", "exp_20_batch_micro"],
        "Discount/Clip Sweep (exp_21-24)": ["exp_21_gamma_low", "exp_22_gamma_high",
                                             "exp_23_clip_low", "exp_24_clip_high"],
        "Observation Mode (exp_25-27)": ["exp_25_minimap", "exp_26_hybrid", "exp_27_minimap_lr_low"],
        "Combined/Special (exp_28-30)": ["exp_28_vf_coef_high", "exp_29_curriculum", "exp_30_best_combo"],
    }

    for group_name, exp_ids in groups.items():
        lines.append(f"## {group_name}")
        lines.append("")

        group_results = [results.get(eid) for eid in exp_ids if eid in results]
        if not group_results:
            lines.append("No results available.\n")
            continue

        lines.append("| Experiment | LR | Ent | Net Arch | Obs | Score | Days | Bases |")
        lines.append("|-----------|-----|-----|----------|-----|-------|------|-------|")

        for r in group_results:
            params = r.get("params", {})
            meta = r.get("best_meta", {})
            lr = params.get("lr", "N/A")
            ent = params.get("ent-coef", "N/A")
            arch = params.get("net-arch", "N/A")
            obs = params.get("obs-mode", "N/A")
            score = meta.get("best_score", "N/A")
            days = meta.get("best_days", "N/A")
            bases = meta.get("best_bases", "N/A")

            score_s = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            days_s = f"{days:.0f}" if isinstance(days, (int, float)) else str(days)
            bases_s = f"{bases:.1f}" if isinstance(bases, (int, float)) else str(bases)

            lines.append(f"| {r['id']} | {lr} | {ent} | {arch} | {obs} | {score_s} | {days_s} | {bases_s} |")

        lines.append("")

        # Group analysis
        lines.append(f"### Analysis: {group_name}")
        if group_results:
            best = max(group_results, key=lambda r: r.get("best_meta", {}).get("best_score", 0))
            worst = min(group_results, key=lambda r: r.get("best_meta", {}).get("best_score", 0))
            best_score = best.get("best_meta", {}).get("best_score", 0)
            worst_score = worst.get("best_meta", {}).get("best_score", 0)
            lines.append(f"- **Best:** {best['id']} (score={best_score:.1f})")
            lines.append(f"- **Worst:** {worst['id']} (score={worst_score:.1f})")
            if best_score > 0 and worst_score > 0:
                lines.append(f"- **Spread:** {best_score/worst_score:.2f}x difference")
        lines.append("")

    # Top 5 models detailed analysis
    lines.append("## Top 5 Models - Detailed Analysis")
    lines.append("")
    for i, r in enumerate(sorted_results[:5], 1):
        meta = r.get("best_meta", {})
        params = r.get("params", {})
        metrics = r.get("metrics", {})
        lines.append(f"### #{i}: {r['id']}")
        lines.append(f"- **Description:** {r['desc']}")
        lines.append(f"- **Score:** {meta.get('best_score', 'N/A')}")
        lines.append(f"- **Days survived:** {meta.get('best_days', 'N/A')}")
        lines.append(f"- **Buildings:** {meta.get('best_bases', 'N/A')}")
        lines.append(f"- **Population:** {meta.get('best_people', 'N/A')}")
        lines.append(f"- **Return:** {meta.get('best_return', 'N/A')}")
        lines.append(f"- **Parameters:** lr={params.get('lr')}, ent={params.get('ent-coef')}, "
                     f"net={params.get('net-arch')}, obs={params.get('obs-mode')}, "
                     f"gamma={params.get('gamma')}, clip={params.get('clip-range')}, "
                     f"batch={params.get('batch-size')}, n_steps={params.get('n-steps')}")
        lines.append(f"- **Training metrics:** entropy={metrics.get('final_entropy', 'N/A')}, "
                     f"KL={metrics.get('final_approx_kl', 'N/A')}, "
                     f"p_loss={metrics.get('final_policy_loss', 'N/A')}, "
                     f"v_loss={metrics.get('final_value_loss', 'N/A')}")
        lines.append(f"- **Training time:** {r.get('elapsed_sec', 0):.0f}s "
                     f"({r.get('elapsed_sec', 0)/60:.1f}min)")
        lines.append("")

    # Conclusions
    lines.append("## Conclusions & Recommendations")
    lines.append("")

    # Find best overall
    if sorted_results:
        best = sorted_results[0]
        best_meta = best.get("best_meta", {})
        best_params = best.get("params", {})
        lines.append(f"### Best Overall: {best['id']}")
        lines.append(f"- Score: {best_meta.get('best_score', 'N/A')}")
        lines.append(f"- Parameters: {json.dumps(best_params, indent=2)}")
        lines.append("")

    lines.append("### Key Findings")
    lines.append("")

    # Analyze LR group
    lr_exps = [results.get(eid) for eid in ["exp_01_baseline", "exp_02_lr_low", "exp_03_lr_high",
               "exp_04_lr_very_low", "exp_05_lr_mid", "exp_06_lr_1e5"] if eid in results]
    if lr_exps:
        best_lr = max(lr_exps, key=lambda r: r.get("best_meta", {}).get("best_score", 0))
        lr_val = best_lr.get("params", {}).get("lr", "N/A")
        lr_score = best_lr.get("best_meta", {}).get("best_score", 0)
        lines.append(f"1. **Optimal Learning Rate:** {lr_val} (score={lr_score:.1f})")

    # Analyze entropy group
    ent_exps = [results.get(eid) for eid in ["exp_07_ent_low", "exp_08_ent_mid", "exp_09_ent_high",
               "exp_10_ent_very_high", "exp_11_ent_zero"] if eid in results]
    if ent_exps:
        best_ent = max(ent_exps, key=lambda r: r.get("best_meta", {}).get("best_score", 0))
        ent_val = best_ent.get("params", {}).get("ent-coef", "N/A")
        ent_score = best_ent.get("best_meta", {}).get("best_score", 0)
        lines.append(f"2. **Optimal Entropy Coefficient:** {ent_val} (score={ent_score:.1f})")

    # Analyze net arch group
    net_exps = [results.get(eid) for eid in ["exp_12_net_small", "exp_13_net_large", "exp_14_net_xl",
               "exp_15_net_deep", "exp_16_net_tiny"] if eid in results]
    if net_exps:
        best_net = max(net_exps, key=lambda r: r.get("best_meta", {}).get("best_score", 0))
        net_val = best_net.get("params", {}).get("net-arch", "N/A")
        net_score = best_net.get("best_meta", {}).get("best_score", 0)
        lines.append(f"3. **Optimal Network Architecture:** {net_val} (score={net_score:.1f})")

    lines.append("")
    lines.append("### Agent Behavior Observations")
    lines.append("")
    lines.append("(To be filled based on observe.py logs and training action distributions)")
    lines.append("")

    # Write
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {report_path}")


def main():
    model_dir = str(Path.home() / "colony_runs" / "models")
    log_dir = str(Path.home() / "colony_runs" / "logs")
    eval_dir = str(PROJECT_ROOT / "obs_logs")
    results_path = PROJECT_ROOT / "training_cycle_results.json"
    report_path = PROJECT_ROOT / "TRAINING_CYCLE_REPORT.md"

    print(f"Model dir: {model_dir}")
    print(f"Log dir: {log_dir}")
    print(f"Eval dir: {eval_dir}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Estimated time: ~{len(EXPERIMENTS) * 7:.0f} min (~{len(EXPERIMENTS) * 7 / 60:.1f} hours)")

    results = {}

    # Load existing results if resuming
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    for i, exp in enumerate(EXPERIMENTS):
        if exp["id"] in results and results[exp["id"]].get("returncode") == 0:
            print(f"\nSkipping {exp['id']} (already completed)")
            continue

        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Running {exp['id']}...")
        result = run_experiment(exp, model_dir, log_dir, results)

        # Save intermediate results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    # Evaluate top models
    print(f"\n{'='*70}")
    print("EVALUATION PHASE - Running observe.py on top models")
    print(f"{'='*70}")

    sorted_results = sorted(
        results.values(),
        key=lambda r: r.get("best_meta", {}).get("best_score", 0),
        reverse=True
    )

    eval_results = {}
    for r in sorted_results[:5]:
        if r.get("best_model_exists") or r.get("final_model_exists"):
            exp_id = r["id"]
            print(f"\nEvaluating {exp_id}...")
            eval_result = evaluate_model(model_dir, exp_id, eval_dir)
            eval_results[exp_id] = eval_result
            print(f"  Eval done: {eval_result.get('log_file', 'no log')}")

    # Save eval results
    eval_path = PROJECT_ROOT / "training_cycle_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)

    # Generate report
    generate_report(results, report_path)

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"Results: {results_path}")
    print(f"Evaluation: {eval_path}")
    print(f"Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
