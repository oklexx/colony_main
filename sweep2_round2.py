"""Round 2: R9 variants on 5M steps - find profitable config."""
import json
import os
import subprocess
import sys
import time
import random
from pathlib import Path

PYTHON = sys.executable
WORKER = str(Path(__file__).parent / "train_ui" / "worker.py")
OUTPUT_DIR = Path(__file__).parent / "sweep2_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# R9 base config (the winner)
R9_BASE = {
    "use_amp": True, "amp_dtype": "float16", "torch_compile": False,
    "obs_mode": "hybrid", "minimap_radius": 14,
    "total_timesteps": 5_000_000, "n_envs": 64, "n_steps": 4096,
    "batch_size": 8192, "n_epochs": 5,
    "learning_rate": 0.0003, "gamma": 0.99, "gae_lambda": 0.98,
    "clip_range": 0.2, "ent_coef": 0.05, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "net_arch": [256, 256], "map_size": 100, "curriculum_stage": 1,
    "curriculum_schedule": [],
    # R9 aggressive rewards
    "build_bonus": 25.0, "chain_bonus": 1.0, "chain_daily": 0.5,
    "novelty": 15.0, "daily_income": 4.0, "sale_bonus": 0.2,
    "tax_daily_bonus": 0.3, "survival_bonus": 0.0,
    "game_over_penalty": 10.0, "diversity_bonus": 8.0,
    "error_penalty": -1.0, "preserve_penalty": 0.0,
    "demolish_penalty": -3.0, "manual_tax_penalty": -0.5,
    "build_cost_penalty": 0.0001, "idle_build_penalty": -2.0,
    "idle_build_threshold_days": 3, "milestone_base_bonus": 30.0,
    "milestone_people_bonus": 2.0, "milestone_day_bonus": 2.0,
    "milestone_year_bonus": 5.0, "proximity_bonus": 0.5,
    "clip_reward_min": -50.0, "clip_reward_max": 50.0,
    "tax_fail_penalty": 5.0, "death_penalty": 2.0,
    "base_lost_penalty": 30.0, "born_bonus": 1.0,
    "debt_coeff": 0.02, "home_overflow_penalty": 2.0,
    "housing_need_bonus": 8.0, "food_need_bonus": 6.0, "water_need_bonus": 6.0,
}

CONFIGS = [
    # 0: R9 exact copy on 5M
    {"name": "R2_0_R9x5M", "desc": "R9 exact on 5M steps"},
    # 1: R9 + higher income
    {"name": "R2_1_R9inc7", "desc": "R9 + income=7",
     "daily_income": 7.0},
    # 2: R9 + lower debt
    {"name": "R2_2_R9debt", "desc": "R9 + debt_coeff=0.005",
     "debt_coeff": 0.005, "daily_income": 5.0},
    # 3: R9 + max income + low penalties
    {"name": "R2_3_R9max", "desc": "R9 + income=10, death=1, debt=0.003",
     "daily_income": 10.0, "death_penalty": 1.0, "debt_coeff": 0.003},
    # 4: R9 + high sale bonus
    {"name": "R2_4_R9sale", "desc": "R9 + sale=2.0, income=6",
     "sale_bonus": 2.0, "daily_income": 6.0},
]


def make_config(overrides: dict) -> dict:
    cfg = dict(R9_BASE)
    cfg.update(overrides)
    return cfg


def run_one(cfg: dict, run_name: str) -> dict:
    seed = random.randint(1, 999999999)
    cfg["seed"] = seed

    config_path = OUTPUT_DIR / f"{run_name}_config.json"
    log_path = OUTPUT_DIR / f"{run_name}_log.jsonl"

    with open(config_path, "w") as f:
        json.dump(cfg, f)

    cmd = [
        PYTHON, "-u", WORKER,
        "--config", str(config_path),
        "--name", run_name,
        "--output", str(log_path),
    ]

    print(f"\n{'='*60}")
    print(f"START: {run_name} | seed={seed} | 5M steps")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    result = {
        "name": run_name, "seed": seed,
        "elapsed_s": round(elapsed, 1), "returncode": proc.returncode,
    }

    # Parse JSONL for eval results
    evals = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "done":
                    result["best_reward"] = msg.get("best_reward", 0)
                    result["episodes"] = msg.get("episodes", 0)
                if msg.get("type") == "log" and "Eval @" in msg.get("message", ""):
                    evals.append(msg["message"])

    # Parse last eval from log messages
    if evals:
        last_eval = evals[-1]
        result["last_eval_raw"] = last_eval
        # Extract bases, score, return
        try:
            parts = last_eval.split("]")
            for p in parts:
                p = p.strip()
                if p.startswith("days="):
                    import re
                    m = re.search(r'bases=(\d+\.?\d*)', p)
                    if m: result["eval_bases"] = float(m.group(1))
                    m = re.search(r'score=(\d+\.?\d*)', p)
                    if m: result["eval_score"] = float(m.group(1))
                    m = re.search(r'return=(-?\d+\.?\d*)', p)
                    if m: result["eval_return"] = float(m.group(1))
                    m = re.search(r'days=(\d+\.?\d*)', p)
                    if m: result["eval_days"] = float(m.group(1))
                    m = re.search(r'people=(\d+\.?\d*)', p)
                    if m: result["eval_people"] = float(m.group(1))
        except Exception:
            pass

    print(f"DONE: {run_name} | {elapsed:.0f}s | best_reward={result.get('best_reward', '?')}")
    if evals:
        print(f"  Last eval: {evals[-1]}")
    return result


def main():
    all_results = []
    for i, cfg_def in enumerate(CONFIGS):
        name = cfg_def["name"]
        desc = cfg_def["desc"]
        overrides = {k: v for k, v in cfg_def.items() if k not in ("name", "desc")}
        cfg = make_config(overrides)

        print(f"\n[{i+1}/{len(CONFIGS)}] {name}: {desc}")

        result = run_one(cfg, name)
        result["desc"] = desc
        all_results.append(result)

        # Save intermediate
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Print summary so far
        print(f"\n  Summary:")
        for r in all_results:
            score = r.get("eval_score", "?")
            bases = r.get("eval_bases", "?")
            ret = r.get("eval_return", "?")
            br = r.get("best_reward", "?")
            print(f"    {r['name']}: score={score} bases={bases} return={ret} best_reward={br}")

    # Final
    print(f"\n{'='*60}")
    print("ROUND 2 RESULTS")
    print(f"{'='*60}")
    print(f"{'Name':<22} {'Score':>8} {'Bases':>6} {'Return':>10} {'BestR':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<22} {r.get('eval_score','?'):>8} {r.get('eval_bases','?'):>6} "
              f"{r.get('eval_return','?'):>10} {r.get('best_reward','?'):>8}")

    # Find best
    valid = [r for r in all_results if r.get("eval_bases", 0) >= 5 and r.get("eval_return", 0) > 0]
    if valid:
        best = max(valid, key=lambda x: x.get("eval_score", 0))
        print(f"\nBEST: {best['name']} (score={best.get('eval_score')}, bases={best.get('eval_bases')}, return={best.get('eval_return')})")
    else:
        print("\nNo config met criteria (bases>5, return>0)")
        # Show best by score
        scored = [r for r in all_results if r.get("eval_score")]
        if scored:
            best = max(scored, key=lambda x: x.get("eval_score", 0))
            print(f"BEST SCORE: {best['name']} (score={best.get('eval_score')}, bases={best.get('eval_bases')}, return={best.get('eval_return')})")

    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
