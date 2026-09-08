"""Reward sweep: 12 configurations, sequential runs, 1M steps each."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
WORKER = str(Path(__file__).parent / "train_ui" / "worker.py")
OUTPUT_DIR = Path(__file__).parent / "sweep_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Base config (from last successful run)
BASE = {
    "use_amp": True, "amp_dtype": "float16", "torch_compile": False,
    "obs_mode": "hybrid", "minimap_radius": 14,
    "total_timesteps": 1_000_000, "n_envs": 64, "n_steps": 4096,
    "batch_size": 8192, "n_epochs": 5,
    "learning_rate": 0.0003, "gamma": 0.99, "gae_lambda": 0.98,
    "clip_range": 0.2, "ent_coef": 0.05, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "net_arch": [256, 256], "map_size": 100, "curriculum_stage": 1,
    "curriculum_schedule": [],
    # baseline rewards
    "build_bonus": 16.0, "chain_bonus": 1.0, "chain_daily": 0.5,
    "novelty": 15.0, "daily_income": 1.0, "sale_bonus": 0.2,
    "tax_daily_bonus": 0.3, "survival_bonus": 0.0,
    "game_over_penalty": 10.0, "diversity_bonus": 8.0,
    "error_penalty": -1.0, "preserve_penalty": 0.0,
    "demolish_penalty": -3.0, "manual_tax_penalty": -0.5,
    "build_cost_penalty": 0.0001, "idle_build_penalty": -2.0,
    "idle_build_threshold_days": 3, "milestone_base_bonus": 30.0,
    "milestone_people_bonus": 2.0, "milestone_day_bonus": 2.0,
    "milestone_year_bonus": 5.0, "proximity_bonus": 0.5,
    "clip_reward_min": -50.0, "clip_reward_max": 50.0,
    "tax_fail_penalty": 5.0, "death_penalty": 5.0,
    "base_lost_penalty": 30.0, "born_bonus": 1.0,
    "debt_coeff": 0.02, "home_overflow_penalty": 2.0,
    "housing_need_bonus": 6.0, "food_need_bonus": 4.0, "water_need_bonus": 4.0,
}

# 12 reward configurations to test
CONFIGS = [
    # 0: Baseline (current)
    {"name": "R0_baseline", "desc": "Current best config"},
    # 1: Income Focus
    {"name": "R1_income", "desc": "High income, lower production",
     "daily_income": 5.0, "housing_need_bonus": 4.0, "food_need_bonus": 3.0, "water_need_bonus": 3.0},
    # 2: Low Penalties
    {"name": "R2_low_pen", "desc": "Reduce fear of loss",
     "death_penalty": 2.0, "base_lost_penalty": 10.0, "game_over_penalty": 5.0,
     "idle_build_penalty": -1.0, "daily_income": 2.0},
    # 3: High Build
    {"name": "R3_high_build", "desc": "Maximum build incentive",
     "build_bonus": 30.0, "daily_income": 2.0, "death_penalty": 3.0,
     "housing_need_bonus": 5.0, "food_need_bonus": 3.0, "water_need_bonus": 3.0},
    # 4: Production Focus
    {"name": "R4_production", "desc": "Food/housing/water production",
     "build_bonus": 10.0, "daily_income": 3.0, "death_penalty": 3.0,
     "housing_need_bonus": 10.0, "food_need_bonus": 8.0, "water_need_bonus": 8.0},
    # 5: Chain+Novelty
    {"name": "R5_chain", "desc": "Reward diversity",
     "build_bonus": 10.0, "chain_bonus": 5.0, "novelty": 25.0,
     "daily_income": 2.0, "death_penalty": 3.0},
    # 6: Balanced
    {"name": "R6_balanced", "desc": "Everything moderate",
     "build_bonus": 20.0, "daily_income": 3.0, "death_penalty": 3.0,
     "housing_need_bonus": 5.0, "food_need_bonus": 5.0, "water_need_bonus": 5.0},
    # 7: No Death Penalty
    {"name": "R7_no_death", "desc": "Remove death fear",
     "build_bonus": 20.0, "daily_income": 3.0, "death_penalty": 0.0,
     "base_lost_penalty": 5.0, "game_over_penalty": 2.0},
    # 8: High Milestones
    {"name": "R8_milestones", "desc": "Milestone chasing",
     "build_bonus": 10.0, "daily_income": 2.0, "death_penalty": 3.0,
     "milestone_base_bonus": 50.0, "milestone_people_bonus": 5.0},
    # 9: Aggressive Growth
    {"name": "R9_aggressive", "desc": "High everything positive",
     "build_bonus": 25.0, "daily_income": 4.0, "death_penalty": 2.0,
     "housing_need_bonus": 8.0, "food_need_bonus": 6.0, "water_need_bonus": 6.0},
    # 10: Minimal Penalties
    {"name": "R10_minimal", "desc": "Almost no penalties",
     "build_bonus": 20.0, "daily_income": 3.0, "death_penalty": 1.0,
     "base_lost_penalty": 5.0, "game_over_penalty": 2.0,
     "idle_build_penalty": 0.0, "error_penalty": 0.0},
    # 11: Income + Low Debt
    {"name": "R11_low_debt", "desc": "High income, low debt",
     "build_bonus": 15.0, "daily_income": 5.0, "death_penalty": 3.0,
     "debt_coeff": 0.005, "housing_need_bonus": 5.0,
     "food_need_bonus": 4.0, "water_need_bonus": 4.0},
]


def make_config(cfg_override: dict) -> dict:
    cfg = dict(BASE)
    cfg.update(cfg_override)
    return cfg


def run_one(cfg: dict, run_name: str) -> dict:
    """Run one training session, return parsed results."""
    import random
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
    print(f"START: {run_name} | seed={seed}")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    # Parse JSONL output for results
    result = {
        "name": run_name,
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
    }

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
                if msg.get("type") == "eval_result":
                    result["eval_days"] = msg.get("days", 0)
                    result["eval_people"] = msg.get("people", 0)
                    result["eval_bases"] = msg.get("bases", 0)
                    result["eval_return"] = msg.get("avg_return", msg.get("return", 0))
                    result["eval_score"] = msg.get("score", 0)

    # Fallback: parse stderr for key metrics
    if "best_reward" not in result and proc.stderr:
        for line in proc.stderr.splitlines():
            if "best_reward" in line:
                try:
                    parts = line.split("best_reward=")
                    if len(parts) > 1:
                        result["best_reward"] = float(parts[1].split("|")[0].strip())
                except (ValueError, IndexError):
                    pass

    print(f"DONE: {run_name} | {elapsed:.0f}s | best_reward={result.get('best_reward', '?')}")
    return result


def main():
    all_results = []
    for i, cfg_def in enumerate(CONFIGS):
        name = cfg_def["name"]
        desc = cfg_def["desc"]
        overrides = {k: v for k, v in cfg_def.items() if k not in ("name", "desc")}
        cfg = make_config(overrides)

        print(f"\n[{i+1}/{len(CONFIGS)}] {name}: {desc}")
        print(f"  Overrides: {overrides}")

        result = run_one(cfg, name)
        result["desc"] = desc
        all_results.append(result)

        # Save intermediate results
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"  Results so far:")
        for r in all_results:
            score = r.get("eval_score", "?")
            bases = r.get("eval_bases", "?")
            ret = r.get("eval_return", "?")
            br = r.get("best_reward", "?")
            print(f"    {r['name']}: score={score} bases={bases} return={ret} best_reward={br}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Name':<20} {'Score':>8} {'Bases':>6} {'Return':>10} {'BestR':>8} {'Time':>6}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<20} {r.get('eval_score','?'):>8} {r.get('eval_bases','?'):>6} "
              f"{r.get('eval_return','?'):>10} {r.get('best_reward','?'):>8} {r.get('elapsed_s','?'):>6}")

    # Find best
    valid = [r for r in all_results if r.get("eval_bases", 0) >= 5 and r.get("eval_return", 0) > 0]
    if valid:
        best = max(valid, key=lambda x: x.get("eval_score", 0))
        print(f"\nBEST: {best['name']} (score={best.get('eval_score')}, bases={best.get('eval_bases')}, return={best.get('eval_return')})")
    else:
        print("\nNo config met criteria (bases>5, return>0)")

    print(f"\nResults saved to: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
