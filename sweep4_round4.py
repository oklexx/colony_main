"""Round 4: Aggressive income to push return > 0."""
import json, subprocess, sys, time, random, re
from pathlib import Path

PYTHON = sys.executable
WORKER = str(Path(__file__).parent / "train_ui" / "worker.py")
OUTPUT_DIR = Path(__file__).parent / "sweep4_results"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE = {
    "use_amp": True, "amp_dtype": "float16", "torch_compile": False,
    "obs_mode": "hybrid", "minimap_radius": 14,
    "total_timesteps": 10_000_000, "n_envs": 64, "n_steps": 4096,
    "batch_size": 8192, "n_epochs": 5,
    "learning_rate": 0.0003, "gamma": 0.99, "gae_lambda": 0.98,
    "clip_range": 0.2, "ent_coef": 0.05, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "net_arch": [256, 256], "map_size": 100, "curriculum_stage": 1,
    "curriculum_schedule": [],
    "build_bonus": 25.0, "chain_bonus": 1.0, "chain_daily": 0.5,
    "novelty": 15.0, "daily_income": 6.0, "sale_bonus": 2.0,
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
    {"name": "R4_0_inc15", "desc": "income=15, debt=0.001",
     "daily_income": 15.0, "debt_coeff": 0.001},
    {"name": "R4_1_inc20", "desc": "income=20, debt=0.001, sale=5",
     "daily_income": 20.0, "sale_bonus": 5.0, "debt_coeff": 0.001},
    {"name": "R4_2_inc25", "desc": "income=25, sale=5, debt=0.001, death=1",
     "daily_income": 25.0, "sale_bonus": 5.0, "debt_coeff": 0.001, "death_penalty": 1.0},
    {"name": "R4_3_survival", "desc": "income=15, survival=10, debt=0.001",
     "daily_income": 15.0, "survival_bonus": 10.0, "debt_coeff": 0.001},
]

def make_config(overrides):
    cfg = dict(BASE); cfg.update(overrides); return cfg

def run_one(cfg, run_name):
    seed = random.randint(1, 999999999)
    cfg["seed"] = seed
    config_path = OUTPUT_DIR / f"{run_name}_config.json"
    log_path = OUTPUT_DIR / f"{run_name}_log.jsonl"
    with open(config_path, "w") as f: json.dump(cfg, f)
    cmd = [PYTHON, "-u", WORKER, "--config", str(config_path),
           "--name", run_name, "--output", str(log_path)]
    print(f"\n{'='*60}\nSTART: {run_name} | seed={seed} | 10M\n{'='*60}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.time() - t0
    result = {"name": run_name, "seed": seed, "elapsed_s": round(elapsed, 1)}
    evals = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: msg = json.loads(line)
                except: continue
                if msg.get("type") == "done":
                    result["best_reward"] = msg.get("best_reward", 0)
                    result["episodes"] = msg.get("episodes", 0)
                if msg.get("type") == "log" and "Eval @" in msg.get("message", ""):
                    evals.append(msg["message"])
    if evals:
        last = evals[-1]; result["last_eval_raw"] = last
        for pat, key in [(r'bases=(\d+\.?\d*)', "eval_bases"), (r'score=(\d+\.?\d*)', "eval_score"),
                         (r'return=(-?\d+\.?\d*)', "eval_return"), (r'people=(\d+\.?\d*)', "eval_people"),
                         (r'days=(\d+\.?\d*)', "eval_days")]:
            m = re.search(pat, last)
            if m: result[key] = float(m.group(1))
    print(f"DONE: {run_name} | {elapsed:.0f}s | best={result.get('best_reward','?')}")
    if evals: print(f"  {evals[-1]}")
    return result

def main():
    all_results = []
    for i, c in enumerate(CONFIGS):
        name, desc = c["name"], c["desc"]
        overrides = {k: v for k, v in c.items() if k not in ("name", "desc")}
        print(f"\n[{i+1}/{len(CONFIGS)}] {name}: {desc}")
        r = run_one(make_config(overrides), name); r["desc"] = desc
        all_results.append(r)
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}\nROUND 4 RESULTS\n{'='*60}")
    print(f"{'Name':<22} {'Score':>8} {'Bases':>6} {'Return':>10} {'People':>7}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<22} {r.get('eval_score','?'):>8} {r.get('eval_bases','?'):>6} "
              f"{r.get('eval_return','?'):>10} {r.get('eval_people','?'):>7}")
    valid = [r for r in all_results if r.get("eval_bases", 0) >= 5 and r.get("eval_return", 0) > 0]
    if valid:
        best = max(valid, key=lambda x: x.get("eval_score", 0))
        print(f"\nSUCCESS! {best['name']}: score={best.get('eval_score')}, return={best.get('eval_return')}")
    else:
        print("\nReturn still negative.")
        best = sorted(all_results, key=lambda x: x.get("eval_return", -99999), reverse=True)[0]
        print(f"Closest: {best['name']}: return={best.get('eval_return')} score={best.get('eval_score')}")

if __name__ == "__main__":
    main()
