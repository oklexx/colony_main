"""Round 6: Inverse - penalize overbuilding, reward population."""
import json, subprocess, sys, time, random, re
from pathlib import Path

PYTHON = sys.executable
WORKER = str(Path(__file__).parent / "train_ui" / "worker.py")
OUTPUT_DIR = Path(__file__).parent / "sweep6_results"
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
    "build_bonus": 5.0, "chain_bonus": 2.0, "chain_daily": 1.0,
    "novelty": 10.0, "daily_income": 20.0, "sale_bonus": 5.0,
    "tax_daily_bonus": 0.3, "survival_bonus": 0.0,
    "game_over_penalty": 10.0, "diversity_bonus": 4.0,
    "error_penalty": -1.0, "preserve_penalty": 0.0,
    "demolish_penalty": -1.0, "manual_tax_penalty": -0.5,
    "build_cost_penalty": 0.001, "idle_build_penalty": -2.0,
    "idle_build_threshold_days": 3, "milestone_base_bonus": 30.0,
    "milestone_people_bonus": 10.0, "milestone_day_bonus": 2.0,
    "milestone_year_bonus": 5.0, "proximity_bonus": 0.5,
    "clip_reward_min": -50.0, "clip_reward_max": 50.0,
    "tax_fail_penalty": 5.0, "death_penalty": 2.0,
    "base_lost_penalty": 30.0, "born_bonus": 20.0,
    "debt_coeff": 0.001, "home_overflow_penalty": 2.0,
    "housing_need_bonus": 15.0, "food_need_bonus": 10.0, "water_need_bonus": 10.0,
    "survival_coeff": 0.05,
}

CONFIGS = [
    {"name": "R6_0_pop_focus", "desc": "Low build, high pop, high income, surv_coeff=0.05"},
    {"name": "R6_1_s2_pop", "desc": "Same + stage=2",
     "curriculum_stage": 2, "born_bonus": 25.0, "milestone_people_bonus": 15.0},
    {"name": "R6_2_min_build", "desc": "build=2, build_cost=0.01, born=25",
     "build_bonus": 2.0, "build_cost_penalty": 0.01, "born_bonus": 25.0},
    {"name": "R6_3_inc30", "desc": "income=30, build=3, born=20, debt=0.0005",
     "daily_income": 30.0, "build_bonus": 3.0, "debt_coeff": 0.0005},
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
    print(f"\n{'='*60}\nSTART: {run_name} | seed={seed}\n{'='*60}")
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
    print(f"\n{'='*60}\nROUND 6 RESULTS\n{'='*60}")
    print(f"{'Name':<22} {'Score':>8} {'Bases':>6} {'Return':>10} {'People':>7}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<22} {r.get('eval_score','?'):>8} {r.get('eval_bases','?'):>6} "
              f"{r.get('eval_return','?'):>10} {r.get('eval_people','?'):>7}")
    valid = [r for r in all_results if r.get("eval_bases", 0) >= 5 and r.get("eval_return", 0) > 0]
    if valid:
        best = max(valid, key=lambda x: x.get("eval_score", 0))
        print(f"\nSUCCESS! {best['name']}")
    else:
        best = sorted(all_results, key=lambda x: x.get("eval_return", -99999), reverse=True)[0]
        print(f"\nClosest: {best['name']}: return={best.get('eval_return')} score={best.get('eval_score')} bases={best.get('eval_bases')} people={best.get('eval_people')}")

if __name__ == "__main__":
    main()
