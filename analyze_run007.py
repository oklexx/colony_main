#!/usr/bin/env python3
"""Analyze run_007 training progression: reward normalization, observation stats."""
import json
from pathlib import Path

run_dir = Path.home() / "colony_runs" / "models" / "run_007"

# Load meta
with open(run_dir / "meta.json") as f:
    meta = json.load(f)

print("=" * 70)
print("RUN 007 ANALYSIS")
print("=" * 70)
print(f"Created:  {meta['created']}")
print(f"Steps:    {meta['steps']:,}")
print(f"Episodes: {meta['episodes']:,}")
print(f"Best reward: {meta['best_reward']:.4f}")
print(f"Train time:  {meta['train_time_sec']:.1f}s = {meta['train_time_sec']/3600:.1f}h")
print(f"FPS: {meta['steps']/meta['train_time_sec']:.0f}")
print()

cfg = meta["config"]
print("CONFIG:")
print(f"  map_size={cfg['map_size']}, n_envs={cfg['n_envs']}, seed={cfg['seed']}")
print(f"  lr={cfg['learning_rate']}, n_steps={cfg['n_steps']}, batch_size={cfg['batch_size']}")
print(f"  n_epochs={cfg['n_epochs']}, gamma={cfg['gamma']}, gae_lambda={cfg['gae_lambda']}")
print(f"  clip_range={cfg['clip_range']}, ent_coef={cfg['ent_coef']}, vf_coef={cfg['vf_coef']}")
print(f"  net_arch={cfg['net_arch']}, obs_mode={cfg['obs_mode']}")
print(f"  device={cfg['device']}, use_amp={cfg['use_amp']}, amp_dtype={cfg['amp_dtype']}")
print(f"  torch_compile={cfg['torch_compile']}")
print(f"  total_timesteps={cfg['total_timesteps']:,}, save_freq={cfg['save_freq']:,}")
print()

print("REWARD CONFIG:")
rc = cfg["reward"]
for k, v in sorted(rc.items()):
    print(f"  {k}: {v}")
print()

# Extract reward normalization progression
print("REWARD NORMALIZATION PROGRESSION:")
print(f"{'Steps':>12}  {'rew_mean':>10}  {'rew_std':>10}  {'count':>15}")
print("-" * 55)

checkpoints = sorted(run_dir.glob("checkpoint_*_steps.norm.json"))
for cp in checkpoints:
    step_str = cp.name.split("_")[1]
    steps = int(step_str)
    with open(cp) as f:
        data = json.load(f)
    rew = data.get("rew_rms", {})
    mean = rew.get("mean", [0])[0]
    var = rew.get("var", [1])[0]
    count = rew.get("count", 0)
    std = var ** 0.5
    print(f"{steps:>12,}  {mean:>10.4f}  {std:>10.4f}  {count:>15,.0f}")

# Final model
with open(run_dir / "final_model.norm.json") as f:
    final = json.load(f)
rew = final.get("rew_rms", {})
mean = rew.get("mean", [0])[0]
var = rew.get("var", [1])[0]
std = var ** 0.5
count = rew.get("count", 0)
print(f"{'FINAL':>12}  {mean:>10.4f}  {std:>10.4f}  {count:>15,.0f}")
print()

# Observation stats summary
print("OBSERVATION STATS (final):")
obs = final.get("obs_rms", {})
obs_mean = obs.get("mean", [])
obs_var = obs.get("var", [])
obs_count = obs.get("count", 0)
print(f"  count: {obs_count:,.0f}")
if obs_mean:
    import numpy as np
    m = np.array(obs_mean)
    v = np.array(obs_var)
    s = np.sqrt(v)
    print(f"  mean range: [{m.min():.6f}, {m.max():.6f}]")
    print(f"  std  range: [{s.min():.6f}, {s.max():.6f}]")
    print(f"  mean avg:   {m.mean():.6f}")
    print(f"  std  avg:   {s.mean():.6f}")
print()
print("Analysis complete.")
