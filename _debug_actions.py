"""Diag: what actions does the champion actually pick, and do builds succeed?"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from cpp_env import CppColonyEnv
from train_ui.evaluator import _load_policy

MODEL_DIR = Path.home() / "colony_runs" / "models" / "run"
model_path = MODEL_DIR / "final_model.pt"
norm_path = MODEL_DIR / "final_model.norm.json"

print("== model files ==")
for f in sorted(MODEL_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name:45s} {f.stat().st_size:>12,}")

print(f"\nLoading policy {model_path}")
policy = _load_policy(model_path, torch.device("cpu"))
print("policy type:", type(policy).__name__)

env = CppColonyEnv(map_size=200)
if norm_path.exists():
    env.normalizer.load(str(norm_path))
    env.normalizer.set_update(False)
    print(f"norm loaded: {norm_path}")
else:
    print("WARNING: no norm")

print("\n== env info ==")
print("n_actions:", env.cpp_env.n_actions())
print("action names:", env._action_names)

actor_w = None
for name, p in policy.named_parameters():
    if "actor" in name and p.dim() == 2:
        actor_w = (name, p.shape)
print("actor head:", actor_w)

names = env._action_names
n_build_actions = len(names) - 2 - 11  # total - day/week - managers
print(f"build actions: {n_build_actions} (indices 2..{1 + n_build_actions})")

obs, _ = env.reset(seed=42)
counts = Counter()
build_success = 0
build_fail = 0
total = 0
max_steps = 2000
for step in range(1, max_steps + 1):
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        logits, _ = policy(t)
    a = int(logits.argmax(-1).item())
    counts[names[a] if a < len(names) else str(a)] += 1
    obs, rew, term, trunc, info = env.step(a)
    total += 1
    if a >= 2 and a < 2 + n_build_actions:
        ok = info.get("action_name", "") and rew > -2.9
        if rew > -2.0:
            build_success += 1
        else:
            build_fail += 1
    if term or trunc:
        print(f"\nepisode ended at step {step}, day={info.get('days')}, "
              f"bases={info.get('bases')}, money={info.get('money')}")
        break

print(f"\n== action distribution over {total} steps ==")
for name, c in counts.most_common():
    print(f"  {name:18s} {c:5d}  {100.0 * c / total:5.1f}%")
print(f"build attempts success~{build_success} fail~{build_fail}")

env.close()
