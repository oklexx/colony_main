"""Diag: why do builds fail? Track money, build results, and compare checkpoints."""
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
names = None

def run_episode(model_file, max_steps=1500, seed=42, verbose=False):
    mp = MODEL_DIR / model_file
    npth = MODEL_DIR / (model_file.replace(".pt", ".norm.json"))
    policy = _load_policy(mp, torch.device("cpu"))
    env = CppColonyEnv(map_size=200)
    if npth.exists():
        env.normalizer.load(str(npth))
        env.normalizer.set_update(False)
    global names
    names = env._action_names
    n_build = 32
    obs, _ = info0 = env.reset(seed=seed)
    info0 = None
    money0 = None
    counts = Counter()
    fails = Counter()
    ok_builds = 0
    last = None
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
            logits, _ = policy(t)
        a = int(logits.argmax(-1).item())
        obs, rew, term, trunc, info = env.step(a)
        counts[names[a]] += 1
        is_build = 2 <= a < 2 + n_build
        if is_build:
            if rew < -2.0:  # error penalty
                fails[names[a]] += 1
            else:
                ok_builds += 1
                last = (step, names[a], rew, info.get("money"), info.get("bases"))
        if verbose and step % 200 == 0:
            print(f"  [{model_file}] step {step} money={info.get('money')} "
                  f"bases={info.get('bases')} pop={info.get('people')} day={info.get('days')}")
        if term or trunc:
            break
    print(f"\n=== {model_file} (steps={step}) ===")
    print(f"  last successful build: {last}")
    print("  action dist:", dict(counts.most_common(6)))
    print(f"  successful builds: {ok_builds}, failed: {sum(fails.values())}")
    if fails:
        print("  failed builds by type:", dict(fails.most_common(5)))
    print(f"  final: money={info.get('money')} bases={info.get('bases')} "
          f"pop={info.get('people')} day={info.get('days')}")
    env.close()
    return info

for mf in ["final_model.pt", "checkpoint_4194304_steps.pt"]:
    run_episode(mf, verbose=True)
