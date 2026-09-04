"""Diag: why do Sawmill/CoalCut builds fail? Check lot types around the colony."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
obs, info = env.reset(seed=42)

# What does the env expose?
print("env attrs:", [a for a in dir(env.cpp_env) if not a.startswith('_')])
print()
print("initial info:", info)
print()

# Try building each building type and see the result
names = env._action_names
n_build = 32
print("== trying each build action once from initial state ==")
for i in range(n_build):
    a = 2 + i
    name = names[a]
    obs, rew, term, trunc, inf = env.step(a)
    status = "OK" if rew > -2.0 else "FAIL"
    print(f"  {status} {name:18s} rew={rew:+7.2f} money={inf.get('money')} bases={inf.get('bases')}")
    if term or trunc:
        break
    # reset back to initial state for next attempt
    obs, info = env.reset(seed=42)
