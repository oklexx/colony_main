#!/usr/bin/env python3
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
am = {n:i for i,n in enumerate(env._action_names)}

# Build Road first, then other buildings
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road:          R={r['reward']:+.2f} (no diversity, no chain, no provider)")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (new):    R={r['reward']:+.2f} (with diversity +8.0)")

env.cpp_env.reset(42)
env.cpp_env.step(am['Farm'])
r = env.cpp_env.step(am['Farm'])
print(f"Farm (repeat): R={r['reward']:+.2f} (no diversity)")

# Build sequence: Road + buildings
env.cpp_env.reset(42)
total = 0
for name in ['WaterChannel','Farm','Road','Apiary','SmallHouse','Garden']:
    r = env.cpp_env.step(am[name])
    total += r['reward']
    print(f"  {name:16s}: R={r['reward']:+.2f} cumul={total:+.2f}")

env.close()
