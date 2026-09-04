#!/usr/bin/env python3
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
am = {n:i for i,n in enumerate(env._action_names)}

# Build Road, then Farm, then Road again
env.cpp_env.reset(42)
r1 = env.cpp_env.step(am['Road'])
r2 = env.cpp_env.step(am['Farm'])
r3 = env.cpp_env.step(am['Road'])
print(f"Road (1st):  R={r1['reward']:+.2f}")
print(f"Farm (new):  R={r2['reward']:+.2f}")
print(f"Road (2nd):  R={r3['reward']:+.2f}")

# Compare: Farm, Road, Farm again
env.cpp_env.reset(42)
r1 = env.cpp_env.step(am['Farm'])
r2 = env.cpp_env.step(am['Road'])
r3 = env.cpp_env.step(am['Farm'])
print(f"\nFarm (1st):  R={r1['reward']:+.2f}")
print(f"Road:        R={r2['reward']:+.2f}")
print(f"Farm (2nd):  R={r3['reward']:+.2f}")

# Compare: 6 different buildings (no Road)
env.cpp_env.reset(42)
total = 0
for name in ['WaterChannel','Farm','Apiary','SmallHouse','Garden','CowFarm']:
    r = env.cpp_env.step(am[name])
    total += r['reward']
    print(f"  {name:16s}: R={r['reward']:+.2f} cumul={total:+.2f}")

env.close()
