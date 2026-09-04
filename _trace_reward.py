#!/usr/bin/env python3
"""Trace Road reward component by component."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200, disable_net_worth=True)
am = {n:i for i,n in enumerate(env._action_names)}

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (no NW): R={r['reward']:+.2f}")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (no NW): R={r['reward']:+.2f}")

env.close()

# Now WITH net worth
env2 = CppColonyEnv(map_size=200)
am2 = {n:i for i,n in enumerate(env2._action_names)}

env2.cpp_env.reset(42)
r = env2.cpp_env.step(am2['Road'])
print(f"Road (NW on): R={r['reward']:+.2f}")

env2.cpp_env.reset(42)
r = env2.cpp_env.step(am2['Farm'])
print(f"Farm (NW on): R={r['reward']:+.2f}")

env2.close()
