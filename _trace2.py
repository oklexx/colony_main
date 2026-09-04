#!/usr/bin/env python3
"""Granular debug: build Road and show raw+clip."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
am = {n:i for i,n in enumerate(env._action_names)}

# Test 1: Road on fresh env
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road: R={r['reward']:+.2f}")

# Test 2: Farm on fresh env  
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm: R={r['reward']:+.2f}")

env.close()
