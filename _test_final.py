#!/usr/bin/env python3
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
am = {n:i for i,n in enumerate(env._action_names)}

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (1st):  R={r['reward']:+.2f}")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (new):  R={r['reward']:+.2f}")

env.cpp_env.reset(42)
env.cpp_env.step(am['Farm'])
r = env.cpp_env.step(am['Road'])
print(f"Farm+Road:   Road R={r['reward']:+.2f}")

env.close()
