#!/usr/bin/env python3
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
am = {n:i for i,n in enumerate(env._action_names)}

# Test: Road on fresh env
env.cpp_env.reset(42)
print("Before Road:")
print(f"  obs[24] (money) = {env.cpp_env.obs()[24]}")
print(f"  obs[25] (bases) = {env.cpp_env.obs()[25]}")
r = env.cpp_env.step(am['Road'])
print(f"  Road R={r['reward']:+.2f}")
print(f"  After: money={r['money']} bases={r['bases']}")

# Test: Road after WaterChannel
env.cpp_env.reset(42)
env.cpp_env.step(am['WaterChannel'])
r = env.cpp_env.step(am['Road'])
print(f"\n  WaterChannel+Road: Road R={r['reward']:+.2f}")

# Test: What does Farm get?
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"\n  Farm (fresh): R={r['reward']:+.2f}")

env.close()
