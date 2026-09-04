#!/usr/bin/env python3
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
rc = env.cpp_env.reward_config()
rc.disable_net_worth = True  # isolate build reward
env.cpp_env.set_rewards(rc)
am = {n:i for i,n in enumerate(env._action_names)}

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (no net_worth): R={r['reward']:+.2f}")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (no net_worth): R={r['reward']:+.2f}")

# Now with net_worth
rc.disable_net_worth = False
env.cpp_env.set_rewards(rc)
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (with net_worth): R={r['reward']:+.2f}")

env.close()
