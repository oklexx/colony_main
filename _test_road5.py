#!/usr/bin/env python3
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
rc = env.cpp_env.reward_config()

# Disable everything except build_bonus
rc.disable_net_worth = True
rc.disable_daily_income = True
rc.novelty = 0.0
rc.diversity_bonus = 0.0
rc.chain_bonus = 0.0
rc.chain_daily = 0.0
rc.survival_bonus = 0.0
rc.tax_daily_bonus = 0.0
rc.daily_income = 0.0
rc.proximity_bonus = 0.0
rc.disable_provider_bonus = True
env.cpp_env.set_rewards(rc)

am = {n:i for i,n in enumerate(env._action_names)}

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (minimal): R={r['reward']:+.4f}")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (minimal): R={r['reward']:+.4f}")

# Now enable diversity only
rc.diversity_bonus = 8.0
env.cpp_env.set_rewards(rc)
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (diversity=8): R={r['reward']:+.4f}")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (diversity=8): R={r['reward']:+.4f}")

# Now enable novelty only
rc.diversity_bonus = 0.0
rc.novelty = 15.0
env.cpp_env.set_rewards(rc)
env.cpp_env.reset(42)
r = env.cpp_env.step(am['Road'])
print(f"Road (novelty=15): R={r['reward']:+.4f}")

env.cpp_env.reset(42)
r = env.cpp_env.step(am['Farm'])
print(f"Farm (novelty=15): R={r['reward']:+.4f}")

env.close()
