#!/usr/bin/env python3
"""Test new reward configuration."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "python")

import numpy as np
from cpp_env import CppColonyEnv

# Test with NEW default rewards
env = CppColonyEnv(map_size=200)
env.cpp_env.reset(42)

action_names = env._action_names
build_ids = env.cpp_env.build_ids()
action_map = {name: i for i, name in enumerate(action_names)}

print("=" * 60)
print("NEW DEFAULT REWARDS")
print("=" * 60)

# Check default values
rc = env.cpp_env.reward_config()
print(f"  build_bonus:        {rc.build_bonus}")
print(f"  tax_daily_bonus:    {rc.tax_daily_bonus}")
print(f"  preserve_penalty:   {rc.preserve_penalty}")
print(f"  idle_build_penalty: {rc.idle_build_penalty}")
print(f"  idle_build_threshold_days: {rc.idle_build_threshold_days}")
print(f"  diversity_bonus:    {rc.diversity_bonus}")
print(f"  daily_income:       {rc.daily_income}")
print(f"  error_penalty:      {rc.error_penalty}")

print()
print("=" * 60)
print("TEST: Immediate rewards (step 1)")
print("=" * 60)

# Test each action
for name in ["DAY", "PAY_TAX", "PRESERVE", "UNPRESERVE",
             "WaterChannel", "Farm", "Apiary", "Road", "SmallHouse"]:
    env.cpp_env.reset(42)
    idx = action_map[name]
    r = env.cpp_env.step(idx)
    print(f"  {name:<16s}: R={r['reward']:+.2f}")

print()
print("=" * 60)
print("TEST: PRESERVE spam (should degrade)")
print("=" * 60)
env.cpp_env.reset(42)
total = 0
for i in range(5):
    r = env.cpp_env.step(action_map["PRESERVE"])
    total += r["reward"]
    print(f"  PRESERVE {i+1}: R={r['reward']:+.2f} cumul={total:+.2f}")

print()
print("=" * 60)
print("TEST: Build chain (WaterChannel -> Farm -> Apiary -> Road)")
print("=" * 60)
env.cpp_env.reset(42)
total = 0
for name in ["WaterChannel", "Farm", "Apiary", "Road", "DAY", "DAY", "DAY"]:
    r = env.cpp_env.step(action_map[name])
    total += r["reward"]
    print(f"  {name:<16s}: R={r['reward']:+.2f} cumul={total:+.2f}")

print()
print("=" * 60)
print("TEST: Diversity bonus (build different types)")
print("=" * 60)
env.cpp_env.reset(42)
total = 0
types_built = []
for name in ["WaterChannel", "Farm", "Apiary", "Road", "SmallHouse", "Apiary"]:
    r = env.cpp_env.step(action_map[name])
    total += r["reward"]
    if "built" in r and r["built"]:
        types_built.append(name)
    print(f"  {name:<16s}: R={r['reward']:+.2f} cumul={total:+.2f} built={r.get('built', 'N/A')}")

print()
print("=" * 60)
print("TEST: Compare strategies (20 steps)")
print("=" * 60)

# Strategy 1: Build
env.cpp_env.reset(42)
total = 0
actions = ["WaterChannel", "Farm", "Apiary", "SmallHouse", "Road",
           "DAY", "DAY", "DAY", "DAY", "DAY",
           "Apiary", "Road", "Garden", "Road", "Apiary",
           "DAY", "DAY", "DAY", "DAY", "DAY"]
for name in actions:
    r = env.cpp_env.step(action_map[name])
    total += r["reward"]
print(f"  Build strategy: {total:+.2f}")

# Strategy 2: DAY only
env.cpp_env.reset(42)
total = 0
for _ in range(20):
    r = env.cpp_env.step(action_map["DAY"])
    total += r["reward"]
print(f"  DAY-only:       {total:+.2f}")

# Strategy 3: PRESERVE spam
env.cpp_env.reset(42)
total = 0
for _ in range(20):
    r = env.cpp_env.step(action_map["PRESERVE"])
    total += r["reward"]
print(f"  PRESERVE spam:  {total:+.2f}")

# Strategy 4: PAY_TAX spam
env.cpp_env.reset(42)
total = 0
for _ in range(20):
    r = env.cpp_env.step(action_map["PAY_TAX"])
    total += r["reward"]
print(f"  PAY_TAX spam:   {total:+.2f}")

env.close()
