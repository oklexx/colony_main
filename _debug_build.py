import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build Sawmill first
obs, rew, term, trunc, info = env.step(5)
print('After Sawmill build: bases =', g.bases())

# The Sawmill was NOT built (bases only has City)
# Let's check: what does the game think about building a Sawmill?
# Check money
print('money:', g.money(), 'credit:', g.credit())

# Try building directly via game.build()
print()
print('Attempting game.build("Sawmill")...')
try:
    result = g.build('Sawmill')
    print('  result:', result)
except Exception as ex:
    print('  exception:', ex)

print('bases after build attempt:', g.bases())
print('money:', g.money())

# Let's also check: what actions are available?
print()
print('Action names:', env._action_names)
print('n_actions:', env.cpp_env.n_actions())
print('n_build:', env.cpp_env.n_build())
