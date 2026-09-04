import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

print('Initial state:')
print('  money:', g.money)
print('  credit:', g.credit)
print('  bases:', g.bases())

# Build Sawmill first
obs, rew, term, trunc, info = env.step(5)
print()
print('After step(5) [Sawmill]:')
print('  rew:', rew)
print('  bases:', g.bases())
print('  money:', g.money)

# Try building directly
print()
print('Attempting direct build...')
for bname in ['Sawmill', 'CoalCut', 'IronMine', 'OilWell', 'GoldMine']:
    try:
        result = g.build(bname)
        print('  build(%s): %s' % (bname, result))
    except Exception as ex:
        print('  build(%s): EXCEPTION %s' % (bname, ex))
    print('  bases now:', len(g.bases()), ' money:', g.money)
