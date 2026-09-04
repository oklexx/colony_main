import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
names = env._action_names

# Sawmill action = 2+3 = 5
obs, rew, term, trunc, info = env.step(5)
print('Sawmill:', 'rew=%.2f' % rew, 'bases=%s' % info.get('bases'), 'money=%s' % info.get('money'))
print('  find_lot(3,False):', env.cpp_env.debug_find_lot(3, False))

# Try again
obs, rew, term, trunc, info = env.step(5)
print('Sawmill again:', 'rew=%.2f' % rew, 'bases=%s' % info.get('bases'), 'money=%s' % info.get('money'))
print('  find_lot(3,False):', env.cpp_env.debug_find_lot(3, False))

# CoalCut action = 2+13 = 15
obs, rew, term, trunc, info = env.step(15)
print('CoalCut:', 'rew=%.2f' % rew, 'bases=%s' % info.get('bases'), 'money=%s' % info.get('money'))
print('  find_lot(4,False):', env.cpp_env.debug_find_lot(4, False))

# What lots exist near the colony?
print()
print('debug_lot_ok around init:')
g = env.cpp_env.game
ix, iy = g.earth.init_sel_x, g.earth.init_sel_y
print('  init_sel:', ix, iy)
for dx in range(-3, 4):
    row = []
    for dy in range(-3, 4):
        x, y = ix+dx, iy+dy
        if g.earth.in_bounds(x, y):
            lot = g.earth.lot(x, y)
            row.append('%2d' % lot)
        else:
            row.append(' .')
    print('  ', ' '.join(row))
