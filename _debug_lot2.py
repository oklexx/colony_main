import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth
ix, iy = e.init_sel_x, e.init_sel_y
print('init_sel:', ix, iy, ' map_size:', g.map_size())
print()
print('Lot map around colony (LT: 1=normal,2=good,3=excellent,4=coal,10=road):')
for dy in range(-4, 5):
    row = []
    for dx in range(-4, 5):
        x, y = ix+dx, iy+dy
        if e.in_bounds(x, y):
            row.append('%d' % e.lot(x, y))
        else:
            row.append('.')
    print('  ', ' '.join(row))

print()
print('find_lot results:')
for need, label in [(1, 'LT_NORMAL'), (2, 'LT_GOOD'), (3, 'LT_EXCELLENT'), (4, 'LT_COAL'), (10, 'LT_ROAD')]:
    print('  need_earth=%d (%s): %s' % (need, label, env.cpp_env.debug_find_lot(need, False)))

print()
print('lot_ok results at specific cells:')
for dx in range(-2, 3):
    for dy in range(-2, 3):
        x, y = ix+dx, iy+dy
        if e.in_bounds(x, y):
            for need in (1, 2, 3, 4):
                ok = env.cpp_env.debug_lot_ok(x, y, need, False)
                if ok:
                    print('  lot_ok(%d,%d, need=%d) = %s' % (x, y, need, ok))
