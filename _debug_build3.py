import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

print('Initial: money=%d, bases=%d' % (g.money, len(g.bases())))

# Build Sawmill directly at a WOOD cell
print()
print('=== Building at specific locations ===')
for bname, need_lot in [('Sawmill', 3), ('CoalCut', 4), ('IronMine', 5), ('OilWell', 6), ('GoldMine', 7)]:
    # Find a cell with the right lot type
    found = None
    for dy in range(-20, 21):
        for dx in range(-20, 21):
            x, y = 100+dx, 100+dy
            if e.in_bounds(x, y) and e.lot(x, y) == need_lot:
                found = (x, y)
                break
        if found:
            break
    if found:
        x, y = found
        ok, msg = g.build(bname, x, y)
        print('  build(%s, %d, %d): ok=%s msg=%s' % (bname, x, y, ok, msg))
        if ok:
            print('    money:', g.money, 'bases:', len(g.bases()))
    else:
        print('  %s: no %d cell found nearby' % (bname, need_lot))
