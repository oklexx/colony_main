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
print('After Sawmill build:')
print('  bases count:', len(g.bases()) if hasattr(g, 'bases') else '?')
print('  find_lot(WOOD=3):', env.cpp_env.debug_find_lot(3, False))
print('  find_lot(COAL=4):', env.cpp_env.debug_find_lot(4, False))
print('  find_lot(EVERYWHERE=10):', env.cpp_env.debug_find_lot(10, False))

# Check if WOOD cells exist near the colony
print()
print('WOOD cells near colony (100,100):')
found_wood = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 3:
            found_wood.append((x, y))
print('  count within 20 cells:', len(found_wood))
if found_wood:
    print('  first few:', found_wood[:5])
    # Check lot_ok on one
    wx, wy = found_wood[0]
    print('  lot_ok(%d,%d, WOOD=3):' % (wx, wy), env.cpp_env.debug_lot_ok(wx, wy, 3, False))
    print('  lot_ok(%d,%d, EVERYWHERE=10):' % (wx, wy), env.cpp_env.debug_lot_ok(wx, wy, 10, False))

# Check all lot types in a 50-cell radius
print()
print('Lot type distribution within 50 cells of colony:')
from collections import Counter
counts = Counter()
for dy in range(-50, 51):
    for dx in range(-50, 51):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y):
            counts[e.lot(x, y)] += 1
for k in sorted(counts):
    print('  lot=%d: %d' % (k, counts[k]))
