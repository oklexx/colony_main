import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a Farm at (101,100)
ok, _ = g.build('Farm', 101, 100)
print('Farm at (101,100): ok=%s' % ok)

# Build a Road at (102,100)
ok, _ = g.build('Road', 102, 100)
print('Road at (102,100): ok=%s' % ok)

# Build another Road at (103,100)
ok, _ = g.build('Road', 103, 100)
print('Road at (103,100): ok=%s' % ok)

print('Bases:', len(g.bases()))
print()

# Now try building a Farm at (104,100) — should be connected via Road chain
ok, msg = g.build('Farm', 104, 100)
print('Farm at (104,100) via Road chain: ok=%s' % ok)
if not ok:
    print('  msg: %s' % msg)
else:
    print('  SUCCESS! The issue is that WOOD/COAL cells need to be ADJACENT to a non-Road building, not just reachable via Roads.')

print()
# Let's verify: can we build a Road on a WOOD cell?
wood_cells = []
for dy in range(-5, 6):
    for dx in range(-5, 6):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 3:
            wood_cells.append((x, y))
wood_cells.sort(key=lambda c: (c[0]-100)**2 + (c[1]-100)**2)
print('Closest WOOD cells:')
for wx, wy in wood_cells[:5]:
    ok, msg = g.build('Road', wx, wy)
    print('  Road at (%d,%d) [WOOD]: ok=%s' % (wx, wy, ok))
    if ok:
        break
