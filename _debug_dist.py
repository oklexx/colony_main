import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a Farm next to colony first
ok, _ = g.build('Farm', 101, 100)
print('Farm at (101,100): ok=%s, bases=%d' % (ok, len(g.bases())))

# Now try to build Sawmill at a WOOD cell
# Find WOOD cells near the colony
wood_cells = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 3:
            wood_cells.append((x, y))
print('WOOD cells within 20 of colony: %d' % len(wood_cells))

# Try building Sawmill at the closest WOOD cell
if wood_cells:
    wood_cells.sort(key=lambda c: (c[0]-101)**2 + (c[1]-100)**2)
    wx, wy = wood_cells[0]
    ok, msg = g.build('Sawmill', wx, wy)
    print('Sawmill at (%d,%d) [WOOD]: ok=%s' % (wx, wy, ok))
    if not ok:
        print('  msg: %s' % msg)
    print('  bases:', len(g.bases()))

# Try building Coalmine at a COAL cell
coal_cells = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 4:
            coal_cells.append((x, y))
print()
print('COAL cells within 20 of colony: %d' % len(coal_cells))
if coal_cells:
    coal_cells.sort(key=lambda c: (c[0]-101)**2 + (c[1]-100)**2)
    cx, cy = coal_cells[0]
    ok, msg = g.build('Coalmine', cx, cy)
    print('Coalmine at (%d,%d) [COAL]: ok=%s' % (cx, cy, ok))
    if not ok:
        print('  msg: %s' % msg)

# Check: is the issue that the game requires the cell to be "connected" to the colony?
# The lot_ok function calls cell_connected. Let's check what that does.
# In C++ code, cell_connected probably checks if the cell is reachable from the colony
# via a path of non-water cells.
# But the game's build() might have its own connectivity check.
# Let's try building at various distances to see the pattern.

print()
print('=== Distance test: building Farm at various distances ===')
env2 = CppColonyEnv(map_size=200)
env2.reset(seed=42)
g2 = env2.cpp_env.game()
e2 = g2.earth

for dist in [1, 2, 3, 5, 10, 15, 20, 30]:
    x, y = 100+dist, 100
    if e2.in_bounds(x, y):
        ok, msg = g2.build('Farm', x, y)
        print('  Farm at (%d,%d) dist=%d: ok=%s lot=%d' % (x, y, dist, ok, e2.lot(x,y)))
