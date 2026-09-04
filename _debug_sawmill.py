import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a chain from (100,100) to (98,92) — adjacent to WOOD cell (98,93)
path = [(99,100), (98,100), (98,99), (98,98), (98,97), (98,96), (98,95), (98,94), (98,93), (98,92)]
for x, y in path:
    ok, _ = g.build('Farm', x, y)
    if not ok:
        ok, _ = g.build('Road', x, y)
print('Chain built, bases:', len(g.bases()))
print()

# Now try Sawmill at (98,93) — the WOOD cell
ok, msg = g.build('Sawmill', 98, 93)
print('Sawmill at (98,93): ok=%s' % ok)
if not ok:
    # Try other nearby WOOD cells
    wood_cells = []
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            x, y = 98+dx, 93+dy
            if e.in_bounds(x, y) and e.lot(x, y) == 3:
                wood_cells.append((x, y))
    print('WOOD cells near (98,93):', wood_cells)
    for wx, wy in wood_cells:
        ok, msg = g.build('Sawmill', wx, wy)
        print('  Sawmill at (%d,%d): ok=%s' % (wx, wy, ok))
        if ok:
            print('  SUCCESS!')
            break

# Also check: what does the C++ build() function check?
# Let's look at the C++ source to understand the connectivity requirement.
print()
print('=== Checking C++ build() implementation ===')
# The build() function in colony.cpp probably checks:
# 1. lot_ok (lot type match + cell_connected)
# 2. money
# 3. cell_connected (path from colony to this cell)
# The cell_connected function likely does a BFS/DFS from the colony
# through cells that have buildings (or roads) to see if the target
# cell is reachable.

# Let's check if the C++ code has a specific connectivity check
# by looking at the build() function.
