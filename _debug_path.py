import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Find the closest WOOD cell
wood_cells = []
for dy in range(-30, 31):
    for dx in range(-30, 31):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 3:
            wood_cells.append((x, y))
wood_cells.sort(key=lambda c: (c[0]-100)**2 + (c[1]-100)**2)
print('Closest 5 WOOD cells:')
for wx, wy in wood_cells[:5]:
    print('  (%d,%d) dist=%.1f' % (wx, wy, ((wx-100)**2 + (wy-100)**2)**0.5))

# Build a Farm at (101,100) to start the chain
ok, _ = g.build('Farm', 101, 100)
print()
print('Farm at (101,100): ok=%s' % ok)

# Now build a chain of Farms from (101,100) toward the WOOD cell
# Path: (101,100) -> (102,100) -> ... -> (101,99) -> (101,98) -> ... -> WOOD cell
# Actually, let's just build Farms on ALL cells between (100,100) and the WOOD cell
# This is the key insight: the cell_connected() function requires the target cell
# to be ADJACENT to a non-Road building, OR reachable via a Road chain that
# ends at a non-Road building.

# Let's try: build a Farm at each cell from (101,100) to the WOOD cell
# This is expensive but should work.

wx, wy = wood_cells[0]
print('Target WOOD cell: (%d,%d)' % (wx, wy))

# Build Farms on a straight-line path from (101,100) to (wx, wy)
# Simple approach: build Farms on all cells in the bounding box
# This is wasteful but should work.

# Actually, let's just build Farms on a simple path
# From (101,100) to (wx, wy):
# - Move horizontally to wx, then vertically to wy
# - Or vice versa

# Let's try: move right to wx, then up/down to wy
path = []
cx, cy = 101, 100
while cx != wx:
    cx += 1 if wx > cx else -1
    path.append((cx, cy))
while cy != wy:
    cy += 1 if wy > cy else -1
    path.append((cx, cy))

print('Path length: %d' % len(path))
for x, y in path:
    ok, _ = g.build('Farm', x, y)
    if not ok:
        ok, _ = g.build('Road', x, y)
    if not ok:
        print('  FAIL at (%d,%d)' % (x, y))
        break
else:
    print('  All path cells built!')

print('Bases:', len(g.bases()))
print()

# Now try Sawmill at the WOOD cell
ok, msg = g.build('Sawmill', wx, wy)
print('Sawmill at (%d,%d): ok=%s' % (wx, wy, ok))
if ok:
    print('  SUCCESS!')
    print('  bases:', len(g.bases()))
else:
    print('  Still fails. msg: %s' % msg)
    # The issue is that the WOOD cell is NOT adjacent to a non-Road building
    # in the path. The path ends at (wx, wy), but the cell_connected() function
    # checks if (wx, wy) is adjacent to a non-Road building.
    # Let's check: is (wx, wy) adjacent to any non-Road building?
    print()
    print('  Checking neighbors of (%d,%d):' % (wx, wy))
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = wx+dx, wy+dy
        b = g.base_in_box(nx, ny)
        if b:
            print('    (%d,%d): %s' % (nx, ny, b['name']))
        else:
            print('    (%d,%d): empty' % (nx, ny))
