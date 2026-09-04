import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build Sawmill first (action 5)
obs, rew, term, trunc, info = env.step(5)
print('After Sawmill build:')

# Check lot_ok for WOOD cells at various distances
print()
print('lot_ok for WOOD cells (need=3):')
for (wx, wy) in [(87, 83), (88, 83), (89, 83), (87, 84), (82, 85)]:
    ok = env.cpp_env.debug_lot_ok(wx, wy, 3, False)
    ok_all = env.cpp_env.debug_lot_ok(wx, wy, 10, False)
    dist = ((wx-100)**2 + (wy-100)**2)**0.5
    print('  (%d,%d) dist=%.1f: lot_ok(WOOD)=%s lot_ok(ALL)=%s' % (wx, wy, dist, ok, ok_all))

# Check cell_connected - maybe the issue is connectivity
print()
print('Checking cell_connected for WOOD cells:')
# The lot_ok function calls g.cell_connected(x, y) at the end
# Let's see if there's a debug method for that
print('  (no direct debug method available)')

# Let's check: what does find_lot see for WOOD?
# find_lot does BFS from colony. Let's trace it.
# The BFS starts from neighbors of each base.
# After building Sawmill at some location, what are the neighbors?
print()
print('Bases after build:')
for i, b in enumerate(g.bases()):
    print('  base %d: id=%s x=%d y=%d' % (i, b.data().id, b.x, b.y))
    # Check neighbors
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = b.x+dx, b.y+dy
        if e.in_bounds(nx, ny):
            print('    neighbor (%d,%d): lot=%d' % (nx, ny, e.lot(nx, ny)))
