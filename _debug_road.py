import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a Farm next to colony
ok, _ = g.build('Farm', 101, 100)
print('Farm at (101,100): ok=%s' % ok)

# Now build Road to extend connectivity
# Road can be built on LOT_EVERYWHERE
ok, _ = g.build('Road', 99, 100)
print('Road at (99,100): ok=%s' % ok)

# Build another Road
ok, _ = g.build('Road', 100, 99)
print('Road at (100,99): ok=%s' % ok)

ok, _ = g.build('Road', 100, 101)
print('Road at (100,101): ok=%s' % ok)

print('Bases:', len(g.bases()))
print()

# Now try building Sawmill at a WOOD cell
wood_cells = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 3:
            wood_cells.append((x, y))
wood_cells.sort(key=lambda c: (c[0]-100)**2 + (c[1]-100)**2)

# Try the closest WOOD cells
for wx, wy in wood_cells[:5]:
    ok, msg = g.build('Sawmill', wx, wy)
    dist = ((wx-100)**2 + (wy-100)**2)**0.5
    print('Sawmill at (%d,%d) dist=%.1f: ok=%s' % (wx, wy, dist, ok))
    if ok:
        print('  SUCCESS! bases:', len(g.bases()))
        break

# Try Coalmine at COAL cell
print()
coal_cells = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 4:
            coal_cells.append((x, y))
coal_cells.sort(key=lambda c: (c[0]-100)**2 + (c[1]-100)**2)
for cx, cy in coal_cells[:5]:
    ok, msg = g.build('Coalmine', cx, cy)
    dist = ((cx-100)**2 + (cy-100)**2)**0.5
    print('Coalmine at (%d,%d) dist=%.1f: ok=%s' % (cx, cy, dist, ok))
    if ok:
        print('  SUCCESS! bases:', len(g.bases()))
        break
