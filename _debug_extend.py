import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build Farm next to colony to extend connectivity
ok, _ = g.build('Farm', 101, 100)
print('Farm at (101,100): ok=%s' % ok)

# Build Roads to extend connectivity in all directions
for x, y in [(99,100), (100,99), (100,101), (102,100), (101,99), (101,101)]:
    ok, _ = g.build('Road', x, y)
    if ok:
        print('  Road at (%d,%d): OK' % (x, y))

print('Bases:', len(g.bases()))
print()

# Now try Sawmill at WOOD cells
wood_cells = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 3:
            wood_cells.append((x, y))
wood_cells.sort(key=lambda c: (c[0]-100)**2 + (c[1]-100)**2)

print('Trying Sawmill at closest WOOD cells:')
for wx, wy in wood_cells[:10]:
    ok, msg = g.build('Sawmill', wx, wy)
    dist = ((wx-100)**2 + (wy-100)**2)**0.5
    print('  (%d,%d) dist=%.1f: ok=%s' % (wx, wy, dist, ok))
    if ok:
        print('  SUCCESS!')
        break

print()
print('Trying Coalmine at closest COAL cells:')
coal_cells = []
for dy in range(-20, 21):
    for dx in range(-20, 21):
        x, y = 100+dx, 100+dy
        if e.in_bounds(x, y) and e.lot(x, y) == 4:
            coal_cells.append((x, y))
coal_cells.sort(key=lambda c: (c[0]-100)**2 + (c[1]-100)**2)
for cx, cy in coal_cells[:10]:
    ok, msg = g.build('Coalmine', cx, cy)
    dist = ((cx-100)**2 + (cy-100)**2)**0.5
    print('  (%d,%d) dist=%.1f: ok=%s' % (cx, cy, dist, ok))
    if ok:
        print('  SUCCESS!')
        break
