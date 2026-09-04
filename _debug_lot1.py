import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

print('=== Testing: can we build on ANY lot type at distance 1? ===')
# Check lot types at distance 1
for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    x, y = 100+dx, 100+dy
    print('  (%d,%d): lot=%d' % (x, y, e.lot(x, y)))

# Build Farm on each neighbor
print()
for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    x, y = 100+dx, 100+dy
    ok, msg = g.build('Farm', x, y)
    print('  Farm at (%d,%d) lot=%d: ok=%s' % (x, y, e.lot(x,y), ok))

print('Bases:', len(g.bases()))
print()

# Now try Sawmill at distance 1 (even though lot is 1, not 3)
print('=== Testing: Sawmill at distance 1 (lot=1) ===')
ok, msg = g.build('Sawmill', 101, 100)
print('  Sawmill at (101,100) lot=1: ok=%s' % ok)
if not ok:
    print('  msg: %s' % msg)

# Try at all 4 neighbors
for x, y in [(99,100), (100,101), (100,99)]:
    ok, msg = g.build('Sawmill', x, y)
    print('  Sawmill at (%d,%d) lot=%d: ok=%s' % (x, y, e.lot(x,y), ok))

print()
print('=== Testing: Coalmine at distance 1 (lot=1) ===')
for x, y in [(101,100), (99,100), (100,101), (100,99)]:
    ok, msg = g.build('Coalmine', x, y)
    print('  Coalmine at (%d,%d) lot=%d: ok=%s' % (x, y, e.lot(x,y), ok))
