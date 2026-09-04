import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

print('Initial: money=%d, bases=%d' % (g.money, len(g.bases())))
print()

# Try building RIGHT NEXT to the colony (100,100)
# Neighbors: (101,100), (99,100), (100,101), (100,99)
neighbors = [(101,100), (99,100), (100,101), (100,99)]
print('Neighbors of colony:')
for x, y in neighbors:
    print('  (%d,%d): lot=%d' % (x, y, e.lot(x, y)))

print()
print('Attempting builds at neighbors:')
for x, y in neighbors:
    ok, msg = g.build('Farm', x, y)
    print('  build(Farm, %d, %d): ok=%s' % (x, y, ok))
    if ok:
        print('    SUCCESS! money:', g.money, 'bases:', len(g.bases()))
        break

# Try building a Road next to colony
print()
print('Attempting Road at neighbors:')
for x, y in neighbors:
    ok, msg = g.build('Road', x, y)
    print('  build(Road, %d, %d): ok=%s' % (x, y, ok))
    if ok:
        print('    SUCCESS! money:', g.money, 'bases:', len(g.bases()))
        break

# Try building at distance 2 from colony
print()
print('Attempting builds at distance 2:')
for dx, dy in [(2,0), (-2,0), (0,2), (0,-2), (2,2), (-2,-2)]:
    x, y = 100+dx, 100+dy
    ok, msg = g.build('Farm', x, y)
    print('  build(Farm, %d, %d): ok=%s lot=%d' % (x, y, ok, e.lot(x,y)))
