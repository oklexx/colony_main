import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a chain from (100,100) to (98,92) — just next to the WOOD cell (98,93)
print('Building chain to (98,92):')
path = [(99,100), (98,100), (98,99), (98,98), (98,97), (98,96), (98,95), (98,94), (98,93), (98,92)]
for x, y in path:
    ok, _ = g.build('Farm', x, y)
    if not ok:
        ok, _ = g.build('Road', x, y)
    print('  (%d,%d) lot=%d: %s' % (x, y, e.lot(x,y), 'OK' if ok else 'FAIL'))

print('Bases:', len(g.bases()))
print()

# Now try Sawmill at (98,93) — the WOOD cell
ok, msg = g.build('Sawmill', 98, 93)
print('Sawmill at (98,93) [WOOD, surrounded]: ok=%s' % ok)
if ok:
    print('  SUCCESS! bases:', len(g.bases()))
else:
    print('  Still fails!')
    print()
    # Try building a Road on the WOOD cell
    ok, msg = g.build('Road', 98, 93)
    print('  Road at (98,93) [WOOD]: ok=%s' % ok)
    if ok:
        print()
        # Now try Sawmill again
        ok, msg = g.build('Sawmill', 98, 93)
        print('  Sawmill at (98,93) after Road: ok=%s' % ok)
