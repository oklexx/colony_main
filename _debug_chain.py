import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a chain of Farms/Roads from (100,100) to (98,93)
# Path: (100,100) -> (99,100) -> (98,100) -> (98,99) -> (98,98) -> (98,97) -> (98,96) -> (98,95) -> (98,94) -> (98,93)
print('Building chain to (98,93):')
path = [(99,100), (98,100), (98,99), (98,98), (98,97), (98,96), (98,95), (98,94), (98,93)]
for x, y in path:
    # Try Farm first, then Road
    ok, _ = g.build('Farm', x, y)
    if not ok:
        ok, _ = g.build('Road', x, y)
    print('  (%d,%d): %s' % (x, y, 'OK' if ok else 'FAIL'))

print('Bases:', len(g.bases()))
print()

# Now try Sawmill at (98,93)
ok, msg = g.build('Sawmill', 98, 93)
print('Sawmill at (98,93) after chain: ok=%s' % ok)
if ok:
    print('  SUCCESS! bases:', len(g.bases()))
else:
    print('  Still fails. Trying with a different approach...')
    # Maybe the issue is that the WOOD cell needs to be ADJACENT to a non-wood building?
    # Let's try building a Farm on the WOOD cell itself
    ok, msg = g.build('Farm', 98, 93)
    print('  Farm at (98,93) [WOOD]: ok=%s' % ok)
