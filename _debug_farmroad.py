import sys
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main')
sys.path.insert(0, r'C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\python')
from cpp_env import CppColonyEnv

env = CppColonyEnv(map_size=200)
env.reset(seed=42)
g = env.cpp_env.game()
e = g.earth

# Build a Farm at (99,100) — adjacent to colony
ok, _ = g.build('Farm', 99, 100)
print('Farm at (99,100): ok=%s' % ok)

# Build a Farm at (99,99) — adjacent to (99,100)
ok, _ = g.build('Farm', 99, 99)
print('Farm at (99,99): ok=%s' % ok)

# Build a Farm at (99,98) — adjacent to (99,99)
ok, _ = g.build('Farm', 99, 98)
print('Farm at (99,98): ok=%s' % ok)

# Build a Farm at (99,97) — adjacent to (99,98)
ok, _ = g.build('Farm', 99, 97)
print('Farm at (99,97): ok=%s' % ok)

# Build a Road at (99,96) — adjacent to (99,97)
ok, _ = g.build('Road', 99, 96)
print('Road at (99,96): ok=%s' % ok)

# Build a Road at (99,95) — adjacent to (99,96)
ok, _ = g.build('Road', 99, 95)
print('Road at (99,95): ok=%s' % ok)

# Build a Road at (99,94) — adjacent to (99,95)
ok, _ = g.build('Road', 99, 94)
print('Road at (99,94): ok=%s' % ok)

# Build a Road at (98,94) — adjacent to (99,94), and (98,94) is adjacent to WOOD (98,93)
ok, _ = g.build('Road', 98, 94)
print('Road at (98,94): ok=%s' % ok)

print('Bases:', len(g.bases()))
print()

# Now try Sawmill at (98,93) — the WOOD cell
ok, msg = g.build('Sawmill', 98, 93)
print('Sawmill at (98,93): ok=%s' % ok)
if ok:
    print('  SUCCESS!')
    print('  bases:', len(g.bases()))
else:
    print('  Still fails. msg: %s' % msg)
