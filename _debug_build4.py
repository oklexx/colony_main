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

# Try each building type at its required lot
tests = [
    ('Farm', 1), ('Sawmill', 3), ('Coalmine', 4), ('Ironmine', 5),
    ('Refinery', 6), ('Goldmine', 7), ('House', 1), ('Road', 10),
]
for bname, need_lot in tests:
    found = None
    for dy in range(-30, 31):
        for dx in range(-30, 31):
            x, y = 100+dx, 100+dy
            if e.in_bounds(x, y) and e.lot(x, y) == need_lot:
                found = (x, y)
                break
        if found:
            break
    if found:
        x, y = found
        ok, msg = g.build(bname, x, y)
        msg_bytes = msg.encode('utf-8') if isinstance(msg, str) else msg
        print('  build(%s, %d, %d): ok=%s' % (bname, x, y, ok))
        if not ok:
            print('    msg: %s' % msg)
    else:
        print('  %s: no lot=%d cell found within 30 cells' % (bname, need_lot))
