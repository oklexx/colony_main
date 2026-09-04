"""Verify building placement is always adjacent to a building or road-connected to the colony."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from colony_cpp import ColonyEnvCpp, load_base_data, load_events  # noqa: E402


def main():
    base_data = load_base_data(str(ROOT / "configs" / "bases.json"))
    events = load_events(str(ROOT / "configs" / "events.json"))
    env = ColonyEnvCpp(base_data, events, seed=42, map_size=120)
    env.reset(42)
    g = env.game()

    build_ids = env.build_ids()
    print("build ids:", build_ids)
    road_idx = build_ids.index("Road") if "Road" in build_ids else None
    n_build = env.n_build()

    def cell_dict(x, y):
        for d in g.bases():
            if d["x"] == x and d["y"] == y:
                return d
        return None

    def is_road(x, y):
        d = cell_dict(x, y)
        return d is not None and d["id"] == "Road"

    def is_building(x, y):
        d = cell_dict(x, y)
        return d is not None and d["id"] != "Road"

    def check_cell(x, y, failures):
        dx4 = (1, -1, 0, 0)
        dy4 = (0, 0, 1, -1)
        for i in range(4):
            if is_building(x + dx4[i], y + dy4[i]):
                return
        visited = set()
        q = []
        for i in range(4):
            nx, ny = x + dx4[i], y + dy4[i]
            if is_road(nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
        while q:
            cx, cy = q.pop(0)
            for i in range(4):
                nx, ny = cx + dx4[i], cy + dy4[i]
                if (nx, ny) in visited:
                    continue
                d = cell_dict(nx, ny)
                if d is None:
                    continue
                visited.add((nx, ny))
                if d["id"] == "Road":
                    q.append((nx, ny))
                else:
                    return
        failures.append(f"cell ({x},{y}) is isolated (no adjacent building, no road link)")

    failures = []
    for step in range(3000):
        action = 2 + (road_idx if road_idx is not None and step % 5 == 0 else (step % (n_build - 1)))
        env.step(action)
        g = env.game()
        for d in g.bases():
            if d["id"] == "Road":
                continue
            if d["x"] == 60 and d["y"] == 60:
                continue
            check_cell(d["x"], d["y"], failures)
        if failures:
            break

    print(f"steps={step+1} bases={len(g.bases())}")
    if failures:
        print("FAIL:")
        for f in failures[:10]:
            print("  ", f)
        sys.exit(1)
    print("PASS: all buildings are adjacent to a building or road-connected to the colony")


if __name__ == "__main__":
    main()
