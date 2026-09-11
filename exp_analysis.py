#!/usr/bin/env python3
"""Количественный анализ: достижимость цепочек, шкалы наград, поведение random-политики."""
import sys, collections
import numpy as np
sys.path.insert(0, "python")
import colony_cpp

base_data = colony_cpp.load_base_data("configs/bases.json")
events = colony_cpp.load_events("configs/events.json")

def make_env(seed=42, difficulty="normal", map_size=280):
    rc = colony_cpp.RewardConfig()
    return colony_cpp.ColonyEnvCpp(base_data, events, seed, map_size, 0, [], rc, difficulty)

A_DAY, A_WEEK = 0, 1

def build_idx(env, name):
    return 2 + env.build_ids().index(name)

def mgr_idx(env, j):
    return 2 + env.n_build() + j

def run_scripted(seed=42):
    """Игровая стратегия: вода->ферма, нефть->пила/шахта, уголь->энергия, жильё."""
    env = make_env(seed)
    plan = [
        (1,   "BUILD:Farm"),
        (2,   "BUILD:WaterChannel"),
        (40,  "BUILD:Garden"),
        (41,  "BUILD:Refinery"),
        (70,  "BUILD:Sawmill"),
        (80,  "BUILD:Coalmine"),
        (140, "BUILD:PowerStation"),
        (150, "BUILD:House"),
        (200, "BUILD:CowFarm"),
        (250, "BUILD:Mushroom"),
        (300, "BUILD:House"),
        (400, "BUILD:Apiary"),
        (500, "BUILD:Fish"),
    ]
    env.reset(seed)
    log_rows = []
    first_chain = None
    steps = 0
    while True:
        name = None
        for d, a in plan:
            if d == steps + 1:
                name = a
        if name:
            act = build_idx(env, name[6:])
            if np.array(env.action_mask())[act] == 0:
                act = A_DAY  # не можем — пропускаем день
        else:
            act = A_DAY
        out = env.step(act)
        steps += 1
        if out["terminated"] or out["truncated"]:
            break
        cd = env.last_chain_daily()
        if cd > 0 and first_chain is None:
            first_chain = steps
        if steps % 50 == 0:
            g = env.game()
            log_rows.append((steps, g.days_alive, g.money, g.credit, g.people,
                             len(g.bases()), env.last_chain_daily(),
                             out["metrics"].chains_activated))
    print("=== SCRIPTED CHAIN STRATEGY (seed=%d) ===" % seed)
    print("  step  day   money  credit people bases chain_d chains_act")
    for r in log_rows:
        print("  %4d %4d %7d %6d %6d %5d %8.2f %5d" % r)
    m = out["metrics"]
    g = env.game()
    print("  END step=%d terminated=%s truncated=%s reason=%s" %
          (steps, out["terminated"], out["truncated"], g.game_over()))
    print("  days=%d builds=%d unique=%d chains_activated=%d total_reward=%.1f" %
          (m.days_survived, m.total_builds, m.unique_build_types, m.chains_activated, m.total_reward))
    print("  first chain at step=%s | final money=%d credit=%d" % (first_chain, g.money, g.credit))
    return m

def run_random(seed=7, episodes=5):
    print("=== RANDOM POLICY (masked), v2 rewards ===")
    env = make_env(seed)
    for ep in range(episodes):
        env.reset(seed + ep * 1000)
        acts = collections.Counter()
        ep_r = 0.0
        steps = 0
        while True:
            mask = np.array(env.action_mask(), dtype=float)
            avail = np.nonzero(mask)[0]
            a = int(avail[int(np.random.default_rng(seed + ep * 31 + steps).integers(len(avail)))])
            if a == A_DAY: nm = "DAY"
            elif a == A_WEEK: nm = "WEEK"
            elif a < 2 + env.n_build(): nm = "B:" + env.build_ids()[a - 2]
            else: nm = "M%d" % (a - 2 - env.n_build())
            acts[nm] += 1
            out = env.step(a)
            ep_r += out["reward"]
            steps += 1
            if out["terminated"] or out["truncated"]:
                break
        m = out["metrics"]
        top = ", ".join(f"{k}:{v}" for k, v in acts.most_common(6))
        print("  ep%d: steps=%4d days=%4d bases=%2d chains=%d reward=%8.1f | %s"
              % (ep, steps, m.days_survived, m.base_count_peak, m.chains_activated, ep_r, top))

def run_normalization_check(seed=42, steps=400):
    env = make_env(seed)
    env.reset(seed)
    raws = [np.array(env.obs(), dtype=float)]
    for _ in range(steps):
        env.step(A_DAY)
        raws.append(np.array(env.obs(), dtype=float))
    raws = np.array(raws)
    mean = raws.mean(axis=0)
    std = raws.std(axis=0)
    print("=== OBS STATISTICS (%d DAY-steps, seed=%d, no buildings) ===" % (steps, seed))
    print("  f0  year/50    : mean=%.3f std=%.3f" % (mean[0], std[0]))
    print("  f4  money/2e5  : mean=%.3f std=%.3f" % (mean[4], std[4]))
    print("  f6  people/100 : mean=%.3f std=%.3f" % (mean[6], std[6]))
    n_off = int((np.abs(mean) > 0.1).sum())
    print("  фич с |mean|>0.1 (нецентрированы): %d из %d" % (n_off, len(mean)))
    print("  -> модель обучалась на (x-mean)/std; после resume без load_normalization")
    print("     RMS=identity и она получает raw: смещение входов до %.2f сигм на фичу" %
          (np.abs(mean)[np.abs(mean) > 0.1].max()))

if __name__ == "__main__":
    for s in (42, 7):
        run_scripted(s)
    run_random()
    run_normalization_check()
