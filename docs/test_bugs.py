# -*- coding: utf-8 -*-
"""Живые тесты багов, найденных при аудите кода обучения (RL_TRAINING_AUDIT.md v2).

ВАЖНО (ревизия v2): тесты написаны против md-экспорта (training_performance_report/
CODE_IMPLEMENTATION), который УСТАРЕЛ. В живом коде уже исправлены:
  - A1 (avg_return NameError)   — живой async_trainer.py:543-553
  - A2 (is_hybrid)              — живой ppo.py:64-65
  - B5 (_last_terminateds)      — живой cpp_vecenv.py:180 всегда пишет атрибут
При запуске этого файла против ЖИВОГО rl/ (скопировать живые файлы в rl/)
тест is_hybrid должен дать NOT REPRODUCED — это означает "баг исправлен".
Остальные тесты (B1, A4, A6, B3, B6, C1, C6, F3) подтверждены в живом коде.
Запуск: cd work && python3 test_bugs.py
"""
import sys, math, json, traceback
import torch

RESULTS = []
def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "CONFIRMED"))
        print(f"[CONFIRMED] {name}")
    except AssertionError as e:
        RESULTS.append((name, f"NOT REPRODUCED: {e}"))
        print(f"[NOT REPRODUCED] {name}: {e}")
    except Exception as e:
        RESULTS.append((name, f"ERROR {type(e).__name__}: {e}"))
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")

# ---------- 1. RewardConfig: from_dict defaults != dataclass defaults ----------
def t_reward_defaults():
    from rl.config import RewardConfig
    import dataclasses
    d = RewardConfig()
    rt = RewardConfig.from_dict({})  # пустой JSON
    diffs = []
    for f in dataclasses.fields(RewardConfig):
        a, b = getattr(d, f.name), getattr(rt, f.name)
        if a != b:
            diffs.append(f"{f.name}: dataclass={a} vs from_dict={b}")
    assert diffs, "различий нет"
    print("   " + "; ".join(diffs))
check("RewardConfig.from_dict меняет веса наград при round-trip через JSON", t_reward_defaults)

# ---------- 2. Config.load_from_file -> NameError (return cfg) ----------
def t_load_from_file():
    from rl.config import Config
    cfg = Config()
    p = "/tmp/_cfg_test.json"
    with open(p, "w") as f:
        json.dump({"seed": 123}, f)
    try:
        cfg.load_from_file(p)
    except NameError as e:
        assert "cfg" in str(e)
        return
    raise AssertionError("NameError не возник")
check("Config.load_from_file падает: NameError 'cfg' (должно быть return self)", t_load_from_file)

# ---------- 3. get_curriculum_progress default schedule -> TypeError ----------
def t_curriculum():
    default_schedule = [(100000, 1), (500000, 2), (1000000, 3)]
    try:
        schedule = [(t, s+1) for t, s in enumerate(default_schedule)]
    except TypeError as e:
        print(f"   TypeError: {e}")
        return
    raise AssertionError("TypeError не возник")
check("env_manager.get_curriculum_progress: список по умолчанию падает с TypeError", t_curriculum)

# ---------- 4. PPO.is_hybrid не определён -> AttributeError ----------
def t_is_hybrid():
    from rl.actor_critic import ActorCritic
    from rl.rollout_buffer import RolloutBuffer
    from rl.ppo import PPO
    dev = torch.device("cpu")
    model = ActorCritic(obs_size=16, n_actions=5, hidden_sizes=[8,8], device=dev)
    buf = RolloutBuffer(4, 2, 16, 5, 0.99, 0.95, dev)
    ppo = PPO(model, buf, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
              ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5, n_epochs=2,
              batch_size=4, use_amp=False, device=dev)
    assert not hasattr(ppo, "is_hybrid"), "атрибут is_hybrid откуда-то появился"
    obs = torch.randn(2, 16)
    try:
        ppo.collect_step(obs)
    except AttributeError as e:
        print(f"   AttributeError: {e}")
        return
    raise AssertionError("collect_step не упал")
check("PPO.collect_step падает: 'PPO' object has no attribute 'is_hybrid'", t_is_hybrid)

# ---------- 5. GAE: сверка с эталонной реализацией (CleanRL-style) ----------
def t_gae():
    from rl.rollout_buffer import RolloutBuffer
    torch.manual_seed(0)
    dev = torch.device("cpu")
    n_steps, n_envs, gamma, lam = 6, 3, 0.99, 0.95
    buf = RolloutBuffer(n_steps, n_envs, 4, 2, gamma, lam, dev)
    T = n_steps * n_envs
    obs = torch.randn(T, 4); act = torch.randint(0, 2, (T,))
    rew = torch.randn(T); lp = torch.randn(T); val = torch.rand(T)
    dn = torch.rand(T) < 0.3; term = dn.clone()
    buf.obs[:]=obs; buf.actions[:]=act; buf.rewards[:]=rew
    buf.log_probs[:]=lp; buf.values[:]=val; buf.dones[:]=dn; buf.terminated[:]=term
    last_value = torch.rand(n_envs); last_done = torch.rand(n_envs) < 0.5
    buf.compute_gae(last_value, last_done)
    # эталон: dones в конвенции "terminated[t] == terminal(s_{t+1})"
    vals = val.view(n_steps, n_envs); rs = rew.view(n_steps, n_envs); tms = term.view(n_steps, n_envs)
    adv_ref = torch.zeros(n_steps, n_envs); lastg = torch.zeros(n_envs)
    for t in reversed(range(n_steps)):
        if t == n_steps-1:
            nn_, nv = 1.0 - last_done.float(), last_value
        else:
            nn_, nv = 1.0 - tms[t].float(), vals[t+1]
        delta = rs[t] + gamma*nv*nn_ - vals[t]
        lastg = delta + gamma*lam*nn_*lastg
        adv_ref[t] = lastg
    adv_ref = adv_ref.reshape(-1)
    diff = (buf.advantages - (adv_ref - adv_ref.mean())/(adv_ref.std()+1e-8)).abs().max()
    assert diff < 1e-5, f"GAE расходится с эталоном: {diff}"
    ret_diff = (buf.returns - (adv_ref + val)).abs().max()
    assert ret_diff < 1e-5, f"returns расходятся: {ret_diff}"
    print("   GAE и returns совпадают с эталоном (рекуррентность корректна)")
check("GAE: рекуррентность корректна (сверка с эталоном)", t_gae)

# ---------- 6. Пустая маска действий -> NaN ----------
def t_empty_mask():
    logits = torch.randn(2, 5)
    masks = torch.zeros(2, 5)  # все действия заблокированы
    masked = logits.masked_fill(masks == 0, float("-inf"))
    dist = torch.distributions.Categorical(logits=masked)
    a = dist.sample()
    lp = dist.log_prob(a)
    ent = dist.entropy()
    assert torch.isnan(lp).any() or torch.isnan(ent).any(), "NaN не возник"
    print(f"   sample={a.tolist()} log_prob={lp.tolist()} entropy={ent.tolist()} -> NaN в буфере и в loss")
check("collect_step: если все действия замаскированы -> NaN log_prob попадает в буфер", t_empty_mask)

# ---------- 7. action_masks в batch всегда не-None (даже если env их не давал) ----------
def t_masks_always_yielded():
    from rl.rollout_buffer import RolloutBuffer
    dev = torch.device("cpu")
    buf = RolloutBuffer(2, 2, 4, 3, 0.99, 0.95, dev)
    for _ in range(2):  # add БЕЗ action_masks (env их не предоставил)
        buf.add(torch.randn(2,4), torch.randint(0,3,(2,)), torch.randn(2),
                torch.randn(2), torch.randn(2), torch.zeros(2,dtype=torch.bool))
    batch = next(iter(buf.get_batches(4)))
    assert batch["action_masks"] is not None
    m = batch["action_masks"]
    # неинициализированная память torch.empty -> произвольные нули/единицы
    print(f"   masks всегда в batch; примеры значений из неинициализированной памяти: {m.flatten()[:8].tolist()}")
check("update(): мусорные action_masks (torch.empty) применяются к логитам, если env их не давал", t_masks_always_yielded)

# ---------- 8. async_trainer: 1D actions -> loop-detection видит только шаг 0 ----------
def t_action_indexing():
    from rl.rollout_buffer import RolloutBuffer
    dev = torch.device("cpu")
    n_steps, n_envs = 5, 2
    buf = RolloutBuffer(n_steps, n_envs, 4, 6, 0.99, 0.95, dev)
    # действия: на шаге t все env делают действие t+1
    for t in range(n_steps):
        a = torch.full((n_envs,), t+1)
        buf.add(torch.randn(n_envs,4), a, torch.randn(n_envs),
                torch.randn(n_envs), torch.randn(n_envs), torch.zeros(n_envs,dtype=torch.bool))
    seen = []
    for step in range(n_steps):  # повторяем логику _collect_rollout
        pos = buf.pos - (n_steps - step)  # имитация pos на шаге step
        actions_tensor = buf.actions
        actions_np = actions_tensor.numpy()  # .cpu().numpy() в оригинале
        if actions_np.ndim == 2:
            step_actions = actions_np[pos, :]
        elif actions_np.ndim == 1:
            step_actions = actions_np     # <-- ветка, которая реально выполняется (буфер 1D!)
        else:
            step_actions = []
        got = [int(step_actions[i]) for i in range(min(n_envs, len(step_actions)))]
        seen.append(got)
    expected = [[t+1, t+1] for t in range(n_steps)]
    print(f"   ожидалось (действия по шагам): {expected}")
    print(f"   фактически видит loop-detection: {seen}")
    assert all(g == [1,1] for g in seen), "не воспроизвелось"
check("async_trainer: буфер actions 1D -> loop-detection все 4096 шагов видит действия шага 0", t_action_indexing)

# ---------- 9. Orthogonal gain=1.0 на actor head -> неравномерная стартовая политика ----------
def t_init_entropy():
    from rl.actor_critic import ActorCritic
    dev = torch.device("cpu")
    torch.manual_seed(42)
    ents = []
    for _ in range(20):
        m = ActorCritic(obs_size=209, n_actions=45, hidden_sizes=[256,256], device=dev)
        with torch.no_grad():
            logits, _ = m(torch.randn(64, 209))
            ents.append(torch.distributions.Categorical(logits=logits).entropy().mean().item())
    max_ent = math.log(45)
    mean_ent = sum(ents)/len(ents)
    print(f"   max entropy(45) = {max_ent:.3f}; средняя стартовая entropy = {mean_ent:.3f} "
          f"({mean_ent/max_ent*100:.0f}% от максимума; при gain=0.01 было бы ~100%)")
    assert mean_ent < max_ent * 0.97, "энтропия почти максимальная"
check("actor head с gain=1.0: стартовая политика далека от равномерной (низкая энтропия с шага 0)", t_init_entropy)

# ---------- 10. approx_kl = |(old-new).mean| занижает KL ----------
def t_kl():
    torch.manual_seed(0)
    n = 8192
    d = torch.randn(n) * 0.2          # симметричные изменения log-prob
    old = torch.randn(n); new = old + d
    their = (old - new).mean().abs().item()
    k3 = (((new-old).exp() - 1) - (new-old)).mean().item()  # стандартная k3-оценка
    print(f"   их метрика |mean(old-new)| = {their:.5f}  vs  истинная KL-оценка k3 = {k3:.5f}")
    assert their < k3 * 0.3, "не воспроизвелось"
check("approx_kl: |(old-new).mean()| ~ 0 из-за компенсации, реальная KL большая", t_kl)

# ---------- 11. Config: нет поля loop_detection -> LoopDetector мёртв ----------
def t_loop_dead():
    from rl.config import Config
    cfg = Config()
    loop_config = getattr(cfg, "loop_detection", {})
    assert not loop_config, "loop_detection появился"
    print("   getattr(cfg,'loop_detection',{}) == {} -> LoopDetector(...) не создаётся; "
          "детекция петель и AutoBoost энтропии — мёртвый код")
check("LoopDetector никогда не включается: в Config нет поля loop_detection", t_loop_dead)

# ---------- 12. argparse: '--compile False' из их же гайда ----------
def t_compile_flag():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--compile", action="store_true")
    try:
        a = p.parse_args(["--compile", "False"])
        raise AssertionError(f"распарсилось: compile={a.compile}")
    except SystemExit:
        print("   argparse: error: unrecognized arguments: False — команда из SUMMARY.md не работает; "
              "правильно: просто не передавать --compile")
check("SUMMARY.md/OPTIMIZATION_GUIDE: рекомендация '--compile False' — нерабочая команда", t_compile_flag)

# ---------- 13. eval_score_weights: масштаб компонентов ----------
def t_eval_weights():
    w1,w2,w3,w4 = (0.4, 3.0, 0.2, 0.0001)
    days, bases, people, ret = 10000, 50, 5000, 1_000_000
    parts = dict(days=days*w1, bases=bases*w2, people=people*w3, ret=max(0,ret)*w4)
    total = sum(parts.values())
    print("   вклад в score: " + ", ".join(f"{k}={v:.1f} ({v/total*100:.1f}%)" for k,v in parts.items()))
    assert parts["days"]/total > 0.5
check("eval_score_weights (0.4,3.0,0.2,0.0001): выживание (days) доминирует, bases почти не влияют", t_eval_weights)

# ---------- 14. top_actions: берутся ПЕРВЫЕ 5 действий, а не топ-5 ----------
def t_top_actions():
    action_names = ["DAY","WEEK","BUILD_HOUSE","BUILD_FARM","BUILD_ROAD","X","Y"]
    action_counts = [1, 2, 900, 50, 40, 5, 2]   # BUILD_HOUSE — явный топ
    top = {}
    for i, count in enumerate(action_counts[:5]):   # логика из async_trainer.train()
        top[action_names[i]] = round(count/max(sum(action_counts),1)*100, 2)
    print(f"   реальный топ: BUILD_HOUSE=90%; их 'top_actions': {top}")
    assert "BUILD_FARM" in top and top["BUILD_HOUSE"] > top["BUILD_FARM"]
check("metrics.top_actions показывает первые 5 действий по порядку, а не самые частые", t_top_actions)

print("\n===== ИТОГ =====")
for name, res in RESULTS:
    print(f"{'✔' if res=='CONFIRMED' else '✘'} {name}: {res}")
