# -*- coding: utf-8 -*-
"""Проверка, что фиксы из FIXES_P0_P3.md реально работают (на rl_fixed)."""
import math, torch
import rl_fixed.config as cfgmod
from rl_fixed.config import Config, RewardConfig
from rl_fixed.actor_critic import ActorCritic
from rl_fixed.rollout_buffer import RolloutBuffer
from rl_fixed.ppo import PPO

ok = 0; fail = 0
def report(name, passed, detail=""):
    global ok, fail
    ok, fail = ok + passed, fail + (not passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))

# 1. B1: round-trip идентичен
d = RewardConfig()
rt = RewardConfig.from_dict(d.to_dict())
same = all(getattr(d, f) == getattr(rt, f) for f in d.to_dict())
report("P0-1 B1: from_dict(to_dict(x)) == x (все 29 полей)", same)
rt2 = RewardConfig.from_dict({})
same2 = all(getattr(d, k) == getattr(rt2, k) for k in d.to_dict())
report("P0-1 B1: from_dict({}) == дефолты датакласса", same2,
       f"daily_income={rt2.daily_income}, clip=±{rt2.clip_reward_max}, milestone={rt2.milestone_base_bonus}")
rt3 = RewardConfig.from_dict({"daily_income": 0.7, "junk_key": 1})
report("P0-1 B1: частичный JSON переопределяет только заданное", rt3.daily_income == 0.7 and rt3.novelty == d.novelty)

# 2. A3: load_from_file работает
import json
c = Config()
json.dump({"seed": 123, "reward": {"novelty": 9.0}}, open("/tmp/cfg2.json", "w"))
ret = c.load_from_file("/tmp/cfg2.json")
report("A3: load_from_file возвращает self, seed и reward применены",
       ret is c and c.seed == 123 and c.reward.novelty == 9.0 and c.reward.daily_income == d.daily_income)

# общий стенд
dev = torch.device("cpu")
def make_ppo(target_kl=0.0, total_steps=0):
    torch.manual_seed(0)
    m = ActorCritic(obs_size=16, n_actions=5, hidden_sizes=[16,16], device=dev)
    b = RolloutBuffer(8, 4, 16, 5, 0.99, 0.95, dev)
    ppo = PPO(m, b, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
              vf_coef=0.5, max_grad_norm=0.5, n_epochs=3, batch_size=16, use_amp=False,
              device=dev, total_training_steps=total_steps, target_kl=target_kl)
    for _ in range(8):
        obs = torch.randn(4, 16)
        out = ppo.collect_step(obs)
        b.add(obs, out["action"], torch.randn(4), out["log_prob"], out["value"],
              torch.zeros(4, dtype=torch.bool))
    return ppo

# 3. A2-фикс в конструкторе + update работает
ppo = make_ppo()
report("P0-2: is_hybrid задаётся конструктором", ppo.is_hybrid is False)
stats = ppo.update(torch.rand(4), torch.zeros(4, dtype=torch.bool))
report("P0-2 B6: approx_kl конечный и >= 0 (k3)", math.isfinite(stats["approx_kl"]) and stats["approx_kl"] >= 0,
       f"approx_kl={stats['approx_kl']:.6f}")

# 4. A5+A6: пустые маски больше не крашат, маски по умолчанию — единицы
ppo2 = make_ppo()
b = ppo2.buffer
report("P2-7 A6: маски по умолчанию — все True (dtype bool)",
       bool(b.action_masks.all()) and b.action_masks.dtype == torch.bool)
out = ppo2.collect_step(torch.randn(4,16), action_masks=torch.zeros(4,5))   # ВСЁ заблокировано
report("P2-8 A5: полностью пустая маска -> нет краша/NaN",
       torch.isfinite(out["log_prob"]).all().item())
# update при env без масок (буфер уже заполнен в make_ppo, маски в нём — ones)
st2 = ppo2.update(torch.rand(4), torch.zeros(4,dtype=torch.bool))
report("P2-7 A6: update без env-масок -> конечный KL (был inf на мусоре)",
       math.isfinite(st2["approx_kl"]), f"approx_kl={st2['approx_kl']:.6f}")

# 5. B7: LR decay заработал
ppo3 = make_ppo(total_steps=100)
lr0 = ppo3.optimizer.param_groups[0]["lr"]
for _ in range(100): ppo3.scheduler.step()
lr1 = ppo3.optimizer.param_groups[0]["lr"]
# lr_lambda: старт 1.1 (110% базы), финиш 0.1 (10% базы) => от 3.3e-4 к 3.0e-5
report("P0-2 B7: LR decays 110%->10% базы при total_training_steps",
       abs(lr0 - 3.3e-4) < 1e-9 and abs(lr1 - 3e-5) < 1e-9,
       f"{lr0:.6f} -> {lr1:.6f} (формулу старта см. в FIXES P0-2e)")

# 6. KL early-stop: при крошечном target_kl эпохи обрываются после 1-го батча
ppo4 = make_ppo(target_kl=1e-12)
calls = {"n": 0}
orig = ppo4.optimizer.step
ppo4.optimizer.step = lambda *a, **k: (calls.__setitem__("n", calls["n"]+1), orig(*a, **k))[1]
ppo4.update(torch.rand(4), torch.zeros(4,dtype=torch.bool))
# первый мини-батч: new==old -> KL=0, стоп невозможен; ожидем остановку на 2-м из 6
report("P0-2: KL early-stop обрывает обучение (2 шага вместо 6; KL 1-го батча = 0)",
       calls["n"] == 2, f"optimizer.step вызван {calls['n']} раз(а) из 6 возможных")

# 7. D1: срез текущего шага возвращает правильные действия
b5 = RolloutBuffer(5, 2, 4, 6, 0.99, 0.95, dev)
for t in range(5):
    b5.add(torch.randn(2,4), torch.full((2,), t+1), torch.randn(2),
           torch.randn(2), torch.randn(2), torch.zeros(2,dtype=torch.bool))
seen = []
for pos in range(5):
    step_actions = b5.actions[pos*2:(pos+1)*2].cpu().numpy()
    seen.append([int(x) for x in step_actions])
report("P1-4 D1: срез [pos*n_envs:(pos+1)*n_envs] даёт действия ТЕКУЩЕГО шага",
       seen == [[1,1],[2,2],[3,3],[4,4],[5,5]], f"seen={seen}")

print(f"\nИТОГ: {ok} PASS, {fail} FAIL")
