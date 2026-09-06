"""
FIX_01: Исправление порядка obs в буфере переходов
====================================================

ПРОБЛЕМА:
  В env_manager.py:collect_step() буфер получает new_obs (s') вместо obs (s).
  PPO обучается на паре (s', a) вместо (s, a) — policy gradient полностью искажён.

  Было (НЕПРАВИЛЬНО):
    obs = current_obs                # s_t
    action = policy(obs)             # a_t ~ π(a|s_t)
    new_obs, reward, done = step(action)   # s_{t+1}
    buffer.add(obs=new_obs, action=action) # ХРАНИТ s' ВМЕСТО s!

  Должно быть (ПРАВИЛЬНО):
    buffer.add(obs=obs, action=action)     # ХРАНИТ s_t

ИСПРАВЛЯЕТ: rl/env_manager.py
"""


# ============================================================
# ИЗМЕНЕНИЕ В rl/env_manager.py
# ============================================================

# СТРОКА 218-241 (БЫЛО):
def collect_step_BROKEN(self, obs) -> tuple:
    """One full step: policy -> env step -> buffer add."""
    action_masks_np = getattr(self.env, "action_masks", None)
    if action_masks_np is not None:
        action_masks = torch.from_numpy(action_masks_np).to(self.device)
    else:
        action_masks = None

    if self.obs_mode == "hybrid":
        flat, minimap = obs
        self._last_flat = flat
        policy_out = self.ppo.collect_step(flat, minimap, action_masks=action_masks)
    else:
        policy_out = self.ppo.collect_step(obs, action_masks=action_masks)
    action_gpu = policy_out["action"]
    action_np = action_gpu.cpu().numpy().astype(np.int32)

    env_out = self.step(action_np)
    new_obs = env_out["obs"]
    rewards = env_out["rewards"]
    dones = env_out["dones"]
    terminated = env_out["terminated"]
    infos = env_out["infos"]

    if self.obs_mode == "hybrid":
        flat, minimap = obs                    # <-- s_t
        self.buffer.add(
            obs=minimap,                       # <-- s'_t (ОШИБКА!)
            action=action_gpu,
            reward=rewards,
            log_prob=policy_out["log_prob"],
            value=policy_out["value"],
            done=dones,
            terminated=terminated,
            flat=flat,                         # <-- s_t flat (правильно)
            action_masks=action_masks,
        )
    else:
        self.buffer.add(
            obs=new_obs,                       # <-- s'_t (ОШИБКА!)
            action=action_gpu,
            reward=rewards,
            log_prob=policy_out["log_prob"],
            value=policy_out["value"],
            done=dones,
            terminated=terminated,
            action_masks=action_masks,
        )

    return new_obs, infos


# СТРОКА 218-241 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def collect_step_FIXED(self, obs) -> tuple:
    """One full step: policy -> env step -> buffer add.

    Buffer stores (s_t, a_t, r_t, done_t, V(s_t)) — the CURRENT transition.
    """
    action_masks_np = getattr(self.env, "action_masks", None)
    if action_masks_np is not None:
        action_masks = torch.from_numpy(action_masks_np).to(self.device)
    else:
        action_masks = None

    if self.obs_mode == "hybrid":
        flat, minimap = obs
        self._last_flat = flat
        policy_out = self.ppo.collect_step(flat, minimap, action_masks=action_masks)
    else:
        policy_out = self.ppo.collect_step(obs, action_masks=action_masks)
    action_gpu = policy_out["action"]
    action_np = action_gpu.cpu().numpy().astype(np.int32)

    env_out = self.step(action_np)
    new_obs = env_out["obs"]
    rewards = env_out["rewards"]
    dones = env_out["dones"]
    terminated = env_out["terminated"]
    infos = env_out["infos"]

    if self.obs_mode == "hybrid":
        flat, minimap = obs                    # s_t
        self.buffer.add(
            obs=minimap,                       # s_t minimap (FIXED)
            action=action_gpu,
            reward=rewards,
            log_prob=policy_out["log_prob"],
            value=policy_out["value"],
            done=dones,
            terminated=terminated,
            flat=flat,                         # s_t flat
            action_masks=action_masks,
        )
    else:
        self.buffer.add(
            obs=obs,                           # s_t (FIXED — was new_obs)
            action=action_gpu,
            reward=rewards,
            log_prob=policy_out["log_prob"],
            value=policy_out["value"],
            done=dones,
            terminated=terminated,
            action_masks=action_masks,
        )

    return new_obs, infos


# ============================================================
# ПРИМЕНЕНИЕ К ФАЙЛУ
# ============================================================
# Открыть rl/env_manager.py
# Найти collect_step() (строка ~193)
# В ветке else (строка ~232):
#   Заменить: obs=new_obs  →  obs=obs
#
# В ветке if self.obs_mode == "hybrid" (строка ~220):
#   obs=minimap остаётся — это s_t minimap (правильно, т.к. obs = (flat, minimap))
#   flat=flat остаётся — это s_t flat (правильно)
#
# ИТОГО: единственная замена — строка ~232:
#   БЫЛО:  obs=new_obs,
#   СТАЛО: obs=obs,
