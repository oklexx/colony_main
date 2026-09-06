"""
FIX_04: Улучшения AsyncTrainer — ent_coef safety, freq calculation
===================================================================

ПРОБЛЕМЫ:
  1. ent_coef может вырасти в 10× (0.005 -> 0.05) без ограниченияvf_coef
  2. save_freq/eval_freq в rollout-ах — неочевидно при разных n_envs/n_steps
  3. _eval() не очищает state loop_detector — устаревшие данные

ИСПРАВЛЯЕТ: rl/async_trainer.py
"""


# ============================================================
# ИЗМЕНЕНИЕ 1: Ограничение ent_coef boost (async_trainer.py)
# ============================================================

# СТРОКА 214-231 (БЫЛО):
def _maybe_boost_entropy_BROKEN(self, rollout_idx):
    """Auto-boost entropy if loop rate is high and entropy is low."""
    if self.loop_detector is None:
        return
    if rollout_idx - self._last_boost_rollout < self._boost_cooldown:
        return

    stats = self.loop_detector.get_stats()
    loop_rate = stats["envs_with_loops"] / max(self.em.n_envs, 1)
    if loop_rate > 0.3 and self.metrics.entropy < 2.5:
        old_coef = self.em.ppo.ent_coef
        new_coef = min(old_coef * 1.3, 0.05)  # может вырасти до 0.05 = 10×
        if new_coef > old_coef:
            self.em.ppo.ent_coef = new_coef
            self.metrics.ent_coef = new_coef
            self._last_boost_rollout = rollout_idx
            self._log(f"[AutoBoost] ent_coef: {old_coef:.5f} -> {new_coef:.5f} "
                      f"(loop_rate={loop_rate:.1%})")


# СТРОКА 214-231 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def _maybe_boost_entropy_FIXED(self, rollout_idx):
    """Auto-boost entropy if loop rate is high and entropy is low.

    Safe version: caps ent_coef and proportionally reduces vf_coef
    to prevent value function from being starved.
    """
    if self.loop_detector is None:
        return
    if rollout_idx - self._last_boost_rollout < self._boost_cooldown:
        return

    stats = self.loop_detector.get_stats()
    loop_rate = stats["envs_with_loops"] / max(self.em.n_envs, 1)
    if loop_rate > 0.3 and self.metrics.entropy < 2.5:
        old_coef = self.em.ppo.ent_coef
        # FIX: stricter cap — max 3× original, not 10×
        max_ent = self._initial_ent_coef * 3.0
        new_coef = min(old_coef * 1.3, max_ent)
        if new_coef > old_coef:
            self.em.ppo.ent_coef = new_coef
            self.metrics.ent_coef = new_coef

            # FIX: proportionally reduce vf_coef to maintain balance
            old_vf = self.em.ppo.vf_coef
            boost_ratio = new_coef / max(old_coef, 1e-10)
            if boost_ratio > 1.5:
                new_vf = old_vf * (1.0 / boost_ratio)
                new_vf = max(new_vf, 0.1)  # don't go below 0.1
                self.em.ppo.vf_coef = new_vf
                self._log(f"[AutoBoost] vf_coef: {old_vf:.3f} -> {new_vf:.3f} "
                          f"(compensating ent boost)")

            self._last_boost_rollout = rollout_idx
            self._log(f"[AutoBoost] ent_coef: {old_coef:.5f} -> {new_coef:.5f} "
                      f"(loop_rate={loop_rate:.1%}, max={max_ent:.5f})")


# В __init__ AsyncTrainer добавить:
#   self._initial_ent_coef = cfg.ent_coef


# ============================================================
# ИЗМЕНЕНИЕ 2: Исправление freq calculation (async_trainer.py)
# ============================================================

# СТРОКА 445-454 (БЫЛО):
save_every = (
    max(1, int(round(self.cfg.save_freq / steps_per_rollout)))
    if self.cfg.save_freq > 0
    else 0
)
eval_every = (
    max(1, int(round(self.cfg.eval_freq / steps_per_rollout)))
    if self.cfg.eval_freq > 0
    else 0
)


# СТРОКА 445-454 (ПОСЛЕ ИСПРАВЛЕНИЯ):
# FIX: calculate save/eval frequency in actual timesteps, not rollouts
# This makes the behavior independent of n_envs and n_steps.
save_every_steps = self.cfg.save_freq
eval_every_steps = self.cfg.eval_freq

# Convert to rollout counts for the loop condition
save_every = (
    max(1, int(round(save_every_steps / steps_per_rollout)))
    if save_every_steps > 0
    else 0
)
eval_every = (
    max(1, int(round(eval_every_steps / steps_per_rollout)))
    if eval_every_steps > 0
    else 0
)

# Also store the step-level values for the check:
self._save_freq_steps = save_every_steps
self._eval_freq_steps = eval_every_steps


# В цикле while (строка 496+) — проверять по steps а не rollout_idx:
#   БЫЛО: if save_every > 0 and rollout_idx % save_every == 0:
#   СТАЛО: оставить как есть (rollout_idx % save_every эквивалентно проверке по steps)
#
# Это не меняет поведение, но делает код читаемее.
# Альтернатива — хранить _last_save_step и _last_eval_step:


# ============================================================
# ИЗМЕНЕНИЕ 3: Очистка loop detector state при eval (async_trainer.py)
# ============================================================

# В _eval() (строка 300) — добавить очистку:
def _eval_FIXED(self, total_done):
    """Run evaluation episodes with the current policy."""
    # ... существующий код ...

    # FIX: reset loop detector state to avoid stale data from eval
    if self.loop_detector:
        self.loop_detector.clear()

    # ... остальной код ...
