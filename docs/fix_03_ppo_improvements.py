"""
FIX_03: Улучшения PPO — LR scheduler, AMP inference, compile guard
==================================================================

ПРОБЛЕМЫ:
  1. Нет LR scheduler — constant learning rate на всём обучении
  2. collect_step() не использует AMP — замедляет сбор rollout
  3. torch.compile без проверки CUDA availability

ИСПРАВЛЯЕТ: rl/ppo.py, rl/config.py
"""


# ============================================================
# ИЗМЕНЕНИЕ 1: LR Scheduler (ppo.py)
# ============================================================

# В __init__ (строка 15-61) добавить scheduler:

"""
class PPO:
    def __init__(
        self,
        model, buffer,
        lr=3e-4,
        # ... существующие параметры ...
        use_amp=True,
        amp_dtype="bfloat16",
        torch_compile=False,
        device=None,
        # НОВЫЕ параметры:
        lr_warmup_steps=0,         # количество шагов warmup
        lr_decay=True,             # использовать ли decay
        total_training_steps=0,    # общее число шагов для decay
    ):
        # ... существующий код ...

        # LR Scheduler: linear warmup + cosine decay
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay = lr_decay
        self._current_step = 0
        self._total_training_steps = total_training_steps

        if lr_warmup_steps > 0 or lr_decay:
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(step):
                # Warmup phase
                if step < lr_warmup_steps:
                    return float(step) / max(1, lr_warmup_steps)
                # Decay phase: cosine from 1.0 to 0.1
                if lr_decay and total_training_steps > 0:
                    progress = float(step - lr_warmup_steps) / max(
                        1, total_training_steps - lr_warmup_steps)
                    progress = min(progress, 1.0)
                    return 0.1 + 0.5 * (1.0 + math.cos(math.pi * progress))
                return 1.0
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
"""

# В update() (строка 94) — добавить шаг scheduler:

"""
    def update(self, last_value, last_done):
        self.buffer.compute_gae(last_value, last_done)
        # ... PPO update loop ...

        # После optimizer.step() в mini-batch loop:
        if self.scheduler is not None:
            self.scheduler.step()
            self._current_step += 1

        # ... остальной код ...
        return stats
"""


# ============================================================
# ИЗМЕНЕНИЕ 2: AMP в collect_step() (ppo.py)
# ============================================================

# СТРОКА 63-92 (БЫЛО):
def collect_step_BROKEN(self, flat, minimap=None, action_masks=None):
    self.model.eval()
    with torch.no_grad():
        # AMP не используется!
        if self.is_hybrid:
            logits, values = self.model(flat, minimap)
        else:
            logits, values = self.model(flat)
        # ...


# СТРОКА 63-92 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def collect_step_FIXED(self, flat, minimap=None, action_masks=None):
    """Get action, log_prob, value for current obs (no grad)."""
    self.model.eval()
    with torch.no_grad():
        if self.use_amp:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
            ):
                if self.is_hybrid:
                    logits, values = self.model(flat, minimap)
                else:
                    logits, values = self.model(flat)
        else:
            if self.is_hybrid:
                logits, values = self.model(flat, minimap)
            else:
                logits, values = self.model(flat)

        if action_masks is not None:
            logits = logits.masked_fill(action_masks == 0, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        values = values.squeeze(-1)

    n_actions = self.model.n_actions
    action = action.clamp(0, n_actions - 1)
    return {
        "action": action,
        "log_prob": log_prob,
        "value": values,
    }


# ============================================================
# ИЗМЕНЕНИЕ 3: torch.compile guard (ppo.py)
# ============================================================

# СТРОКА 57-61 (БЫЛО):
if torch_compile:
    self.model = torch.compile(model, mode="reduce-overhead")
    self._compiled = True
else:
    self._compiled = False


# СТРОКА 57-61 (ПОСЛЕ ИСПРАВЛЕНИЯ):
self._compiled = False
if torch_compile:
    if self.device.type == "cuda" and torch.cuda.is_available():
        try:
            self.model = torch.compile(model, mode="reduce-overhead")
            self._compiled = True
        except Exception as e:
            print(f"[PPO] torch.compile failed: {e}. Falling back to eager.")
            self._compiled = False
    else:
        print("[PPO] torch.compile requires CUDA. Disabling.")


# ============================================================
# ИЗМЕНЕНИЕ 4: Добавить math import (ppo.py)
# ============================================================

# СТРОКА 1-6 — добавить import math:
# import math  # для LR scheduler cosine decay
