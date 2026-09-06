"""
FIX_06: Мелкие исправления — model.train(), debug stubs
========================================================

ПРОБЛЕМЫ:
  1. model.train() вызывается после update() — избыточно (collect_step ставит eval)
  2. _queue.Empty импортируется неправильно в async_trainer.py (строка 264)

ИСПРАВЛЯЕТ: rl/ppo.py, rl/async_trainer.py
"""


# ============================================================
# ИЗМЕНЕНИЕ 1: Убрать model.train() из update() (ppo.py)
# ============================================================

# СТРОКА 160-161 (БЫЛО):
def update(self, last_value, last_done):
    # ... PPO update ...
    self.model.train()     # <-- избыточно
    self.buffer.reset()
    return stats

# СТРОКА 160-161 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def update(self, last_value, last_done):
    # ... PPO update ...
    # model.train() is NOT needed here:
    # - collect_step() sets model.eval() before inference
    # - The update itself runs forward passes in train mode (inside autocast)
    self.buffer.reset()
    return stats

# ПРИМЕЧАНИЕ: Если в update() forward pass идёт через _compute_loss_components(),
# модель уже в train mode (по умолчанию nn.Module). Autocast не меняет mode.
#collect_step() явно ставит eval(). Поэтому model.train() после update()
# технически нужен, чтобы следующий _compute_loss_components() был в train mode.
# НО: nn.Module по умолчанию в train mode, и collect_step() ставит eval()
# только для inference. Поэтому model.train() нужен только если где-то
# явно вызван model.eval() и не возвращён обратно.
#
# ВЕРДИКТ: model.train() нужен, но его можно перенести в начало update():
def update_FIXED(self, last_value, last_done):
    self.model.train()     # <-- в начало, для ясности
    self.buffer.compute_gae(last_value, last_done)
    # ... PPO update ...
    self.buffer.reset()
    return stats


# ============================================================
# ИЗМЕНЕНИЕ 2: Fix _queue import (async_trainer.py)
# ============================================================

# СТРОКА 264 (БЫЛО):
except _queue.Empty:
    break

# СТРОКА 264 (ПОСЛЕ ИСПРАВЛЕНИЯ):
except queue.Empty:
    break

# Причина: в async_trainer.py импортирован queue как модуль (строка 5),
# но в _process_commands используется _queue.Empty.
# Нужно либо: import queue as _queue (и исправить все queue.Queue),
# либо: except queue.Empty (как в оригинале).
#
# ПРОВЕРКА: строка 5 — import queue (не as _queue).
# Строка 74 — queue.Queue(maxsize=...) — использует queue.
# Строка 264 — _queue.Empty — ОШИБКА! Нет import _queue.
#
# ИСПРАВЛЕНИЕ: заменить _queue.Empty на queue.Empty


# ============================================================
# ИЗМЕНЕНИЕ 3: Добавить debug информацию приAMP fallback (ppo.py)
# ============================================================

# В __init__ PPO (строка 47-48):
self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp_dtype == "float16"))

# ДОБАВИТЬ после scaler:
if use_amp:
    if self.device.type == "cuda":
        if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print(f"[PPO WARNING] bfloat16 requested but GPU "
                  f"({torch.cuda.get_device_name()}) does not support it. "
                  f"AMP autocast may fall back to float32.")
    else:
        print(f"[PPO WARNING] AMP enabled but device={self.device.type}. "
              f"AMP only accelerates on CUDA GPUs.")
