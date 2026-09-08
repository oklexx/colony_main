# Комплект фиксов P0→P3 — Sakhalin Colony RL

Все якоря — по номерам строк живого кода, предоставленным владельцем. Порядок применения: **P0 → P1 → P2 → P3**, после каждого уровня — smoke-тест (раздел «Проверка»). Фиксы независимы друг от друга, кроме: **D1 использует deque из B3** (можно применить D1 с обычным list, см. примечание) и **target_kl в P0-2 требует поля в Config**.

---

## P0 — качество обучения (тихие искажения)

### P0-1 · B1 — единые дефолты наград (`rl/config.py:85-109`)

Заменить тело `RewardConfig.from_dict` целиком:

```python
@classmethod
def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
    """Дефолты берутся ТОЛЬКО из датакласса; JSON лишь переопределяет явно заданное."""
    base = cls()
    for k, v in d.items():
        if not hasattr(base, k):
            continue
        if k == "idle_build_threshold_days":
            v = int(v)
        setattr(base, k, v)
    return base
```

Эффект: исчезают расхождения `daily_income` 1.0/0.3, `survival_bonus` 0.0/0.1, `survival_coeff` 0.01/0.001, `milestone_base_bonus` 30/10, клип ±50/±10.
⚠️ Если ваши текущие запуски опирались на «JSON-дефолты» (например, ежедневный доход 0.3), после фикса поведение изменится на датаклассовое — зафиксируйте полный JSON наград в репо и передавайте его целиком через `--reward-config`.

### P0-2 · B6+B7 — честный KL, KL-early-stop, рабочий decay LR

**(a) `rl/config.py`** — добавить поле в `Config` (рядом с `max_grad_norm`):
```python
target_kl: float = 0.02          # 0 = early-stop по KL выключен
```
и добавить `"target_kl"` в кортеж ключей `to_dict()` и в `from_dict()` (`target_kl=d.get("target_kl", 0.02)`).

**(b) `rl/ppo.py`, `__init__`** — новый параметр:
```python
def __init__(self, ..., total_training_steps: int = 0, target_kl: float = 0.0):
    ...
    self.target_kl = target_kl
```

**(c) `rl/ppo.py:255`** — заменить формулу KL в `_compute_loss_components`:
```python
with torch.no_grad():
    log_ratio = (new_log_probs - old_log_probs).clamp(-10.0, 10.0)
    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()   # k3-оценка, всегда >= 0
```

**(d) `rl/ppo.py`, `update()`** — ранняя остановка по KL (порог ×1.5 — стандарт SB3):
```python
kl_stop = False
for _ in range(self.n_epochs):
    if kl_stop:
        break
    for batch in self.buffer.get_batches(self.batch_size):
        ...  # без изменений до накопления метрик
        n_batches += 1
        if self.target_kl > 0 and approx_kl.item() > 1.5 * self.target_kl:
            kl_stop = True
            break
```

**(e) `rl/env_manager.py:122-138`** — передать горизонт decay и target_kl:
```python
samples_per_rollout = cfg.n_steps * cfg.n_envs
n_updates = max(1, cfg.total_timesteps // samples_per_rollout)
batches_per_update = max(1, samples_per_rollout // cfg.batch_size)
opt_steps = n_updates * batches_per_update * cfg.n_epochs

self.ppo = PPO(
    ...                                    # существующие аргументы без изменений
    device=device,
    lr_decay=True,
    total_training_steps=opt_steps,        # <-- раньше не передавалось => scheduler был no-op
    target_kl=cfg.target_kl,
)
```
⚠️ **Нюанс `lr_lambda` (выявлен тестом):** существующая формула `0.1 + 0.5*(1+cos(pi*progress))` на старте даёт **1.1** (110% базового LR) и в конце 0.1 (10%). Раньше этого не было видно (scheduler был no-op). Чтобы стартовать ровно с базового LR и уйти к 10%, заменить в `ppo.py` на:
```python
return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))   # [1.0 ... 0.1]
```

⚠️ **Нюанс KL-стопа (выявлен тестом):** в первом мини-батче каждого update `new == old` → KL = 0, поэтому самое раннее срабатывание — на 2-м батче. Это нормально, не «починять».

**(f) `train.py`** — CLI-флаг: `p.add_argument("--target-kl", type=float, default=_d.target_kl)` и `target_kl=args.target_kl` в `Config(...)`.

**(g) Рекомендация по дефолтам** (можно только через CLI для baseline-прогона): `--n-epochs 4 --ent-coef 0.015 --gae-lambda 0.95 --amp off`.

### P0-3 · B2+F3 — eval с нормализацией и адекватный score

**(a) B2, минимальный фикс — `rl/async_trainer.py`, блок сохранения чекпойнта (после строки 618):**
```python
self.em.env.venv.save_normalization(norm_path)
# ДОБАВИТЬ СЛЕДУЮЩУЮ СТРОКУ — eval (строки 334-335) ищет именно это имя:
self.em.env.venv.save_normalization(str(save_dir / "normalization.json"))
```
Дополнительно в `_eval` после строки 335 — страховка от тихой деградации:
```python
if norm_str is None:
    self._log("[Eval] WARNING: normalization.json не найден — eval БЕЗ нормализации obs!")
```

**(b) F3 — `rl/async_trainer.py:376-377`,** нормировка компонентов score:
```python
score = ((days_agg / 1000.0) * w1 + bases_agg * w2
         + (people_agg / 100.0) * w3 + max(0.0, return_agg) * w4)
```
При тех же весах (0.4, 3.0, 0.2, 1e-4) и типичных величинах: days≈4 (было 4000), bases≈150, people≈10 — базы начинают решать. ⚠️ `best_score` до и после правки несопоставимы — старые `best_model.meta.json` не сравнивать с новыми.

---

## P1 — нагрузка на систему

### P1-4 · D1 — убрать ~1 ГБ D2H на rollout (`rl/async_trainer.py:168-172` + чтение terminated)

В `_collect_rollout` заменить пошаговый блок копирования действий:

**Было (копирует весь буфер [32768] каждый шаг):**
```python
actions_tensor = self.em.buffer.actions
if hasattr(actions_tensor, 'dim') and actions_tensor.dim() > 0:
    actions_np = actions_tensor.cpu().numpy()          # 256 КБ × 4096 шагов ≈ 1 ГБ!
    if actions_np.ndim == 2:
        step_actions = actions_np[pos, :]
    elif actions_np.ndim == 1:
        step_actions = actions_np                      # + баг B3: всегда действия шага 0
```

**Стало (64 байта на шаг, заодно чинит индексацию B3):**
```python
pos = self.em.buffer.pos - 1
step_actions = (self.em.buffer.actions[pos * n_envs:(pos + 1) * n_envs]
                .cpu().numpy())                        # только текущий шаг, 8 int64
for env_idx in range(n_envs):
    action_idx = int(step_actions[env_idx])
    action_name = (action_names[action_idx]
                   if action_idx < len(action_names) else f"ACTION_{action_idx}")
    self._action_history.append((env_idx, action_name))   # deque после B3; до B3 — list
```

Чтение `terminated` — убрать из цикла, выполнить **один раз после цикла**:
```python
last_pos = self.em.buffer.pos - 1
terminated = (self.em.buffer.terminated[last_pos * n_envs:(last_pos + 1) * n_envs]
              .cpu().numpy())
last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)
```

И `_action_history` → `collections.deque(maxlen=1000)` в `__init__` (убирает O(50k)-срез каждый шаг; `_calculate_action_distribution` уже берёт `[-1000:]`).

Опционально (нулевая цена вместо 64 Б/шаг): возвращать `action_np` третьим элементом из `EnvManager.collect_step` и брать его в трейнере. ⚠️ Тогда поправить и обёртку `_collect_with_log` в `train.py` (возвращает кортеж).

### P1-5 · D2 — eval реже и легче (без правок кода)
```bash
--eval-freq 500000 --eval-episodes 10 --eval-seeds 42 43 44
```
Код-уровень (опционально, позже): создавать окружение оценщика один раз и переиспользовать; передавать state_dict в память вместо `_eval_temp.pt`.

---

## P2 — дешёвые страховки

### P2-6 · A4 — `rl/env_manager.py:350`
```python
# было:  schedule = [(t, s+1) for t, s in enumerate(default_schedule)]
schedule = list(default_schedule)
```

### P2-7 · A6 — маски из `ones` вместо `empty` (`rl/rollout_buffer.py:45` и `197`)
```python
# было:  self.action_masks = torch.empty(total, n_actions, dtype=torch.float32, device=device)
self.action_masks = torch.ones(total, n_actions, dtype=torch.bool, device=device)
```
(в обоих классах). Присваивание float-масок из env в bool-тензор приводит nonzero→True автоматически; сравнение `action_masks == 0` в ppo.py работает и с bool. Память: 5.9 МБ → 1.5 МБ.

### P2-8 · A5 — fallback при полностью заблокированных действиях (`rl/ppo.py`, оба места маскирования)
Вынести в хелпер и использовать в `collect_step` и `_compute_loss_components`:
```python
def _apply_action_masks(logits: torch.Tensor, masks) -> torch.Tensor:
    if masks is None:
        return logits
    m = masks.bool()
    all_blocked = ~m.any(dim=-1, keepdim=True)          # строки, где недоступно всё
    m = torch.where(all_blocked, torch.ones_like(m), m) # fallback: разрешить всё
    return logits.masked_fill(~m, float("-inf"))
```

### P2-9 · A7 — симметрия save/load + полное состояние (`rl/ppo.py:259-293`)
В `save()` добавить в словарь:
```python
"scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
"ent_coef": self.ent_coef,
"vf_coef": self.vf_coef,
"opt_step": self._current_step,
```
В `load()`:
```python
model = getattr(self.model, "_orig_mod", self.model)    # torch.compile-обёртка
model.load_state_dict(ckpt["model_state"])
self.optimizer.load_state_dict(ckpt["optimizer_state"])
if self.scheduler is not None and ckpt.get("scheduler_state"):
    self.scheduler.load_state_dict(ckpt["scheduler_state"])
self.ent_coef = ckpt.get("ent_coef", self.ent_coef)
self.vf_coef = ckpt.get("vf_coef", self.vf_coef)
self._current_step = ckpt.get("opt_step", 0)
```

### P2-10 · A3 — `load_from_file` (`rl/config.py`, конец класса Config)
```python
# было:  self.__dict__.update({k: v for k, v in data.items() if k != "reward"})
#        ...
#        return cfg          # NameError: cfg не определён
import dataclasses
allowed = {f.name for f in dataclasses.fields(Config)} - {"reward"}
self.__dict__.update({k: v for k, v in data.items() if k in allowed})
if "reward" in data:
    self.reward = RewardConfig.from_dict(data["reward"])
return self
```

### P2-11 · B8 — early stopping на отрицательных score
```python
# было: if self.best_score is not None and self.best_score > (self._es_best_score or 0):
if self.best_score is not None and (self._es_best_score is None
                                    or self.best_score > self._es_best_score):
```

### P2-12 · C5 — `train.py`, выбор устройства
```python
# было: args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
```

---

## P3 — loop-detection и гигиена

### P3-13 · B3 — починить до включения; AutoBoost удалить
1. `rl/config.py`: добавить `loop_detection_enabled: bool = False` и `loop_consecutive_threshold: int = 10` (+ в to_dict/from_dict).
2. `rl/async_trainer.py`:
   * `self.loop_detector = LoopDetector({"consecutive_threshold": cfg.loop_consecutive_threshold}) if cfg.loop_detection_enabled else None`
   * историю действий заполнять из `step_actions` текущего шага (уже исправлено в P1-4 — данные теперь настоящие);
   * в `LoopDetector.update_batch` не считать серии для легальных «ускорителей времени»:
     ```python
     if action in ("DAY", "WEEK"):
         state.consecutive_count = 0   # серии DAY/WEEK — не петля
     ```
   * **удалить** `_maybe_boost_entropy` и поля `_boost_*` целиком (вместе с вызовом в `train()`); при желании оставить только логирование алертов.

### P3-14 · C1 — честный топ действий
```python
order = sorted(range(len(action_counts)), key=lambda i: action_counts[i], reverse=True)[:5]
top_actions = {action_names[i]: round(action_counts[i] / max(total_actions, 1) * 100, 2)
               for i in order}
```

### P3-15 · C2/C3 — метрики и утечка
```python
self.metrics.loop_action_name = loop_action_name      # было: None
# _ep_lengths: либо удалить совсем (никто не читает), либо trim как у _ep_returns
```

### P3-16 · C6 — документация
В `SUMMARY.md` / `OPTIMIZATION_GUIDE.md` заменить `--compile False` на «флаг `--compile` не передавать» (store_true, значение ломает argparse — `train.py:53`).

---

## Проверка после каждого уровня

> **Код фиксов уже проверен запуском:** `work/verify_fixes.py` применяет P0-1, P0-2, P2-7, P2-8, A3, D1 к копии кода (`work/rl_fixed/`) и тестирует результат — **12/12 PASS** (round-trip наград, k3-KL, KL-early-stop, LR-decay 110%→10%, пустые маски без краша, срезы шагов). Прогон: `cd work && python3 verify_fixes.py`.

**1. Статика:** `python -m pyflakes rl/ train.py` — не должно быть `undefined name`.

**2. Smoke-тест (быстро, без eval):**
```bash
python train.py --steps 100000 --envs 8 --eval-freq 0 --amp off \
    --n-epochs 4 --ent-coef 0.015 --target-kl 0.02 --name smoke
```

**3. Контрольные точки по логам/TensorBoard:**

| После фикса | Что должно измениться |
|---|---|
| P0-1 | round-trip `RewardConfig.to_dict → from_dict` даёт идентичные веса (все 6 полей) |
| P0-2 | `train/approx_kl` реалистичный (~1e-3…1e-2, не ~1e-5); `train/learning_rate` **убывает** к концу; в логе update иногда обрывается по KL |
| P0-3 | в логе eval нет WARNING про normalization; `eval/*` метрики заметно выше прежних (нормализация включилась); в score растут базы |
| P1-4 | FPS растёт; нагрузка на CPU/PCIe падает (профиль: ~1 ГБ/rollout D2H исчез) |
| P1-5 | стена времени между eval'ами ≈ чистое обучение |
| P2-7 | `update()` без env-масок даёт конечный `approx_kl` (тест «мусорные маски» из `work/test_bugs.py` → значения масок все True) |

**4. Регрессия:** прогнать `work/test_bugs.py` против живого `rl/` (скопировать живые файлы в `work/rl/`): после P0-1 тест B1 должен дать `NOT REPRODUCED`; после P2-7 тест мусорных масок показывает единицы.

**5. Полный baseline после всех фиксов:**
```bash
python train.py --steps 2000000 --envs 8 --n-steps 4096 --batch-size 8192 \
    --n-epochs 4 --ent-coef 0.015 --gae-lambda 0.95 --target-kl 0.02 \
    --obs-mode flat --amp off \
    --eval-freq 500000 --eval-episodes 10 --eval-seeds 42 43 44 \
    --name stable_baseline
```

Критерии здоровой динамики: entropy плавно снижается от ~3.7 без обвалов; `approx_kl` в коридоре 0.005–0.03; value_loss монотонно вниз после первого плато; eval-score растёт между eval'ами; FPS стабилен.
