# План реализации исправлений — Sakhalin Colony RL

**Основание:** `PROJECT_AUDIT_FULL.md` (v3) + `FIXES_P0_P3.md`. Все статусы багов проверены по живому коду.
**Легенда:** ✅ — код фикса уже прогнан в sandbox (verify_fixes.py 12/12 или snippet-тест); 🆕 — новый код, требует проверки при применении (процедура указана); 🔨 — требует пересборки C++ (`colony_cpp.pyd`).

**Правила применения:**
1. Каждую фазу — отдельным git-коммитом: `fix/phase1-ppo-core`, `fix/phase2-perf`, `fix/phase3-env`, `fix/phase4-hygiene`.
2. После каждой фазы — блок «Проверка фазы» целиком. Не переходить к следующей фазе при красной проверке.
3. C++-правки (N1, N3, N4, N5) можно копить и пересобрать один раз в фазе 3.
4. Откат: `git revert` коммита фазы. Фиксы независимы, кроме: D1 использует deque из B3 (до B3 работает с list), target_kl требует поля в Config.

---

## Фаза 0. Подготовка (30 мин)

```bash
git checkout -b fix/rl-stability
# Базовый прогон для сравнения (зафиксировать FPS, KL, entropy, eval-score в логе):
python train.py --steps 200000 --envs 8 --eval-freq 100000 --eval-episodes 10 --name baseline_before
python -m pyflakes rl/ train.py > pyflakes_before.txt
```
Записать в блокнот: FPS baseline, время rollout, время update, время eval.

---

## Фаза 1 (P0). Ядро обучения: награды, KL, LR, eval — Python + 1 правка C++

### 1.1 · B1 — единые дефолты наград ✅
**Файл:** `rl/config.py` (~строки 84–109). Заменить тело `RewardConfig.from_dict`:
```python
@classmethod
def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
    """Дефолты — только из датакласса (= C++ env.h = reward.json).
    JSON переопределяет лишь явно заданные ключи."""
    base = cls()
    for k, v in d.items():
        if not hasattr(base, k):
            continue
        if k == "idle_build_threshold_days":
            v = int(v)
        setattr(base, k, v)
    return base
```
**Проверка:**
```python
from rl.config import RewardConfig
d = RewardConfig()
assert all(getattr(d,k)==getattr(RewardConfig.from_dict(d.to_dict()),k) for k in d.to_dict())
assert RewardConfig.from_dict({}).daily_income == 1.0        # было 0.3
assert RewardConfig.from_dict({}).clip_reward_max == 50.0    # было 10
```

### 1.2 · A3-хвост — фильтр полей в `load_from_file` ✅
**Файл:** `rl/config.py:278`. Заменить строку `self.__dict__.update(...)`:
```python
import dataclasses
allowed = {f.name for f in dataclasses.fields(Config)} - {"reward"}
self.__dict__.update({k: v for k, v in data.items() if k in allowed})
```

### 1.3 · B6+B7 — честный KL, early-stop, живой LR-decay ✅
**(a) `rl/config.py`** — поле в `Config` (после `max_grad_norm`):
```python
target_kl: float = 0.02          # 0 = выключено
```
+ `"target_kl"` в кортеж `to_dict()` + `target_kl=d.get("target_kl", 0.02)` в `Config.from_dict`.

**(b) `rl/ppo.py` `__init__`** — параметры и поля:
```python
def __init__(self, ..., total_training_steps: int = 0, target_kl: float = 0.0):
    ...
    self.target_kl = target_kl
```

**(c) `rl/ppo.py:255`** — k3-оценка KL:
```python
with torch.no_grad():
    log_ratio = (new_log_probs - old_log_probs).clamp(-10.0, 10.0)
    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()   # >= 0
```

**(d) `rl/ppo.py` `update()`** — остановка по KL:
```python
kl_stop = False
for _ in range(self.n_epochs):
    if kl_stop:
        break
    for batch in self.buffer.get_batches(self.batch_size):
        ...                                    # тело без изменений
        n_batches += 1
        if self.target_kl > 0 and approx_kl.item() > 1.5 * self.target_kl:
            kl_stop = True
            break
```
Норма: первый мини-батч update всегда KL=0 (new==old) — стоп не раньше 2-го батча.

**(e) `rl/ppo.py` lr_lambda** — старт ровно 100% (было 110%):
```python
return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))   # [1.0 ... 0.1]
```

**(f) `rl/env_manager.py:122-138`** — передать горизонт и target_kl:
```python
samples_per_rollout = cfg.n_steps * cfg.n_envs
n_updates = max(1, cfg.total_timesteps // samples_per_rollout)
batches_per_update = max(1, samples_per_rollout // cfg.batch_size)
opt_steps = n_updates * batches_per_update * cfg.n_epochs

self.ppo = PPO(
    ...,                       # текущие аргументы
    device=device,
    lr_decay=True,
    total_training_steps=opt_steps,
    target_kl=cfg.target_kl,
)
```

**(g) `train.py`** — CLI: `p.add_argument("--target-kl", type=float, default=_d.target_kl)`; `target_kl=args.target_kl` в `Config(...)`.

**Проверка:** smoke 100k (блок «Проверка фазы»): `train/approx_kl` ~1e-3…1e-2 (не ~1e-5), `train/learning_rate` убывает, в логе случаются обрывы по KL.

### 1.4 · B2 — eval с нормализацией 🆕
**Файл:** `rl/async_trainer.py`.
**(a)** В блоке сохранения чекпойнта (после строки 619) добавить:
```python
self.em.env.venv.save_normalization(str(save_dir / "normalization.json"))
```
**(b)** В `train()` сразу после `obs = self.em.reset()` — чтобы первый eval не был слепым:
```python
Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)
self.em.env.venv.save_normalization(str(Path(self.cfg.model_dir) / "normalization.json"))
```
**(c)** В `_eval` после строки 335 — страховка:
```python
if norm_str is None:
    self._log("[Eval] WARNING: normalization.json не найден — eval БЕЗ нормализации obs!")
```
**Проверка:** после smoke-прогона с eval: в логе НЕТ warning; `eval/days` заметно выше прежних прогонов (нормализация включилась).

### 1.5 · F3 — адекватный eval score ✅(логика)
**Файл:** `rl/async_trainer.py:376`:
```python
score = ((days_agg / 1000.0) * w1 + bases_agg * w2
         + (people_agg / 100.0) * w3 + max(0.0, return_agg) * w4)
```
⚠️ `best_score` до/после несопоставимы — старые `best_model.meta.json` не сравнивать с новыми.

### 1.6 · N8 — reward-конфиг в meta ✅(snippet)
**Файл:** `rl/async_trainer.py`, в `_eval` при записи meta (~строка 415) добавить ключ:
```python
"config": {"reward": self.cfg.reward.to_dict()},
```
Теперь `evaluator.py:158-170` найдёт `meta["config"]["reward"]` и создаст eval-среду с тренировочными весами.

### 1.7 · N1 — знак manual_tax 🔨
**Файл:** `src/env.cpp:909`:
```cpp
// было: rew -= cfg_.manual_tax_penalty;  c_manual_tax -= cfg_.manual_tax_penalty;
rew += cfg_.manual_tax_penalty;  c_manual_tax += cfg_.manual_tax_penalty;
```
(`manual_tax_penalty = -0.5` → теперь реальные −0.5, как в комментарии «cost of manual tax payment».)

### Проверка фазы 1
```bash
python -m pyflakes rl/ train.py                       # не хуже pyflakes_before.txt
cd work && python3 verify_fixes.py                    # 12/12 PASS (паттерны фиксов)
python train.py --steps 100000 --envs 8 --eval-freq 50000 --eval-episodes 5 --name smoke_p1
```
Ожидание: KL реалистичный и иногда срабатывает стоп; LR убывает; eval без WARNING; крашей нет. Если правили C++ — пересборка (Приложение A) + `python tests/test_reward_clip.py` и `test_tax_daily_bonus.py`.

---

## Фаза 2 (P1). Нагрузка на систему

### 2.1 · D1 — убрать ~1 ГБ D2H на rollout ✅(срезы)
**Файл:** `rl/async_trainer.py`, `_collect_rollout`.
**(a)** Удалить из цикла пошаговое чтение `terminated` (строки 144–146).
**(b)** Заменить блок 164–180 (копирование всего буфера actions):
```python
pos = self.em.buffer.pos - 1
step_actions = (self.em.buffer.actions[pos * n_envs:(pos + 1) * n_envs]
                .cpu().numpy())                        # 64 байта вместо 256 КБ
for env_idx in range(n_envs):
    action_idx = int(step_actions[env_idx])
    action_name = (action_names[action_idx]
                   if action_idx < len(action_names) else f"ACTION_{action_idx}")
    self._action_history.append((env_idx, action_name))
```
**(c)** После цикла — одно чтение terminated:
```python
last_pos = self.em.buffer.pos - 1
terminated = (self.em.buffer.terminated[last_pos * n_envs:(last_pos + 1) * n_envs]
              .cpu().numpy())
last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)
```
**(d)** `__init__`: `self._action_history = deque(maxlen=1000)` (+ `from collections import deque`); удалить пошаговый пересрез `[-50000:]`.

### 2.2 · D2 — eval реже и легче (без кода)
```bash
--eval-freq 500000 --eval-episodes 10 --eval-seeds 42 43 44
```
Опционально позже: кэшировать среду оценщика между eval (создавать один раз в `_eval` первого вызова).

### 2.3 · N12 — экспериментальные конфиги ✅(расчёт)
**Файлы:** `configs/exp_25_minimap.json`, `exp_26_hybrid.json`, `exp_27_minimap_lr_low.json`:
```json
"n_envs": 32,          // было 1024 → буфер minimap 56.4 ГБ VRAM (невозможно)
"n_steps": 512,        // было 2048 → теперь 0.44 ГБ
"batch_size": 4096,    // было 32768 (= весь буфер, 1 батч); 4 батча на эпоху
```
Формула для будущих экспериментов: `VRAM(obs) ≈ n_envs × n_steps × 8×29×29 × 4B` (minimap). Цель: < 4 ГБ на буфер.

### Проверка фазы 2
```bash
python train.py --steps 100000 --envs 8 --eval-freq 0 --name smoke_p2
```
Ожидание: FPS выше baseline; в мониторинге (nvidia-smi / диспетчер задач) исчезла пилообразная нагрузка CPU→GPU каждые ~50 мс.

---

## Фаза 3 (P2). Среда (C++) и страховки

### 3.1 · A4 ✅ — `rl/env_manager.py:350`:
```python
schedule = list(default_schedule)      # было: [(t, s+1) for t, s in enumerate(...)] → TypeError
```

### 3.2 · A6 ✅ — `rl/rollout_buffer.py:45 и 197`:
```python
self.action_masks = torch.ones(total, n_actions, dtype=torch.bool, device=device)
```
Сравнение `action_masks == 0` в ppo.py работает с bool; память 5.9→1.5 МБ.

### 3.3 · A5 ✅ — хелпер масок в `rl/ppo.py`, применить в `collect_step` и `_compute_loss_components`:
```python
def _apply_action_masks(logits, masks):
    if masks is None:
        return logits
    m = masks.bool()
    all_blocked = ~m.any(dim=-1, keepdim=True)
    m = torch.where(all_blocked, torch.ones_like(m), m)
    return logits.masked_fill(~m, float("-inf"))
```
(Через C++ недостижимо — DAY/WEEK всегда доступны, — но защищает от NaN при сбоях.)

### 3.4 · A7 🆕 — симметрия save/load, `rl/ppo.py`:
В `save()` добавить в словарь:
```python
"scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
"ent_coef": self.ent_coef, "vf_coef": self.vf_coef, "opt_step": self._current_step,
```
В `load()`:
```python
model = getattr(self.model, "_orig_mod", self.model)     # torch.compile-обёртка
model.load_state_dict(ckpt["model_state"])
self.optimizer.load_state_dict(ckpt["optimizer_state"])
if self.scheduler is not None and ckpt.get("scheduler_state"):
    self.scheduler.load_state_dict(ckpt["scheduler_state"])
self.ent_coef = ckpt.get("ent_coef", self.ent_coef)
self.vf_coef = ckpt.get("vf_coef", self.vf_coef)
self._current_step = ckpt.get("opt_step", 0)
```
**Проверка:** save → load → `loss` на одном батче совпадает до/после (написать 5-строчный тест).

### 3.5 · B8 ✅(snippet) — `rl/async_trainer.py:637`:
```python
if self.best_score is not None and (self._es_best_score is None
                                    or self.best_score > self._es_best_score):
```

### 3.6 · C5 — `train.py:89` и `train_ui/evaluator.py:151`:
```python
args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
```

### 3.7 · N5 🔨 — idle-штраф только когда есть что строить, `src/env.cpp:1107-1111`:
```cpp
if (days_since_last_build_ >= cfg_.idle_build_threshold_days) {
    bool any_build_available = false;
    for (int i = 0; i < n_build_; ++i) {
        const BaseData* d = build_data_[i];
        if (has_unlocked_ && !unlocked_.count(d->id)) continue;
        if (d->id == ROAD_ID && road_count() >= MAX_ROADS) continue;
        if (g.money < d->price) continue;
        any_build_available = true; break;
    }
    if (any_build_available) {
        rew += cfg_.idle_build_penalty; c_idle += cfg_.idle_build_penalty;
    }
    days_since_last_build_ = 0;
}
```
(Убирает невыбираемую яму «банкрот → −10 каждые 3 шага без возможности строить». Стоимость: BFS find_lot не вызывается — только проверка денег/анлоков, дёшево.)

### 3.8 · N4 🔨 — счётчики дней, `src/env.cpp`:
`tax_due_days_` (строка 1090) и `days_since_last_build_` (строка 1081) инкрементировать на реально прошедшие дни:
```cpp
// после блока advance_day/advance_week (results уже заполнен):
int days_passed = 0;
for (const auto& r : results) { (void)r; days_passed += 1; }   // WEEK даёт 7 DayResult
if (g.annual_tax_due() || g.main_tax_due()) tax_due_days_ += std::max(1, days_passed);
else                                        tax_due_days_ = 0;
days_since_last_build_ += std::max(1, days_passed);
```
⚠️ Убедиться, что `advance_week()` возвращает 7 `DayResult` (проверить в `game.cpp`); если один агрегированный — умножать на 7 по флагу действия. Пороги после фикса: tax grace = честные 60 дней, idle = честные 3 дня.

### 3.9 · N3 🔨 — хардкоженные веса в RewardConfig (механически)
**(a) `src/env.h` struct RewardConfig** — добавить поля (дефолты = текущие хардкоды, чтобы поведение не изменилось):
```cpp
double tax_fail_penalty   = 5.0;    // нет денег на налог
double death_penalty      = 20.0;   // за смерть жителя
double base_lost_penalty  = 30.0;   // за потерю базы
double born_bonus         = 1.0;    // рождение/прибытие
double debt_coeff         = 0.02;   // × credit/1000
double home_overflow_penalty = 2.0;
double housing_need_bonus = 3.0;    // × log1p((people-housing)/10)
double food_need_bonus    = 2.0;    // × log1p(idle/5)
double water_need_bonus   = 2.0;
```
**(b) `src/env.cpp`** — замены (строки): 716/722 `rew -= 5.0` → `rew -= cfg_.tax_fail_penalty`; 940 `0.02` → `cfg_.debt_coeff`; 941 `1.0` → `cfg_.born_bonus`; 942 `20.0` → `cfg_.death_penalty`; 943 `30.0` → `cfg_.base_lost_penalty`; 944 `2.0` → `cfg_.home_overflow_penalty`; 973/985/998 `3.0/2.0/2.0` → соответствующие `cfg_.*`.
**(c)** Зеркально: поля в Python-датакласс `RewardConfig` (+ они автоматически попадут в `to_dict`, а `from_dict` из 1.1 подхватит их сам), `_REWARD_KEYS` в `python/cpp_vecenv.py:54-66`, ключи в `configs/reward.json`.
**Проверка:** `COLONY_DEBUG=1 python -c "from cpp_vecenv import make_cpp_vec_env; ..."` — сверить печать rc.*; прогнать `tests/test_reward_clip.py`, `test_milestones.py`, `test_error_penalty.py`.

### Проверка фазы 3
Пересборка C++ (Приложение A) → весь набор тестов:
```bash
python -m pytest tests/ -x -q        # или последовательно python tests/test_*.py
python train.py --steps 100000 --envs 8 --eval-freq 0 --name smoke_p3
```

---

## Фаза 4 (P3). Loop-detection, метрики, гигиена

### 4.1 · B3 — loop-detection: починить, потом включать
**(a) `rl/config.py`:** `loop_detection_enabled: bool = False`, `loop_consecutive_threshold: int = 10` (+ to_dict/from_dict + CLI-флаги).
**(b) `rl/async_trainer.py:88`:**
```python
self.loop_detector = (LoopDetector({"consecutive_threshold": cfg.loop_consecutive_threshold})
                      if getattr(cfg, "loop_detection_enabled", False) else None)
```
**(c) `rl/loop_detector.py` `update_batch`** — не считать серии легальных «ускорителей»:
```python
if action in ("DAY", "WEEK"):
    state.consecutive_count = 0
    state.last_action = action
    state.action_sequence.append(action)
    continue
```
**(d) Удалить** `_maybe_boost_entropy`, `_boost_history`, `_last_boost_rollout`, `_boost_cooldown`, `_initial_ent_coef` и вызов `self._maybe_boost_entropy(rollout_idx)` в `train()`. (Данные для детектора уже настоящие после 2.1b.)

### 4.2 · C1 ✅(snippet) — честный топ действий, `rl/async_trainer.py:533-540`:
```python
action_counts = self._calculate_action_distribution()
total_actions = sum(action_counts)
order = sorted(range(len(action_counts)), key=lambda i: action_counts[i], reverse=True)[:5]
top_actions = {self._action_names[i]: round(action_counts[i] / max(total_actions, 1) * 100, 2)
               for i in order}
```

### 4.3 · C2/C3 — метрики и утечка:
```python
self.metrics.loop_action_name = loop_action_name    # было: None
# _ep_lengths: удалить поле и append (никто не читает) ЛИБО trim как у _ep_returns
```

### 4.4 · N9 — hybrid-guard в `train_ui/evaluator.py:183-188`:
```python
if normalization_path is not None and is_hybrid:
    norm_path = Path(normalization_path)
    if not norm_path.exists():
        raise FileNotFoundError(
            f"normalization file not found: {norm_path} (hybrid mode requires it)")
    env.normalizer.load(str(norm_path))
    env.normalizer.set_update(False)
```

### 4.5 · N10 ✅(snippet) — eval-сиды без пересечения с тренировкой, `train_ui/evaluator.py:210`:
```python
obs, _info = env.reset(seed=seed + 500_000 + ep)   # тренировочные сиды: base + i*10000
```

### 4.6 · N11 — приоритет reward-JSON над CLI-дефолтами, `python/cpp_vecenv.py:75-77`:
```python
# применять только явно переданные флаги (argparse default=None):
if disable_net_worth is not None:
    rc.disable_net_worth = disable_net_worth
if disable_daily_income is not None:
    rc.disable_daily_income = disable_daily_income
```
(В `train.py` у соответствующих аргументов `default=None` вместо `store_true`-семантики; `store_true` заменить на `action="store_true", default=None`.)

### 4.7 · C6 — документация: в `SUMMARY.md`/`OPTIMIZATION_GUIDE.md` заменить `--compile False` → «флаг `--compile` не передавать» (store_true).

### Проверка фазы 4
```bash
python -m pyflakes rl/ train.py train_ui/evaluator.py python/
python work/test_bugs.py        # против обновлённого rl/: top_actions-тест → NOT REPRODUCED
python train.py --steps 200000 --envs 8 --eval-freq 100000 --eval-episodes 5 \
    --loop-detection-enabled --name smoke_p4      # детектор включён, AutoBoost нет
```

---

## Финал: baseline-прогон и целевые метрики

```bash
python train.py --steps 2000000 --envs 8 --n-steps 4096 --batch-size 8192 \
    --n-epochs 4 --ent-coef 0.015 --gae-lambda 0.95 --target-kl 0.02 \
    --obs-mode flat --amp off \
    --eval-freq 500000 --eval-episodes 10 --eval-seeds 42 43 44 \
    --name stable_after
```

| Метрика | До (оценка) | Целевое после |
|---|---|---|
| `train/approx_kl` | ~1e-5 (занижен ×12) | 0.005–0.03, пики обрезаются стопом |
| `train/learning_rate` | константа 3e-4 | 3e-4 → 3e-5 (косинус) |
| `train/entropy` | обвалы при AutoBoost | плавный спуск от ~3.7 |
| FPS (flat, 8 envs) | baseline | +20–50% (D1+D2) |
| `eval/days` | занижен (без нормализации) | скачок вверх на первом eval после фикса B2 |
| best_score | days-доминирующий | базы реально влияют (F3) |

**Только после зелёного baseline** — запускать `auto_trainer.py` (Optuna): теперь его objective (best_score из meta) осмыслен (N7 закрыт связкой B2+F3+N8).

---

## Приложение A. Пересборка C++ (Windows)
```bash
pip install pybind11
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# результат: python/colony_cpp.pyd (CMakeLists кладёт рядом с обёрткой)
python -c "import sys; sys.path.insert(0,'python'); import colony_cpp; print('OK')"
```

## Приложение B. Сводная таблица «фикс → файл → проверка»

| # | Фикс | Файл(ы) | Тип | Проверка |
|---|------|---------|-----|----------|
| 1.1 | B1 дефолты наград | rl/config.py | ✅ | round-trip assert |
| 1.2 | A3 фильтр полей | rl/config.py | ✅ | load_from_file тест |
| 1.3 | B6+B7 KL/LR | rl/ppo.py, rl/env_manager.py, rl/config.py, train.py | ✅ | smoke: KL, LR-график |
| 1.4 | B2 normalization.json | rl/async_trainer.py | 🆕 | лог eval без WARNING |
| 1.5 | F3 score | rl/async_trainer.py | ✅ | доля bases в score |
| 1.6 | N8 meta.config.reward | rl/async_trainer.py | ✅ | json-ключ в meta |
| 1.7 | N1 знак manual_tax | src/env.cpp:909 | 🔨 | COLONY_DEBUG / тест налога |
| 2.1 | D1 D2H-копии | rl/async_trainer.py | ✅ | FPS, профиль |
| 2.2 | D2 eval-частота | CLI | — | wall-time |
| 2.3 | N12 exp-конфиги | configs/exp_2*.json | ✅ | VRAM < 4 ГБ |
| 3.1 | A4 curriculum | rl/env_manager.py:350 | ✅ | вызов get_curriculum_progress |
| 3.2 | A6 маски ones/bool | rl/rollout_buffer.py:45,197 | ✅ | update без env-масок: KL конечный |
| 3.3 | A5 fallback масок | rl/ppo.py | ✅ | пустая маска → нет NaN |
| 3.4 | A7 save/load | rl/ppo.py | 🆕 | save→load roundtrip |
| 3.5 | B8 early-stop | rl/async_trainer.py:637 | ✅ | отрицательный score-тест |
| 3.6 | C5 device | train.py:89, evaluator.py:151 | 🆕 | `--device cuda:0` |
| 3.7 | N5 idle-ловушка | src/env.cpp:1107 | 🔨 | юнит-тест «банкрот без штрафа» |
| 3.8 | N4 дни vs шаги | src/env.cpp:1081,1090 | 🔨 | WEEK-тест: grace=60 дней |
| 3.9 | N3 веса в конфиг | env.h, env.cpp, cpp_vecenv.py, config.py, reward.json | 🔨 | тесты наград |
| 4.1 | B3 loop-detection | config, async_trainer, loop_detector | 🆕 | smoke с флагом |
| 4.2 | C1 top_actions | rl/async_trainer.py:533 | ✅ | snippet-тест |
| 4.3 | C2/C3 метрики | rl/async_trainer.py | 🆕 | pyflakes |
| 4.4 | N9 hybrid guard | evaluator.py:183 | 🆕 | raise наmissing-файле |
| 4.5 | N10 eval-сиды | evaluator.py:210 | ✅ | нет пересечений |
| 4.6 | N11 CLI-приоритет | cpp_vecenv.py:75 | 🆕 | JSON-флаг survives |
| 4.7 | C6 доки | SUMMARY.md, OPTIMIZATION_GUIDE.md | — | — |
