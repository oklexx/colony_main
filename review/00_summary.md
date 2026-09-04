# СВОДНЫЙ ОТЧЁТ ПО РЕВЬЮ — Sakhalin Colony

**Дата:** автоматическое ревью
**Объём:** C++ игровое ядро (`src/`) + Python RL-пайплайн (`rl/`, `python/`, `train.py`)
**Игнорируемые папки:** `extracted*`, `resources_out`, `promer` (мусор распаковки).

---

## Вердикт

**REQUEST CHANGES.** Проект функционально работает как игра, но содержит **несколько
критических багов**, которые:
1. Ломают использование GUI мышью и консольного меню (падения/недоступность функций).
2. **Систематически портят обучение RL** (неверная маска терминальности в GAE + неверный
   `last_done`) → агент учится на смещённом policy-градиенте.
3. Мешают запуску обучения на CPU и теряют чекпоинты.

Ниже — разбивка по файлам `review/`:
- `01_game_logic_cpp.md` — баги игровой логики C++ (env/game/earth/rng/data).
- `02_gui_bindings_main.md` — баги GUI, pybind11-биндингов и консольного меню.
- `03_rl_training.md` — глубокое ревью обучения PPO (GAE, loss, буфер, async, гиперпараметры).
- `04_data_config_inconsistencies.md` — несоответствия данных/конфигов (reward.json, ui/, README).
- `05_false_positives_and_notes.md` — ложноположительные находки и прочие замечания.

---

## Сводная таблица серьёзности

### CRITICAL (блокируют слияние / ломают обучение)

| # | Файл | Проблема | Кратко |
|---|------|----------|--------|
| C-G1 | `rl/rollout_buffer.py:95` | Маска `dones[t+1]` вместо `dones[t]` в GAE |Advantages/value-таргеты смещены на каждой границе эпизода |
| C-G2 | `rl/async_trainer.py:106-112` | `last_done` = OR по роллауту, а не флаг последнего шага |Бустрап незаконно режется для не-терминального `s_T` |
| C-G3 | `rl/ppo.py:48,101` | `device_type="cuda"` жёстко зашит в AMP |Падение на CPU без `--amp off` |
| C-C1 | `src/gui.cpp:380,1114` | Диалоги, открытые мышью, мгновенно закрываются |GUI неюзабелен мышью |
| C-C2 | `src/gui.cpp:663,773,1176` | Висячий `selected_base` → use-after-free |Segfault при realloc вектора баз |
| C-C3 | `src/main.cpp:67,76,78` | Off-by-one нумерации меню |OOB-чтение + недостижимый Quit |
| C-E1 | `src/env.cpp:140-152,348,479` | `reset()` не сбрасывает кэш `net_worth` |Искажение награды на старте каждого эпизода |

### MAJOR

| # | Файл | Проблема |
|---|------|----------|
| M-C1 | `src/bindings.cpp:432` | `obs_buffer()` возвращает numpy, разделяющий сырой буфер C++ (aliasing/ lifetime) |
| M-G1 | `src/gui.cpp:465` | Неверный текст сезона (сдвиг на 1) |
| M-G2 | `src/main.cpp` (см. C-C3) | то же меню |
| M-RL1 | `rl/async_trainer.py:198-201` | Периодические чекпоинты никогда не пишутся (условие не срабатывает) |
| M-RL2 | `rl/async_trainer.py:52,151` | «AsyncTrainer» на самом деле синхронный (очередь не используется) |
| M-RL3 | `python/cpp_vecenv.py:121` | `truncated` слит с `terminated` → бустрап режется на time-limit |
| M-RL4 | `rl/config.py:11-19` | Масштаб наград несогласован (`tax_bonus=250` доминирует) |
| M-RL5 | `src/env.cpp:97-105,342` (проверить) | Порядок перечисления `Season` (summer=0 в constants.h vs spring=0 в obs) |
| M-D1 | `configs/reward.json` | Противоречит дефолтам кода по 5 полям + мёртвый блок `curriculum_rewards` |

### MINOR — см. соответствующие файлы.

---

## Приоритет исправлений

1. `rl/rollout_buffer.py` — маска `dones[t]` (C-G1) + обновить `tests/test_gae.py`.
2. `rl/async_trainer.py` — `last_done` = done последнего шага (C-G2).
3. `rl/ppo.py` — `device_type=self.device.type` (C-G3).
4. `src/env.cpp` — сброс `net_worth_valid_` в `reset()` (C-E1).
5. `src/gui.cpp` — порядок `g_dlg_age` / хранить координаты вместо сырого указателя (C-C1, C-C2).
6. `src/main.cpp` — убрать `+1` в печати меню и `quit_key` (C-C3).
7. `rl/async_trainer.py` — корректное условие сохранения (M-RL1).
8. `configs/reward.json` — синхронизировать с кодом (M-D1).

---

## Позитивные паттерны (что сделано хорошо)

- Точное совпадение `obs_size()` (203) с фактическим наполнением `obs()` — нет рассинхрона
  размерностей между C++ и RL (частая причина поломок RL здесь отсутствует).
- Корректная диспетчеризация менеджер-действий, согласованная с `cpp_env.py`.
- `Game(const Game&)` и `operator=` глубоко копируют `rng`/`rng_np` и карты — корректный
  `reset()` и snapshot для RL.
- `MtRandom`/`PCG64` — точные порты CPython/numpy RNG (важно для воспроизводимости).
- Clipped surrogate, entropy/value веса, orthogonal-init, `grad_clip` — стандартно и верно.
- GIL освобождается в многопоточном батч-окружении.
- `delete_base` (game.cpp:295-307) — **корректный** swap-and-pop с верным обновлением
  `base_index_map_` (проверено вручную; см. `05_false_positives_and_notes.md`).
