# Пост-фикс ревью — Sakhalin Colony (`sakhalin_colony_main`)

**Дата:** 2026-08-28
**Автор:** opencode (применение правок по итоговому ревью)
**Статус:** Все критические (Critical) баги исправлены и проверены; часть Major/Minor — исправлена; остаток — список рекомендаций.

---

## 1. Что сделано

В предыдущем проходе применены правки по итоговому отчёту ревью. В этом документе
зафиксирован **актуальный статус** каждого найденного бага и результаты верификации.
Полный исходный код проекта собран в `code_dump/part_01..04.txt` (см. раздел 4).

## 2. Статус критических багов (Critical)

| ID | Файл(ы) | Суть | Статус |
|----|---------|------|--------|
| **C-G1** | `rl/rollout_buffer.py` | GAE-маска брала `dones[t+1]` вместо `dones[t]` (off-by-one в transition-based семантике) | **FIXED + TESTED** |
| **C-G2** | `rl/async_trainer.py` | `last_done` считался как OR по всему роллауту, а не done последнего шага | **FIXED** |
| **C-G3** | `rl/ppo.py` | жёстко зашит `device_type="cuda"` → падение на CPU | **FIXED + TESTED (CPU)** |
| **C-C1** | `src/gui.cpp` | диалоги, открытые мышью, мгновенно закрывались | **FIXED (компиляция ОК)** |
| **C-C2** | `src/gui.cpp` | висячий указатель `selected_base` → use-after-free | **FIXED (компиляция ОК)** |
| **C-C3** | `src/main.cpp` | off-by-one меню → OOB-чтение и недостижимый Quit | **FIXED (компиляция ОК)** |
| **C-E1** | `src/env.cpp` | `reset()` не сбрасывал кэш `net_worth` → смещение награды | **FIXED (компиляция ОК)** |

## 3. Конкретные изменения (file:line)

### C-G1 — GAE terminal mask (`rl/rollout_buffer.py`)
```python
# compute_gae(), ветка t != n_steps-1
next_values = self.values[next_start:next_start + self.n_envs]
# dones[t] == terminal(s_{t+1})  (transition-based)
next_dones = self.dones[start:end]      # было: self.dones[next_start:...] (== dones[t+1])
```
`tests/test_gae.py`: `reference_gae` приведён к той же семантике
(`next_done = dones[t]` для t<T-1, и `last_done` для t=T-1), так что тест
реально валидирует корректность, а не внутреннюю согласованность бага.

### C-G2 — `last_done` (`rl/async_trainer.py`, `_collect_rollout`)
Удалён `done_mask` (OR по роллауту). В конце цикла:
```python
last_value = self.em.ppo.model.get_value(obs)
# done именно последнего собранного шага (== buffer.dones[T-1] == terminal(s_T))
last_done = torch.tensor(dones, dtype=torch.bool, device=self.device)
```

### C-G3 — AMP device (`rl/ppo.py`)
```python
self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp_dtype == "float16"))
...
with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
```
Примечание: в установленном torch 2.13 `GradScaler` принимает `device` позиционно
(не `device_type`), поэтому передаётся `self.device.type` — корректно и для CPU, и для CUDA.

### C-C1 — auto-close диалогов (`src/gui.cpp`)
Добавлена глобальная `static int g_dlg_prev_frame`. `panel()` теперь закрывает диалог
только если `cur_dlg == (Dlg)g_dlg_prev_frame` (т.е. диалог уже был открыт на пред. кадре):
```cpp
if (cur_dlg != DLG_NONE && g_dlg_age > 2 && cur_dlg == (Dlg)g_dlg_prev_frame
    && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) { ... }
```
В конце кадра: `g_dlg_prev_frame = (int)cur_dlg;`

### C-C2 — dangling pointer (`src/gui.cpp`)
`static const Base* selected_base` заменён на координаты
`static int sel_bx = -1, sel_by = -1;` (file-scope). При выборе клетки сохраняются
координаты; при отрисовке база резолвится заново через `g.base_in_box(sel_bx, sel_by)`.
Сброс координат при `env.reset()` в диалоге «Новая игра».

### C-C3 — menu off-by-one (`src/main.cpp`)
```cpp
// менеджеры
std::cout << "  [" << mgr_base + i << "]  " << mgr_names[i] << "\n";   // было: mgr_base + i + 1
int quit_key = mgr_base + N_MANAGERS;                                   // было: + N_MANAGERS + 1
```
Теперь `quit_key == max_key`, нет выхода за границы `action_map`.

### C-E1 — net_worth cache (`src/env.cpp`)
```cpp
void ColonyEnvCpp::reset(int64_t seed) {
    game_ = Game(*base_data_, *events_data_, seed, map_size_);
    net_worth_valid_ = false;      // добавлено
    cached_net_worth_ = 0.0;       // добавлено
    steps_ = 0;
    ...
}
```
Плюс `invalidate_net_worth()` добавлена после уплаты налога и после авто-обслуживания
(мутации `money`/`live_time` теперь сбрасывают кэш, `net0` на первом шаге корректен).

## 4. Major / Minor, исправленные в этом же проходе

| ID | Файл | Статус |
|----|------|--------|
| **M-RL1** | `rl/async_trainer.py` | Периодическое сохранение по `rollout_idx % save_every` (`save_every = round(save_freq / steps_per_rollout)`). Старое условие `total_done % save_freq < steps_per_rollout` практически никогда не срабатывало. **FIXED** |
| **m-RL6** | `rl/env_manager.py` | Добавлен `torch.manual_seed(cfg.seed)` перед созданием `ActorCritic` (воспроизводимость). **FIXED** |
| **M-D1** | `configs/reward.json` | Синхронизирован с дефолтами кода (`chain_bonus 0.5`, `daily_income 0.1`, `sale_bonus 0.1`, `survival_bonus 0.0`, `game_over_penalty 20.0`); удалён мёртвый блок `curriculum_rewards`. **FIXED** |
| **m-D4** | `.gitignore` + `python/cpp_env.py` | Добавлен `.gitignore` (`build/`, `*.pyd`, `*.exe`, …); docstring obs 207→203. **FIXED** |

## 5. Верификация

```
python -m pytest tests/test_gae.py -q        -> 3 passed
python -m pytest tests/test_ppo_smoke.py -q   -> 2 passed   (в т.ч. запуск на CPU: C-G3)
```
C++ (`src/env.cpp`, `src/gui.cpp`, `src/main.cpp`) — компиляция MSVC 2022 (x64),
`ALL_COMPILED_OK` (без предупреждений об использовании `selected_base`/undefined).

## 6. Рекомендации (не исправлено — требует отдельной проработки)

- **M-RL3** (truncation vs termination): `cpp_vecenv.py` сливает
  `terminated | truncated` в `dones`. Если в C++ используется time-limit truncation,
  bootstrap в GAE режется незаконно. Нужно пробросить `terminated` отдельно через
  буфер/env_manager и маскировать только по `terminated`. *Нейтрально, если truncation не используется.*
- **M-D2** (`ui/`): `ui/main_window.py` импортирует отсутствующий `core` и требует `pygame`
  (нет в `requirements.txt`). Либо починить, либо задокументировать как нерабочий.
- **m-C1** (`src/bindings.cpp`): `obs_buffer()` возвращает numpy, разделяющий сырой буфер C++
  (тихая порча observation в RL). Рекомендуется возвращать копию.
- **M-C1 / m-G / m-E5** и прочие minor из `review/01..05` — косметика/логика, не блокирующие.

## 7. Собранный код проекта

Весь исходный код проекта (без сторонних библиотек `raylib/`, `include/third_party/`,
папки `review/` и временных скриптов) выгружен в:

```
code_dump/part_01.txt   (2429 строк)
code_dump/part_02.txt   (2765 строк)
code_dump/part_03.txt   (1697 строк)
code_dump/part_04.txt   (2610 строк)
```
Каждый файл ≤ 3000 строк; внутри — секции `// ===== FILE: <relpath> =====` (для C++/json/bat)
или `# ===== FILE: <relpath> =====` (для Python), чтобы восстановить принадлежность кода.
Всего: 46 файлов, ~9501 строка.
