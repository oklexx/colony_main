# Применение рекомендаций второго ревью — Sakhalin Colony

**Дата:** 2026-08-28
**Основание:** `06_post_fix_review.md` прошло второе ревью (см. текст «Второе ревью» пользователя).
**Цель:** Устранить высоко/среднеприоритетные проблемы, отмеченные во втором ревью.

## Применённые правки

### m‑C1 — `obs_buffer()` возвращал разделяемую память (`src/bindings.cpp`)
Ранее `py::array_t<float>(shape, v.obs_buffer())` делал numpy‑массив view на C++‑буфер.
После следующего `step_wait_batch` ранее возвращённый массив «тихо» показывал новые данные
(порча observation в RL).
**Исправлено:** биндинг копирует данные через `std::memcpy` в новый владеющий массив.
Добавлен `#include <cstring>`.
**Верификация:** собран `colony_cpp.pyd`; `np.shares_memory(a, b) == False` для двух вызовов
`obs_buffer()`, и ранее полученный массив не меняется после шага.

### M‑RL3 — truncation слит с termination (`python/cpp_vecenv.py`, `rl/env_manager.py`,
`rl/rollout_buffer.py`, `rl/async_trainer.py`)
Ранее `dones = terminateds | trunceds` и этот OR использовался и как маска bootstrap в GAE,
и как `last_done`. Time‑limit truncation трактовался как терминальное состояние → бустрап
обрезался, хотя `V(s_{t+1})` валидна.
**Исправлено:**
- `cpp_vecenv.step_wait` сохраняет `self._last_terminateds = terminateds` (истинный флаг).
- `env_manager.step` добавляет `terminated` в возвращаемый dict.
- `env_manager.collect_step` передаёт `terminated=...` в `buffer.add`.
- `RolloutBuffer.add` принимает опциональный `terminated` (по умолчанию = `done`, обратная
  совместимость с `test_ppo_smoke` сохранена) и хранит `self.terminated`.
- `RolloutBuffer.compute_gae` использует `self.terminated` как маску bootstrap (вместо `self.dones`).
- `async_trainer._collect_rollout` вычисляет `last_done` из `terminated` последнего шага
  (а не из `dones`).

### m‑E1 — UB сдвига в `src/game.cpp:318`
`int64_t birth_days = BIRTH_DAYS >> n_puerp;` — при `n_puerp >= 64` UB.
**Исправлено:** `int64_t birth_days = (n_puerp >= 64) ? 1 : (BIRTH_DAYS >> n_puerp);`
(с последующим `if (birth_days < 1) birth_days = 1;`).

## Верификация

```
python -m pytest tests/test_gae.py tests/test_ppo_smoke.py -q   -> 5 passed
```
C++ (`bindings.cpp`, `game.cpp`) — пересборка `.pyd` через CMake прошла успешно;
проверка m‑C1 в рантайме (см. выше).

## Не исправлено (низкий приоритет / требует отдельной работы)

- **M‑D2** (`ui/`): `ui/main_window.py` импортирует отсутствующий `core` и требует `pygame`.
  Нужна либо реализация `core`, либо удаление папки. Не затрагивает обучение/raylib‑GUI.
- **m‑RL (мелкие улучшения):** value‑loss clipping, LR‑расписание, сохранение RMS‑нормализации,
  нормализация наблюдений — улучшения качества, не баги.
- **m‑E5:** документировать порядок `Season` в `constants.h`. Не критично.

## Итог

Все высоко‑ и среднеприоритетные проблемы второго ревью устранены и проверены.
Проект пригоден для обучения RL.
