# Sakhalin Colony (Сахалинская колония) — описание проекта

> Документ сгенерирован автоматическим ревью-агентом для облегчения дальнейшей работы.
> См. папку `review/` для детального отчёта об ошибках и ревью обучения моделей.

## 1. Что это за проект

**Жанр / назначение.** Пошаговая экономическая градостроительная колониальная
стратегия «Сахалинская колония» (оригинал — Жмулевский Григорий, версия 3.47,
см. `ui/main_window.py`). Игрок управляет колонией на острове: строит здания,
добывает и производит ресурсы, платит налоги, переживает случайные события,
удерживает население (еда/вода/жильё/работа). Поверх игры реализован **GPU-тренажёр
PPO** (reinforcement learning), который обучает агента играть в эту игру.

**Стек технологий**
- Игровое ядро: **C++17** (`src/*.cpp`, `include/colony/*.h`), без внешних игровых движков.
- RL-обучение: **Python 3.14** + **PyTorch ≥ 2.13** (AMP bf16/fp16, опционально `torch.compile`),
  собственная реализация PPO (не SB3).
- Мост C++ ↔ Python: **pybind11** (модуль `colony_cpp.pyd`, собирается через CMake).
- GUI (рабочий): **raylib 6.0** (C++, `src/gui.cpp`).
- GUI (Python, **НЕРАБОЧИЙ**): pygame (`ui/`).
- Данные: **JSON** (`configs/bases.json`, `configs/events.json`, `configs/reward.json`).
- Логирование: **TensorBoard** (опционально).

**Ключевые константы** (`include/colony/constants.h`)
- Ресурсы `SUNDUK_SIZE = 9`, порядок: `gold, food, coal, iron, oil, stone, water, wood, energy`.
- Observation size `obs_size = 203`, действий `n_actions = 45`
  (`A_BUILD0(2) + 32 + N_MANAGERS(11)`, `n_build = 32` из `BUILD_SUBSET`).
- Действия: `A_DAY=0`, `A_WEEK=1`, `A_BUILD0=2`, `N_MANAGERS=11`, `MAX_STEPS=10000`.
- Депо: `DEPOT_ID = "City"`, старт `INIT_MONEY=82000`, `INIT_PEOPLE=28`, `START_YEAR=1890`.

## 2. Архитектура: связь C++ игры и Python RL

Три независимых режима сборки/запуска:
1. **Консольная игра** — `sakhalin_colony.exe` (компиляция `main.cpp`, без pybind).
2. **GUI игра** — `sakhalin_colony_gui.exe` (компиляция `gui.cpp` + raylib).
3. **Python RL-модуль** — `colony_cpp.pyd` (pybind11 MODULE, компиляция `bindings.cpp` + ядро).

**Мост pybind11** (`src/bindings.cpp`):
- `PYBIND11_MODULE(colony_cpp, m)` экспонирует: `MtRandom`, `PCG64`, `Earth`,
  `load_base_data`, `load_events`, `BaseEvent`, `BaseData`, `Sunduk`, `Game`,
  `RewardConfig`, `ColonyEnvCpp` (одиночная среда), `ColonyVecEnvCpp`
  (векторизованная, пул потоков), `RunningMeanStd`.
- `ColonyVecEnvCpp` — пакетная среда: `reset_batch`, `step_async_batch`,
  `step_wait_batch`, `obs_buffer`, `save/load_normalization`, `set_rewards`,
  `set_curriculum_stage`.
- `ColonyEnvCpp` — одиночный шаг `step(action) → dict(obs, reward, terminated,
  truncated, days, people, money, bases, seed, tax_due_days, ep_return, steps)`.

**python-обёртки** (`python/`):
- `python/cpp_vecenv.py` — `CppVecEnv(gym/VecEnv)`, наследует
  `stable_baselines3.common.vec_env.VecEnv` (SB3 используется ТОЛЬКО как интерфейс).
- `python/cpp_env.py` — `CppColonyEnv(gym.Env)` одиночная среда, **не используется**
  тренировочным пайплайном (альтернативная точка входа).

## 3. Структура папок

| Путь | Назначение |
|------|-----------|
| `src/` | C++ исходники ядра: `rng.cpp` (PCG64/MtRandom), `data.cpp` (парсеры JSON), `earth.cpp` (карта/ландшафт), `game.cpp` (логика Game: день, налоги, стройка, события, game_over), `env.cpp` (RL-среда `ColonyEnvCpp` + `ColonyVecEnvCpp`), `bindings.cpp` (pybind11), `main.cpp` (консоль), `gui.cpp` (raylib GUI), `resources.cpp` (сундук). |
| `include/colony/` | Заголовки ядра: `constants.h`, `bases.h`, `events.h`, `game.h`, `env.h`, `earth.h`, `resources.h`, `data.h`, `rng.h`, `running_mean_std.h`, `thread_pool.h`. |
| `include/third_party/` | Сторонние заголовки, в т.ч. `json.hpp` (nlohmann/json). |
| `rl/` | Python RL: `config.py`, `actor_critic.py`, `rollout_buffer.py`, `ppo.py`, `env_manager.py`, `async_trainer.py`. |
| `python/` | Python-обёртки C++: `colony_cpp.pyd` (скомпилированный модуль), `cpp_vecenv.py`, `cpp_env.py`. |
| `configs/` | Данные JSON: `bases.json` (33 типа зданий), `events.json` (46 событий), `reward.json` (коэффициенты наград). |
| `ui/` | Python GUI на pygame (`main_window.py`, `render.py`). **НЕ запускается** (импорт отсутствующего `core` + нет `pygame` в `requirements.txt`). |
| `tests/` | Тесты: `test_gae.py`, `test_ppo_smoke.py`, `test_integration.py`, `bench_per_step.py`. |
| `assets/` | PNG-спрайты (`imlBases_*`, `imlEarth_*`, иконки) для raylib GUI. |
| `raylib/` | Пребилд raylib 6.0, нужен `build_gui.bat`. |
| `build/` | Пустая (артефакты не закоммичены; `.pyd` кладётся в `python/`). |
| `extracted*`, `resources_out`, `promer` | **Мусор/артефакты распаковки оригинала — игнорировать.** |

**Папки из ТЗ, которых нет в репозитории:** `scripts/`, `core/`, `env/`, `colony/`.
Их роль: `core/` — пакет оригинального Python-порта, который **отсутствует** (его
импортирует `ui/`); `env/` и `colony/` слиты в `src/` + `include/colony/`.

## 4. Поток данных (RL)

**Observation.** `ColonyEnvCpp::obs()` (`src/env.cpp:384-452`) собирает вектор
фиксированной длины `obs_size() = 203` (`env.h:71`):
- 8 скаляров: year/50, month/12, day/31, season/3, money/2e5, credit/2e5, people/100, busy_people/100;
- 9 (`SUNDUK_SIZE`): ресурсы `sunduk[i]/100`;
- 10: home_places/100, need_workers/100, free_people/100, перенаселение/100, флаги налогов, суммы налогов, days_alive/3650, curriculum_stage/3;
- `n_build=32`: счётчики типов зданий/10;
- 7: суммарный износ, min live_frac, строящиеся/законсервированные/ветхие/простаивающие;
- 9: цены продажи (`sale_prices_`);
- `4*n_build=128`: каталог по сезонам `catalog_by_season_`.

Формула из `env.h:71`: `27 + n_build + 7 + 9 + 4*n_build = 43 + 5*32 = 203`.

**Action.** Дискретное `Discrete(45)` (`env.h:70`):
- `0=DAY`, `1=WEEK`;
- `2..33` (32 значения) = постройка `build_ids_[i]`;
- `34..44` (11 значений `N_MANAGERS`) = менеджмент: улучшить землю, ремонт, ремонт всех,
  снос, консервация, расконсервация, продажа излишков, покупка еды, кредит, возврат
  кредита, ручной налог.

**Reward.** `ColonyEnvCpp::step()` (`src/env.cpp:454-712`), `RewardConfig` (`env.h:20-32`):
авто-налоги, бонус за постройку `build_bonus*year_production_value`, штрафы за
неверные действия (-1.0), компонент чистой стоимости `0.005*(net_worth-net0-built_price)`,
родившиеся/умершие/потерянные базы, цепочки потребления `chain_bonus`/`chain_daily`,
ежедневный доход `daily_income*daily_total`, novelty-бонус, штраф game_over, survival_bonus.

**Обучение (один rollout):**
1. `EnvManager.reset()` → `CppVecEnv.reset()` → `ColonyVecEnvCpp.reset_batch(seeds)`
   (C++ нормализация obs через `RunningMeanStd`).
2. Для каждого из `n_steps` шагов: `ActorCritic.get_action_and_value(obs_gpu)` (no_grad)
   → `CppVecEnv.step_async(actions)` → `step_wait_batch` (auto-reset завершённых сред)
   → `RolloutBuffer.add` (GPU-тензоры).
3. После `n_steps`: `RolloutBuffer.compute_gae` (GPU) → `PPO.update`:
   `n_epochs × minibatches`, loss `−min(ratio·A, clip(ratio)·A) + vf_coef·value_loss − ent_coef·entropy`,
   Adam, `clip_grad_norm_(0.5)`, AMP bf16.
4. Цикл повторяется пока `total_timesteps` не достигнут.

**Сохранение весов.** `PPO.save()` пишет `torch.save({model_state, optimizer_state,
buffer_pos}, path)`. Точки: `checkpoint_{total_done}_steps.pt` и `final_model.pt`.
Каталог по умолчанию — `~/colony_runs/models/<name>`.
⚠️ Нормализация obs/reward (`RunningMeanStd`) **НЕ сохраняется** `train.py` (метод
`save_normalization` есть в C++, но не вызывается из Python) — статы не переносятся
между запусками.

## 5. Точки входа

**Игра (консоль):** `build_exe.bat` → `sakhalin_colony.exe`.
Запуск: `sakhalin_colony.exe [--seed N] [--map-size N] [--stage N]`.

**Игра (GUI/raylib):** `build_gui.bat` → `sakhalin_colony_gui.exe`.

**Сборка RL-модуля:** `CMakeLists.txt` → `add_library(colony_cpp MODULE … bindings.cpp …)`
→ `python/colony_cpp.pyd`. Требует `pybind11` из pip.

**Обучение:** `train.py`
```
python train.py --steps 1000000 --envs 8
python train.py --steps 10000000 --envs 32 --n-epochs 5 --batch-size 16384
python train.py --compile --amp bfloat16 --reward-config configs/reward.json
```
Аргументы: `--steps, --envs, --map-size, --n-steps, --batch-size, --n-epochs, --lr,
--gamma, --gae-lambda, --clip-range, --ent-coef, --vf-coef, --max-grad-norm, --net-arch,
--device, --amp, --compile, --async, --queue-size, --cpp-threads, --torch-threads,
--save-freq, --eval-freq, --eval-episodes, --log-dir, --model-dir, --name,
--reward-config, --disable-net-worth, --disable-daily-income`.

⚠️ `train.py` загружает `reward.json` **только если передан `--reward-config`**; иначе
используются значения по умолчанию из `RewardConfig` (`env.h`/`config.py`), которые
**отличаются** от `reward.json` (см. `review/04_data_config_inconsistencies.md`).

## 6. Конфиги и данные

- `configs/bases.json` (33 типа зданий): `id, caption, price, build_time, live_years,
  home_places, need_workers, need_earth, work_seasons[4], consume{}, profit{},
  profit_range{}, style[], image_index`.
- `configs/events.json` (46 событий): `id, target, message, live_years, live_years_range,
  people, people_range, sunduk{}, sunduk_range{}`.
- `configs/reward.json` — см. `review/04_data_config_inconsistencies.md` (несоответствия).

## 7. Граф зависимостей

```
train.py
  └─ rl/config.py            (Config, RewardConfig)
  └─ rl/env_manager.py       (EnvManager: CppVecEnv + ActorCritic + RolloutBuffer + PPO)
        ├─ python/cpp_vecenv.py  (CppVecEnv ← stable_baselines3 VecEnv)
        │     └─ colony_cpp.ColonyVecEnvCpp   [C++ .pyd]
        │           ├─ colony_cpp.load_base_data(configs/bases.json)
        │           ├─ colony_cpp.load_events(configs/events.json)
        │           └─ N × ColonyEnvCpp → Game (game.cpp) → BaseData/BaseEvent (data.cpp) → Earth (earth.cpp)
        ├─ rl/actor_critic.py     (ActorCritic: MLP 203→256→256→45, ortho-init)
        ├─ rl/rollout_buffer.py   (RolloutBuffer: GPU GAE, mini-batch)
        └─ rl/ppo.py              (PPO: loss, Adam, AMP, save/load .pt)
  └─ rl/async_trainer.py      (AsyncTrainer: rollout→update→checkpoint→TensorBoard)

colony_cpp.pyd (src/bindings.cpp)
  ├─ src/env.cpp ├─ src/game.cpp ├─ src/earth.cpp ├─ src/data.cpp ├─ src/resources.cpp ├─ src/rng.cpp
  └─ include/colony/*.h

src/main.cpp  ── ColonyEnvCpp (консоль)            → sakhalin_colony.exe
src/gui.cpp   ── ColonyEnvCpp + raylib             → sakhalin_colony_gui.exe
ui/main_window.py ── (БИТО: импорт отсутствующего core.* + pygame)
```

## 8. Конвенции

- Язык кода: C++ и Python с комментариями/строками на **русском**. Идентификаторы — английские.
- C++ стиль: `namespace colony`; заголовочники в `include/colony/`; RAII/`shared_ptr` для
  неизменяемых статических данных; константы `constexpr` в `constants.h`. Стандарт C++17.
- pybind11: GIL освобождается в `step_async_batch/step_wait_batch/reset_batch` через
  `py::gil_scoped_release` для параллельного C++-шага.
- RL: гиперпараметры через `Config` dataclass + CLI; обучение на GPU; `VecNormalize`
  реализован внутри C++ (`ColonyVecEnvCpp`), а не через SB3.
- Логирование: `print` + опционально TensorBoard (`train/fps, train/policy_loss,
  train/value_loss, train/entropy, train/approx_kl, train/learning_rate, train/best_reward`).
- Тесты: pytest-совместимые файлы в `tests/`.

## 9. Известные проблемы (кратко)

Полный отчёт — в `review/`. Критические:
- **RL (КРИТ):** неверная маска терминальности в GAE (`dones[t+1]` вместо `dones[t]`)
  и `last_done`, посчитанный как OR по роллауту — систематически портит advantages.
- **RL (КРИТ):** `device_type="cuda"` жёстко зашит в AMP → падение на CPU.
- **C++ GUI (КРИТ):** диалоги, открытые мышью, мгновенно закрываются; висячий
  `selected_base` (use-after-free).
- **C++ main (КРИТ):** off-by-one в нумерации меню → OOB-чтение и недостижимый Quit.
- **C++ env (КРИТ/MAJOR):** `reset()` не сбрасывает кэш чистой стоимости → искажение
  награды на старте каждого эпизода.
