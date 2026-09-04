# РЕВЬЮ 04 — Несоответствия данных и конфигов (configs/, ui/, README)

**Объём:** `configs/bases.json`, `configs/events.json`, `configs/reward.json`,
`requirements.txt`, `ui/`, отсутствующий `README.md`, докстринги.
**Игнорируемые папки:** `extracted*`, `resources_out`, `promer`.

---

## CRITICAL / MAJOR (данные)

### [M-D1] `configs/reward.json` противоречит дефолтам кода (5 полей) + мёртвый блок

**Место:** `configs/reward.json` против `env.h:20-32` и `rl/config.py:9-21`.

| Ключ | `reward.json` | Код по умолчанию | Статус |
|------|--------------|------------------|--------|
| `build_bonus` | 5.0 | 5.0 | совп. |
| `chain_bonus` | **1.0** | **0.5** | **расхождение** |
| `chain_daily` | 2.0 | 2.0 | совп. |
| `novelty` | 20.0 | 20.0 | совп. |
| `daily_income` | **1.0** | **0.1** | **расхождение ×10** |
| `sale_bonus` | **0.5** | **0.1** | **расхождение** |
| `tax_bonus` | 250.0 | 250.0 | совп. |
| `survival_bonus` | **0.1** | **0.0** | **расхождение** |
| `game_over_penalty` | **50.0** | **20.0** | **расхождение** |
| `disable_net_worth` | false | false | совп. |
| `disable_daily_income` | false | false | совп. |
| `curriculum_rewards` | `{1,2,3:{…}}` | **нет в коде** | **мёртвые данные** |

**Критические выводы:**
1. `reward.json` — это **не дефолт**: `train.py` берёт его только при `--reward-config`
   (`train.py:84-87`). Без флага обучение идёт на значениях из `env.h`/`config.py` (иных, см.
   таблицу). Исследователь, меняющий `reward.json` и запускающий `train.py` без флага, **не
   увидит эффекта** своих правок — классическая ловушка.
2. Блок `curriculum_rewards` **нигде не читается**: `RewardConfig.from_dict`
   (`config.py:38-52`) берёт только 11 базовых ключей; курикулум в C++ задаётся целым
   `curriculum_stage` (0..3, `env.cpp:17-27,57-71`), а не пер-стадийными надбавками наград.
   То есть `curriculum_rewards` в `reward.json` — **неиспользуемые/мёртвые данные**, создающие
   ложное впечатление, что они применяются.
3. `REPORT.md:111-124` задокументировал дефолты, совпадающие с кодом (`chain_bonus 0.5`,
   `daily_income 0.1`, `survival_bonus 0.0`, `game_over_penalty 20.0`) — то есть `reward.json`
   **противоречит и коду, и REPORT.md**.

**Рекомендация:** либо сделать `reward.json` единственным источником правды (всегда загружать
его и убрать дефолты-дубли), либо синхронизировать числа и удалить мёртвый `curriculum_rewards`.
Добавить в `train.py` явное предупреждение, если `--reward-config` не передан.
**Серьёзность: major.**

---

## MAJOR / MINOR (структура и запуск)

### [M-D2] `ui/` не запускается (отсутствующий `core` + нет `pygame`)

**Место:** `ui/main_window.py:15-19`
```python
from core import constants, bases, game, resources, save
```
Папки `core/` в репозитории **нет**. Кроме того требуется `pygame`, которого **нет в
`requirements.txt`**. Рабочий GUI — только C++/raylib (`gui.cpp`), собираемый `build_gui.bat`.
Любой, кто попытается запустить `python ui/main_window.py`, получит `ModuleNotFoundError`.
**Серьёзность: major** (функционал заявлен, но неработоспособен).

---

### [M-D3] Отсутствует `README.md` (есть только `REPORT.md`)

Корневой `README.md` нет. Пользователь/агент вынужден восстанавливать структуру по коду.
Сгенерирован полный `README.md` в рамках этого ревью (см. корень проекта).
**Серьёзность: minor** (документация).

---

### [M-D4] `cpp_env.py:15` докстринг лжёт про размерность obs

**Место:** `python/cpp_env.py:14-15`
```python
# Observation space: 207-dim
# Action space: Discrete(45)
```
Реально `obs_size = 203` (см. `review/01`/`02`), `Discrete(45)` верно. Устаревший комментарий
вводит в заблуждение при интеграции.
**Серьёзность: minor.**

---

## MINOR (данные/конфиги)

### [m-D1] `BaseEvent` (pybind) не экспонирует `sunduk`/`sunduk_range` (bindings.cpp)
`bindings.cpp:123-130` экспонирует только `id/target/message/live_years*/people*`. Функционально
не мешает — C++ применяет сундук внутри (`BaseEvent::execute`, `events.h:30-37`), — но любой
Python-код, интроспектирующий события, не увидит ресурсных эффектов. Документировать или
расширить биндинг.

### [m-D2] `requirements.txt` неполный / избыточный
- Заявлены `gymnasium` + `stable-baselines3`, но SB3 используется лишь как базовый класс
  `VecEnv` (`cpp_vecenv.py:6`); собственный PPO не зависит от SB3.
- `pygame` **отсутствует** (нужен для `ui/`, см. M-D2).
- `torch>=2.13.0` — очень свежая версия (для Python 3.14).
**Серьёзность: minor.**

### [m-D3] Порядок перечисления `Season` (constants.h vs data.cpp)
`include/colony/constants.h:56`: `enum Season { SEASON_SUMMER=0, SEASON_AUTUMN=1, SEASON_WINTER=2,
SEASON_SPRING=3 }` (порядок «лето=0»). А `src/data.cpp:50-56` парсит `work_seasons` в порядке
`summer/autumn/winter/spring`. Внутренне согласовано, но порядок enum «лето=0» может сбить с
толку при правке и конфликтует с кодировкой сезона в `obs()` (см. m-E5 в `01_game_logic_cpp.md`).
**Серьёзность: minor.**

### [m-D4] `build/` пуст — артефакты не закоммичены
`.pyd` попадает в `python/` (`CMakeLists.txt:37-40`), объектные файлы не в репозитории. Это
нормально, но стоит добавить `.gitignore` для `build/`, `*.pyd`, `*.exe`.

### [m-D5] Две параллельные среды-обёртки
`CppVecEnv` (используется) и `CppColonyEnv` (`cpp_env.py`, не используется тренировкой) —
дублирование интерфейса. Рекомендуется удалить/задокументировать неиспользуемую.

---

## Проверка `bases.json` / `events.json` (на соответствие коду)

- `configs/bases.json`: 33 типа зданий, `image_index` 0..32 совпадает со спрайтами
  `imlBases_00..32` (assets + ui/assets) — **OK**. Все 32 не-городских `id` входят в
  `BUILD_SUBSET` (`constants.h:136-143`) — **OK**, `n_build=32`. `City` — депо, исключается
  из действий постройки (`env.cpp:47`) — **OK**.
- `configs/events.json`: 46 событий, все `target` ссылаются на существующие `id` из `bases.json`
  (Goldmine, Refinery, Sawmill, City, WaterChannel, Coalmine, Garden, PowerStation, Ironmine,
  HydroStation, HuntingLand, Apiary, BigFarm, Mushroom, SmallHouse, House) — **OK**, нет
  висячих ссылок. `sunduk`/`sunduk_range` загружаются и применяются (`data.cpp:71-91`).
- **Расхождений в ID зданий/событий с кодом на данный момент не найдено** — но связь stringly-typed
  (см. m-E4 в `01_game_logic_cpp.md`) хрупкая.

---

## Итоговый список расхождений (для правки)
1. `reward.json` противоречит дефолтам кода по 5 полям и содержит мёртвый блок `curriculum_rewards` (M-D1).
2. `ui/` неработоспособен: импорт отсутствующего `core` + нет `pygame` в `requirements.txt` (M-D2).
3. `cpp_env.py:15` — неверная размерность obs (207 вместо 203) (M-D4).
4. Отсутствует `README.md` (теперь создан) (M-D3).
5. `BaseEvent` (pybind) не экспонирует `sunduk`/`sunduk_range` (m-D1).
6. Нормализация obs/reward (`RunningMeanStd`) не сохраняется `train.py` (см. m-RL7 в `03_rl_training.md`).

---

## VERDICT (по этому файлу): REQUEST CHANGES
Обязательно: M-D1 (синхронизация reward.json), M-D2 (починить или задокументировать ui/).
