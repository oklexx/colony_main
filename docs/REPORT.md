# Аудит проекта Sakhalin Colony (colony_main) — 11.09.2026

Проверка каждой функции и цепочки: UI → worker → Config → EnvManager → C++ → PPO → eval → champion → watch.
Метод: чтение всего кода, статический анализ скриптами, прогон тестов (там, где возможно без Windows/GPU),
синтаксическая проверка C++ через g++.

**Стенд проверки:** Linux-песочница, Python 3.13, torch 2.14 CPU (установлен для тестов).
`colony_cpp.pyd` — Windows-бинарник, в песочнице не импортируется, поэтому C++-рантайм проверен
чтением кода + `g++ -fsyntax-only`, а Python-логика — реальными тестами.

**Итог:** проект в целом связный и рабочий, обучение по задумке идти должно. Но найдено
**6 критических багов** (2 из них прямо отвечают на «не вижу параметров»), 10 средних и ряд мелких.
Все исправления, которые безопасно вносить без смены дизайна, собраны в **`fixes.patch`** (27 файлов).
Два вопроса дизайна оставлены на решение автора (помечены ⚖).

---

## 1. Ответ на «не вижу параметров»

Причин три, все подтвердились кодом:

1. **4 параметра наград v3 есть в коде, но их нет на вкладке «Награды»** (`train_ui2/main_window.py`,
   `REWARD_GROUPS`). Это `first_extraction_bonus`, `extraction_daily`, `need_fill_bonus`, `loan_penalty`.
   Спецификации в `parameter_widget.py` (`REWARD_SPECS`) есть, а в группы вкладки их забыли добавить —
   спиннеры не создаются, `_collect_config()` их не собирает, обучение всегда идёт с дефолтами.
   → **Исправлено в патче**: новая группа «Добыча и кредиты».
2. **`preserve_penalty` обрезан Qt-диапазоном.** Дефолт v3 = `0.3`, а спиннер имел диапазон `[-50, 0]`
   (остаток от v2, где ключ был мёртвым). Qt молча клампит значение к `0.0` — обучение через UI
   шло с бесплатным preserve/unpreserve вместо платы 0.3. → **Исправлено**: диапазон `[0, 50]`.
3. **Тест, который должен был это ловить, никогда не выполнялся.**
   `tests/test_reward_default_profile.py::test_ui_reward_specs_cover_profile` импортирует несуществующий
   `train_ui.parameter_widget` (пакет переименован в `train_ui2`), `ImportError` глушится в `pytest.skip`.
   На машине автора тест тоже скипается — баг проскочил. → **Исправлено**: импорт `train_ui2` + добавлены
   проверки диапазонов и покрытия `REWARD_GROUPS`.

Дополнительно: в UI штатно отсутствуют продвинутые параметры — `device` (CPU/CUDA),
`eval_seeds`, детектор циклов (`loop_detection_*`), `curriculum_schedule`, `unlock_ids`,
`queue_size`/`async_train`. Это не баги, а пробелы функциональности (см. §6).

---

## 2. Проверенные цепочки (end-to-end)

| # | Цепочка | Вердикт |
|---|---|---|
| 1 | UI Параметры/Награды → `_collect_config` → JSON → worker `Config.from_dict` → C++ | ✅ после патча (были п.1–2 выше); smoke-тест §3 подтверждает: ключи не теряются |
| 2 | `train.py` CLI → Config → EnvManager → AsyncTrainer → чекпойнты/meta | ✅, кроме лога действий (писал `action=-1`, исправлено) |
| 3 | PPO: `collect_step` → буфер → GAE → `update` (AMP/compile/маски/ранний стоп) | ✅ полностью корректно, покрыто тестами |
| 4 | Маски действий C++ → PPO/eval/watch | ✅ в обучении и eval; в headless-watch отсутствовали — добавлено |
| 5 | Нормализация obs (C++ RMS train / Normalizer eval) + `.norm.json` sidecar'ы | ✅ форматы совместимы; resume предупреждает, если файла нет |
| 6 | Нормализация наград (RMS только в train) | ✅ так задумано (SB3-конвенция) |
| 7 | Eval: `_eval` → `run_eval` (argmax+маски, заморозка RMS) → score/пороги → best_model | ✅, кроме: eval игнорировал 4 v3-ключа и флаги disable_* (исправлено) |
| 8 | Финальный турнир чекпойнтов → `best_model.pt` | ✅ после патча (были NameError-фолбэк и рассинхрон meta.json) |
| 9 | Curriculum: UI/stage/schedule → C++ cumulative unlocks → маски | ✅ механика; UI давал стадии 4–5 = молча 3 (исправлено на 0–3) |
| 10 | Difficulty light (×деньги, без главного налога, финал 1950) | ⚖ работает, но финал 1950 = terminated+штраф (см. §5) |
| 11 | Watch текстовый:-argmax→IPC-JSONL→UI-панель | ✅ после патча (маски, difficulty из meta) |
| 12 | Watch визуальный: GUI exe `--headless-ai` ↔ actions/state файлы | ✅ аргументы с обеих сторон совпадают; сетка миникарты выводится из данных |
| 13 | Optuna (`auto_trainer.py`) → train.py → best_score | ✅ ключи читаются верно |
| 14 | `training_cycle.py` (30 эксп.) → отчёт | ✅ ключи читаются верно |
| 15 | Свипы `sweep2–7`, `reward_sweep.py` → worker | ❌ были сломаны (путь `train_ui/worker.py`), исправлено на `train_ui2` |
| 16 | `observe.py` (лог поведения) | ⚠ чинится частично: best_model.pt в приоритете поиска; но скрипт только для MLP-моделей (CNN/hybrid упадут на `load_state_dict`) |
| 17 | Сборка: CMake → `colony_cpp.pyd`, `build_pyext.bat` | ✅ логика верна; .bat захардкожены под машину автора |
| 18 | Тесты `pytest` | ✅ после патча: 113 passed; 21 падение — только отсутствие `colony_cpp`/gymnasium/PySide6 в Linux-песочнице |

---

## 3. Критические баги (P0) — все исправлены в патче

### P0-1. `BUY_FOOD` награждался вместо штрафа — агент учился сливать деньги
`src/env.cpp`, ветка `MGR:buy_food`: комментарий говорит «покупка еды — чистый слив денег, штраф»,
а код делал `rew += cfg_.buy_food_penalty` при дефолте `+3.0`. Каждое действие BUY_FOOD давало **+3**.
Исправление: `rew -= …`. ⚠ Требует **пересборки** `colony_cpp.pyd` (`build_pyext.bat`), иначе старый
бинарник сохранит баг.

### P0-2. В UI не видны 4 параметра наград v3
`train_ui2/main_window.py`, `REWARD_GROUPS`: добавлена группа «Добыча и кредиты»
(`first_extraction_bonus`, `extraction_daily`, `need_fill_bonus`, `loan_penalty`).
Заодно подпись «Профиль v2» → «Профиль v3» (дефолты давно v3).

### P0-3. `preserve_penalty=0.3` обрезался спиннером до 0
`train_ui2/parameter_widget.py`: диапазон `[-50, 0]` → `[0, 50]`, шаг `0.1`, подсказка переписана
под семантику «плата вычитается». Удалена дублированная `spec_for()` в конце файла.

### P0-4. Eval/watch игнорировали v3-награды и флаги отключения
`python/cpp_env.py` (одиночная среда для eval/watch):
- в `_REWARD_KEYS` не было 4 v3-ключей — кастомные значения из JSON/профиля молча отбрасывались
  (тренировка шла с одними наградами, оценка — с другими);
- `rc.disable_net_worth/disable_daily_income = False` перезаписывали значения из словаря безусловно.
Исправлено: ключи добавлены, флаги применяются только если переданы явно (`None` по умолчанию).

### P0-5. Бонусы еды/воды росли от простоя, а не от нехватки
`src/env.cpp`: `food_need_bonus`/`water_need_bonus` масштабировались от `days_since_last_build_`
(копипаст-ошибка) — чем дольше агент ничего не строил, тем больше получал за ферму.
Исправлено на нехватку: `log1p((200-food)/10)`, `log1p((100-water)/10)` — по образцу жилищного бонуса.
⚠ Меняет ландшафт наград → требует пересборки `.pyd`; старые прогоны с этим не сравнивать.

### P0-6. Тест-привратник дефолтов никогда не запускался + ломал сессию сосед
- `tests/test_reward_default_profile.py`: импорт `train_ui.*` → `train_ui2.*`, плюс новые проверки:
  диапазоны спиннеров покрывают профиль; все ключи профиля есть в `REWARD_GROUPS`.
- `tests/test_smoke_fixes.py`: ожидал дефолты **v2** (устарел с переходом на v3 → 2 провала) и делал
  `sys.exit(1)` на уровне модуля → убивал **всю** pytest-сессию (`INTERNALERROR`). Переведён на v3,
  выход под pytest заменён исключением (ошибка только этого модуля).

---

## 4. Средние баги (P1) — все исправлены в патче

| # | Где | Что было | Исправление |
|---|---|---|---|
| P1-1 | `rl/async_trainer.py`, турнир | Фолбэк `norm_str` ссылался на локальную переменную `_eval` → `NameError`, если у кандидата нет `.norm.json` | Своя переменная `tournament_norm_str` |
| P1-2 | там же | Турнир перезаписывал `best_model.pt`, но не `best_model.meta.json` → вкладка «Модели» показывала stale score/дни/базы | Meta дописывается (score/дни/базы/люди/return/победитель) |
| P1-3 | `rl/config.py`, `from_dict` | Фолбэки `gamma=0.995`, `ent_coef=0.05` ≠ дефолтам датакласса `0.999`/`0.01` (било по JSON-конфигам без этих ключей) | Синхронизировано |
| P1-4 | `watch_champion.py`, текстовый режим | Не применялись маски действий (eval и обучение их применяют) — просмотр показывал более глупую игру, чем eval | Маски `-1e9` во всех трёх ветках (flat/cnn/hybrid) |
| P1-5 | `watch_champion.py` | `difficulty` из meta.json не читался (eval читает) — модель с `light` смотрели на `normal` | Читается и передаётся в среду |
| P1-6 | `rl/env_manager.py` | `get_allowed_buildings_for_stage` (подпись «доступные действия» в UI) не совпадал с C++ ни составом, ни именами (`BUILD_WATER_CHANNEL` вместо `BUILD_WATERCHANNEL`) | Переписан как зеркало C++ `CURRICULUM_STAGE_1/2/3` (накопительно) |
| P1-7 | UI, комбо «Курикулум» | Значения 0–5, а в C++ только 0–3 (4–5 молча = 3) | 0–3 + кламп восстановленного состояния |
| P1-8 | `train.py --log-actions` | Читал несуществующий `em.env._last_actions` → в `actions.log` всегда `action=-1` | Действия берутся из rollout-буфера |
| P1-9 | `python/cpp_vecenv.py` | Перезаписывал корректный RAW `terminal_observation` из C++ свежим obs после авто-ресета | Не перезаписывать, если ключ уже есть |
| P1-10 | `requirements.txt` | `torch>=2.13.0` — требование bleeding-edge (сегодня ставится, но хрупко); нет `optuna` (нужен `auto_trainer.py`) | `torch>=2.7.0` + `optuna>=3.0.0` |
| P1-11 | `configs/reward_v3.json` | Нет `buy_food_penalty` (работало «случайно» через дефолт датакласса) | Ключ добавлен (=3.0) |
| P1-12 | 7 свип-скриптов | `WORKER = train_ui/worker.py` — файла нет после переименования → все свипы падают | Путь `train_ui2/worker.py` (CLI совместим) |
| P1-13 | `observe.py` | `find_model` искал `best.pt` (никем не создаётся) и `final_model.pt`, игнорируя `best_model.pt` | Приоритет `best_model.pt` → `final_model.pt` → `best.pt` → чекпойнты |
| P1-14 | `train_ui2/parameter_widget.py` | `idle_build_threshold_days` (int) был float-спиннером | `is_int=True` |
| P1-15 | `train_ui2/evaluator.py` | В лог писался `seed+ep`, реально использовался `seed+500000+ep` | Лог пишет true seed |

Мелкие (P2, тоже в патче): `209 → 246` в 5 комментариях/docs (формула obs при 32 постройках:
27+32+7+9+4·32+9+32+2 = 246); docstring `worker.py` и комментарий в `watch_champion.py`
(`train_ui` → `train_ui2`); README — большой блок правок (см. §7).

---

## 5. Вопросы дизайна — требуют решения автора (НЕ патчились)

1. **Финал `light`-режима (1 марта 1950) = `terminated + game_over_penalty`.**
   `src/game.cpp::game_over()` возвращает `light_end`, `env.cpp` штрафует как смерть (−10).
   Дожить до финала — успех, а агент получает наказание за него. Предложение: для `light_end`
   ставить `truncated=true` без штрафа (GAE тогда корректно бутстрэпится через value).
2. **Потеря города (depot) никогда не завершает эпизод в RL и GUI.**
   `ColonyEnvCpp` и `gui.cpp` создают среду с `no_city_game_over=false` (дефолт), а условие
   срабатывания — `no_city_game_over_ && !depot_exists()`. Только прямой класс `Game` (дефолт `true`)
   завершается. Если для RL это осознанно (не рвать эпизод) — ок, но для GUI это выглядит багом.
   Имя флага двусмысленно («no game over» vs «game over on no-city») — стоит переименовать.
3. **-survival_bonus начисляется и в терминальный шаг** (`env.cpp`, после блока терминалов).
   При дефолте 0.0 безвредно; если включите бонус — будет платить и за шаг смерти.

---

## 6. Проверено и работает как задумано (выборочно, важное)

- **PPO/GAE**: маски `-1e9`, сэмплирование train vs argmax eval, нормализация advantages,
  transition-based `terminated` для бутстрэпа при truncation, ранний стоп `1.5×target_kl`,
  AMP только на CUDA, `torch.compile` с откатом на eager (на CPU/Windows — без падения),
  сохранение без `_orig_mod`-префиксов, LR cosine-decay до ~10% (старая жалоба B7 из README
  устарела — scheduler на месте). Покрыто `test_gae`, `test_ppo_smoke`, `test_smoke_fixes` — **проходят**.
- **C++ arreна**: авто-ресет с `terminal_observation` (RAW) и `episode{r,l}` в info; RMS-нормализация
  obs/reward по конвенции SB3; curriculum cumulative unlocks 1→3 (= все 32 на stage 3);
  `TAX_GRACE_DAYS=60` → terminated; `MAX_STEPS=10000` → truncated; кредит сверх лимита → terminated.
- **Eval/чемпион**: детерминированный argmax с масками, заморозка RMS, медиана по эпизодам×сидам,
  score `0.10·days + 1.0·bases + 0.10·people + 1e-4·max(0,return) − штраф дисперсии`,
  пороги `bases≥5, p25≥3.5, days≥730`, `_eval/`-песочница с собственным meta (защита от stale reward).
- **Minimap/hybrid**: радиус синхронизируется train→eval→watch; CNN нечувствителен к сетке
  (`AdaptiveAvgPool`), hybrid защищён выставлением радиуса из чекпойнта.
- **UI↔worker**: все поля `ProgressMsg` разбираются мониторингом; команды
  `boost_entropy/pause/resume/stop/reset_curriculum` доходят (файловый канал, имена совпадают);
  конфиг собирается полностью (smoke-тест §3 зелёный).
- **GUI-режим**: аргументы `--headless-ai/--actions-file/--state-file/--seed/--map-size/--stage/
  --reward-config/--minimap-radius` совпадают с парсером `gui.cpp`; поиск exe по 7 путям + `COLONY_GUI_EXE`.
- **Знаки наград** (после патча): `manual_tax −0.5` (старый N1 из README давно исправлен),
  `demolish −3`, `error −2`, `preserve −0.3` (вычитание), `loan −0.5`, `buy_food −3`,
  `game_over −10`, налог/долг/смерти/потери — вычитания. Инверсий больше нет.

---

## 7. Документация и гигиена

- **README.md** был сильно устаревшим: пути `train_ui/`, команда `python -m train_ui.app`,
  таблица наград со значениями v1 (`build 1.0`, `diversity 8.0`, `novelty 15`, `sale 0.2`, `error −1`,
  `idle −10`, `survival_coeff 0.01`), «хардкоды» §6.2 (давно в конфиге), curriculum 1–5,
  формула score `0.4/3.0/0.2`, виджеты старого UI в §9, ссылки на несуществующие
  `PROJECT_AUDIT_FULL.md`/`IMPLEMENTATION_PLAN.md`/`FIXES_P0_P3.md`/`fix_01..`. Всё это **исправлено
  в патче** под фактический код.
- `HOW_TO_RUN_UI.md`, `README_UI_READY.md`, `test_ui_improvements.py` описывают старый UI —
  рекомендую удалить/архивировать (в патч не включены, чтобы не уничтожать историю).
- В корне мусор отладки: `_res*.py`, `_trace*.py`, `_dump.py`, `_probe.py`, `temp_*.txt`,
  `result.txt`, `check_result.txt`, `code_dump/`, `obs_logs/`, `build/`, `x64/`, `*.vcxproj`,
  `CMakeCache.txt`, скомпилированные `.pyd/.exe/.dll`. Рекомендую `.gitignore` + чистку.
- `build_*.bat` захардкожены (`C:\Users\oklex\…`, `C:\Python314`, `Visual Studio 18 2026`).
  Работоспособно на машине автора, но непереносимо — лучше `%~dp0` + поиск `vswhere`.
- `pyqtgraph` в requirements нигде не импортируется; `launch_exp25.py` захардкожен под `C:\…`
  и передаёт `--compile` (на Windows честно откатится в eager с варнингом — PPO это держит).
- В UI нет: выбора `device`, `eval_seeds`, детектора циклов, редактора `curriculum_schedule`,
  `unlock_ids`. Через UI также не пишется TensorBoard (в отличие от `train.py`).
- `exp_*.json`: `n_envs=1024` требует десятки ГБ VRAM; `gamma 0.99`/`net_arch [512]` — старые значения;
  секции `reward` нет (ок — подхватится v3).

---

## 8. Тесты

| Набор | Результат в песочнице |
|---|---|
| `test_protocol`, `test_models`, `test_worker`, `test_gae`, `test_ppo_smoke`, `test_hybrid`, `test_parameter_widget`, `test_async_trainer`, `test_smoke_fixes`, `test_reward_default_profile` (слои 1–4, 6 — скип без PySide) | ✅ **113 passed**, 0 падений логики |
| 21 тест + 2 модуля | ⏭ Падают/скипаются **только** из-за отсутствия `colony_cpp` (Windows `.pyd`), `gymnasium`, PySide6 в Linux-песочнице. На машине автора должны идти; перед прогоном убедитесь, что `.pyd` пересобран |
| `g++ -fsyntax-only src/env.cpp` | ✅ синтаксис C++-правок корректен |

---

## 9. Как применить

```bat
cd C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main
git apply fixes.patch
REM C++ менялся (buy_food, food/water) — обязательно пересобрать:
build_pyext.bat
REM Проверка:
python -m pytest -q
python run_train_ui2.py
```

Порядок важен: Python-фиксы работают сразу, C++-фиксы — только после пересборки `.pyd`.
После пересборки старые чекпойнты валидны (меняется reward, не obs/архитектура), но кривые
наград с прошлыми прогонами напрямую не сравнивать (P0-1, P0-5 меняют значения).

---

## 10. Что осталось за кадром (нужны Windows + GPU)

Полный прогон обучения до сходимости, окно GUI, `torch.compile` на CUDA, FPS/профилирование,
игровой баланс (экономика/налоги/сезоны в `game.cpp` проверены только на уровне интерфейса
с RL: коды возврата, `new_day`/`advance_week`, рынок, кредиты — вся обвязка вызывается корректно).
Если хотите — следующим шагом могу: добавить недостающие параметры в UI, почистить корень +
`.gitignore`, вынести `light_end` в truncation, добавить TensorBoard в worker.
