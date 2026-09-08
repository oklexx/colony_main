# Sakhalin Colony — C++ игра + PPO-обучение

Пошаговая экономическая стратегия «Сахалинская колония» (оригинал — Жмулевский
Григорий, v3.47) с собственным GPU-тренажёром PPO поверх неё. Колония на
острове: стройка, ресурсы, население (еда/вода/жильё/работа), события и два
налога — годовой (каждое 1 марта, с 1891) и главный (каждые 10 лет, 1 ноября
1900/1910/…). Агент учится переживать их и расти.

> Свежий разбор багов и исправлений: **[REPORT_2026_09.md](REPORT_2026_09.md)**.
> Механика смертей/налогов и методология свипов: `SWEEP_ANALYSIS` (во внешних
> материалах) и раздел «Экономика выживания» ниже.

---

## 1. Состав репозитория

```
src/                  C++17 ядро игры
  game.cpp            Game: экономика, налоги, события, население
  env.cpp             ColonyEnvCpp (RL-обёртка: obs, rewards, normalizer)
                      + ColonyVecEnvCpp (пакетная, пул потоков)
  bindings.cpp        pybind11-модуль colony_cpp
  gui.cpp             raylib-GUI (игра + headless-AI режим для наблюдения)
  main.cpp            консольная версия
  rng/data/earth/resources.cpp
include/colony/       заголовки (env.h = RewardConfig, constants.h)
python/
  colony_cpp.pyd      собранный pybind-модуль (Windows; пересборка §3)
  cpp_env.py          CppColonyEnv — одиночная среда + Normalizer
  cpp_vecenv.py       CppVecEnv — векторная среда (SB3 VecEnv-интерфейс)
  minimap.py          врапперы миникарты (minimap/hybrid obs-режимы)
rl/
  config.py           Config + RewardConfig (ЕДИНСТВЕННЫЙ источник дефолтов)
  ppo.py              PPO: AMP, torch.compile, маски действий, KL early-stop
  actor_critic*.py    MLP / CNN / hybrid сети
  rollout_buffer.py   GPU-буфер + GAE (terminated vs done)
  env_manager.py      сборка env+модель+буфер+PPO
  async_trainer.py    тренировочный цикл, eval, best_model, чекпоинты
  loop_detector.py    детектор зацикливания действий
train_ui/             PySide6 UI обучения (worker.py — отдельный процесс)
configs/              bases.json, events.json, reward.json (=v2), reward_v2.json
tests/                test_smoke_fixes.py — 27 проверок без GPU и C++
train.py              CLI-обучение
watch_champion.py     наблюдение за моделью (текст / raylib-GUI)
observe.py            пошаговый лог поведения модели
build_pyext.bat       сборка colony_cpp.pyd (CMake+MSVC)
build_gui.bat         сборка sakhalin_colony_gui.exe (MSVC+raylib)
```

Ключевые числа: `obs_size = 54 + 6·n_build = 246` (при 32 постройках),
`n_actions = 45` = `[DAY, WEEK] + 32 BUILD_* + 11 менеджер-действий`;
старт: 82 000$, 28 человек, 1890 год; карта 280×280.

---

## 2. Быстрый старт

```bat
:: 1. Зависимости
pip install torch pybind11 stable_baselines3 PySide6 numpy

:: 2. Пересборка C++ (ОБЯЗАТЕЛЬНА после правок src/ или include/)
build_pyext.bat        &:: python/colony_cpp.pyd
build_gui.bat          &:: sakhalin_colony_gui.exe (нужен raylib в raylib\ или COLONY_RAYLIB_DIR)

:: 3. Обучение через UI
python run_train_ui.py

:: 3'. Или обучение через CLI
python train.py --steps 4000000 --envs 64 --difficulty light --eval-freq 250000

:: 4. Наблюдение за моделью
python watch_champion.py --model-dir %USERPROFILE%\colony_runs\models\run_001 --visual
```

Модели/логи пишутся в `~/colony_runs/{models,logs}/<run_name>/`:
`best_model.pt` (+ `.norm.json` + `.meta.json`) — только когда eval прошёл
пороги; `checkpoint_*.pt`, `final_model.pt`, `normalization.json`.

---

## 3. Обучение

### Поток данных
`ColonyVecEnvCpp` (C++, пул потоков, нормализация obs/reward внутри C++)
→ `EnvManager` (маски действий, миникарта) → `PPO.collect_step` (fp32,
маскирование недопустимых действий −1e9) → `RolloutBuffer` (GPU) → GAE
(бутстрап по `terminated`, не по `done`) → `PPO.update` (AMP bf16 на CUDA,
KL early-stop, clip 0.2) → eval → `best_model.pt`.

### Профиль наград (v2 — дефолт везде: dataclass = env.h = reward.json)
| Группа | Ключи |
|---|---|
| Стройка | build_bonus 2.0, chain_bonus 1.0, chain_daily 0.5, novelty 5.0, diversity_bonus 3.0, proximity_bonus 0.5, build_cost_penalty 1e-4 |
| Экономика | daily_income 1.0, sale_bonus 0.5, tax_daily_bonus 0.3 |
| Выживание | survival_bonus 0, survival_coeff 0 (выкл.), game_over_penalty 10, death_penalty 20, tax_fail_penalty 5, base_lost_penalty 30, born_bonus 1 |
| Дисциплина | error_penalty −1, demolish −3, manual_tax −0.5, idle −2.0 @ 7 дней |
| Milestones | база×5: 30, люди×50: 2, дни×100: 2, год: 5 |
| Прочее | clip ±50, needs-бонусы (жильё 3/еда 2/вода 2), debt_coeff 0.02, home_overflow 2 |

Все 38 ключей проходят сквозную цепочку UI → worker → to_dict → pybind → C++
(9 «бывших хардкодов» добавлены в 09.2026 — нужна пересборка pyd).
Сырая награда нормализуется running-mean-std в C++, поэтому **сравнивать
прогоны по сырому return нельзя** — только по eval score и дням.

### Eval-протокол (v2)
`score = days·0.10 + bases·1.0 + people·0.10 + 1e-4·max(0,return)`;
пороги сохранения best_model: `bases ≥ 5 И days ≥ 730` (пережил 2 годовых
налога); агрегат — медиана по эпизодам. Настройки: `eval_freq`,
`eval_episodes`, `eval_min_days`, `eval_min_bases`, `eval_seeds` (JSON),
`early_stopping_patience`.

### Ключевые гиперпараметры (дефолты `rl/config.py`)
gamma 0.997, gae_lambda 0.98, ent_coef 0.05, target_kl 0.02, clip 0.2,
vf_coef 0.5, lr 3e-4 (warmup+cosine→10%), n_steps 4096, batch 8192,
epochs 10, net [256,256], obs_mode flat|minimap|hybrid.

### Difficulty
`light`: ×2 стартовые деньги, нет главного налога, игра до 1950. Рекомендуется
curriculum: light (навык до 2-го налога) → дообучение на normal.
В UI: ключ `"difficulty": "light"` в загружаемом JSON (сохраняется и доходит
до worker); в CLI: `--difficulty light`.

### Экономика выживания (почему раньше все умирали на 365 дней)
Годовой налог = `420·занятых_баз + 4686 + 1%·покупки + 2%·продажи`
(≈9k при 10 базах, ≈28k при 55). Нет кассы 1 марта → `check_advance`
замораживает дату, −5/шаг, через 60 дней game over. Главный налог
1.11.1900 = 500 000 (светит только difficulty=normal). Иммиграция +28±5
человек — только 20 июня со 2-го года (~день 476): до неё популяционные
рычаги мертвы. Вывод: агент обязан построить 2–3 цепочки
(Refinery→Ironmine ≈ +450$/день каждая) и держать кассу к 1 марта.

### AMP / torch.compile
- AMP (bfloat16) работает в `PPO.update` и только на CUDA; rollout — fp32
  (намеренно: касты дороже выгоды, bf16-шум вредит GAE).
- torch.compile: mode=default, автоматический откат на eager (в т.ч. на
  Windows, где нет triton). Ожидаемый выигрыш на MLP 256×256 небольшой —
  узкое место — env-шаги, не GPU. Замер: FPS в логе при on/off.

---

## 4. Наблюдение за моделью

UI: выбрать модель → «Наблюдать». Флажок «Визуализация» (по умолчанию вкл.)
открывает окно игры (raylib), которым управляет модель; выкл. — текстовый
лог в панели UI. Stage курикулума: из `best_model.meta.json` либо override.

Механика: `watch_champion.py` находит `sakhalin_colony_gui.exe` (корень,
Release/, build/…, либо `COLONY_GUI_EXE`), запускает его с `--headless-ai
--actions-file --state-file`, читает из state.json obs/маску/миникарту,
нормализует по `*.norm.json` модели, возвращает действие. Карта наблюдения
должна совпадать с тренировочной (дефолт 280, UI передаёт свой параметр).
Отладка: `ai_debug_gui.log` (пишется GUI рядом с exe).

Текстовый режим и полный лог шагов: `observe.py --model ... --all`.

---

## 5. UI обучения (train_ui)

- Все параметры и награды — из `rl/config.py` (единые дефолты); правки
  сохраняются между запусками (`~/colony_runs/sakhalin_colony_ui/config.json`).
- Ключи загруженного JSON без виджетов (difficulty, eval_seeds, unlock_ids,
  loop_detection_*) сохраняются и передаются в worker как есть.
- Команды во время обучения: pause/resume, boost entropy, reset curriculum
  (через command-file, обрабатывает worker).
- Worker — отдельный процесс (`train_ui/worker.py`), протокол JSONL;
  лог/прогресс/метрики в панелях dashboard, KL, return-статистика, топ
  действий, курикулум.

---

## 6. Тесты и проверка

```bat
python tests\test_smoke_fixes.py   :: 27 проверок: конфиги, PPO (CPU), UI-спеки
python rl\loop_detector_test.py
```

Ручная проверка C++-ядра без Python: `sakhalin_colony.exe` (консоль).

---

## 7. Известные ограничения

- Committed бинарники (`python/colony_cpp.pyd`, `*.exe`) устаревают при правках
  C++ — пересобирать (§2 п.2). До пересборки 9 новых reward-полей остаются на
  дефолтах C++ (они равны v2) с WARNING в логе.
- torch.compile на Windows не поддерживается (нет triton) — автоматически
  отключается с сообщением.
- В репозитории лежат тяжёлые артефакты сборки (CMakeFiles/, build/, exe, pyd) —
  .gitignore их уже запрещает; можно вычистить `git rm -r --cached` при желании.
- История болезни и полный аудит: `REPORT_2026_09.md`, `review/`,
  `PART0*.md`, `TRAINING_REPORT.md` (более ранние слои анализа — частично устарели).
