# Дизайн: UI обучения моделей (Sakhalin Colony)

Дата: 2026-08-28
Статус: утверждён пользователем (PySide6, subprocess, полная статистика, все параметры PPO)

## 1. Цель

Десктопное приложение на PySide6 для управления обучением PPO-моделей Сахалинской
колонии: запуск обучения с настраиваемыми параметрами, мониторинг в реальном времени
(прогресс, логи), управление сохранёнными моделями (список, удаление), просмотр
статистики модели (метрики обучения + игровая статистика из eval-прогона).

## 2. Архитектура

Три слоя:

- **UI** (PySide6) — окно, блоки параметров, список моделей, статистика, консоль, прогресс.
- **Бизнес-логика** (`train_ui/`) — управление процессом обучения, парсинг протокола,
  реестр моделей, eval-прогон.
- **Хранилище** — файловая система: модели в `~/colony_runs/models/<run_name>/`,
  конфиг UI в `%LOCALAPPDATA%/sakhalin_colony_ui/config.json`.

Обучение выполняется в **отдельном процессе** (subprocess). Протокол — JSONL по stdout:
каждая строка — один JSON-объект `{"type": ..., ...}`. Это даёт изоляцию (crash
обучения не роняет UI), безопасную остановку (kill процесса) и push-обновления.

```
UI (PySide6, главный поток)
  │  QProcess (start, readChannel, kill)
  ▼
train_ui/worker.py (отдельный процесс)
  │  rl/async_trainer.AsyncTrainer.train()
  ▼
rl/env_manager.EnvManager → python/cpp_vecenv.CppVecEnv → colony_cpp (C++)
```

### Протокол JSONL (stdout рабочего процесса)

Все строки — один JSON-объект на строку, UTF-8. Типы:

- `{"type":"ready"}` — процесс готов (после импортов и создания окружения).
- `{"type":"log","level":"info|warn|error","message":"..."}` — сообщение в консоль.
- `{"type":"progress","done":123456,"total":1000000,"fps":38000,"best_reward":123.4,
     "episodes":5,"policy_loss":0.12,"value_loss":0.34,"entropy":0.5,"kl":0.001}`
- `{"type":"saved","path":".../checkpoint_500000_steps.pt"}`
- `{"type":"done","total":1000000,"time_s":26.0,"best_reward":...,"episodes":...}`
- `{"type":"error","message":"..."}` — фатальная ошибка (процесс затем завершается).

Команды UI → процесс: **stdin**, одна JSON-строка:
- `{"cmd":"stop"}` — корректная остановка (trainer._request_stop(), завершить цикл,
  сохранить final_model, выйти с кодом 0).

Ограничение: torch/C++ печатают в stdout — рабочий процесс обязан переопределить
`print` и редиректить обычный вывод в `{"type":"log",...}` строки (обёртка над
sys.stdout в worker.py).

## 3. Файлы

| Файл | Ответственность |
|---|---|
| `train_ui/__init__.py` | Пустой маркер пакета |
| `train_ui/protocol.py` | Данные протокола: `MsgType`, dataclass-сообщения, `encode(msg)->str`, `decode(line)->Msg` |
| `train_ui/models.py` | `ModelInfo` (имя, путь, дата, steps, best_reward, people, bases, days), `ModelRegistry` (scan, delete, load meta из meta.json) |
| `train_ui/evaluator.py` | `run_eval(model_path, episodes, max_days)` — eval-прогон CppColonyEnv с обученной политикой, возвращает средние days/people/bases |
| `train_ui/worker.py` | Точка входа рабочего процесса: аргументы, создание Config/EnvManager/AsyncTrainer, прогресс-колбэк → JSONL, stop по stdin, переопределение print |
| `train_ui/main_window.py` | PySide6 окно: блоки параметров, список моделей, статистика, консоль, прогресс |
| `train_ui/parameter_widget.py` | Компонент «подпись + поле + ×2/÷2 + tooltip» |
| `train_ui/app.py` | `main()`: QApplication, загрузка конфига, окно, сохранение состояния при закрытии |
| `run_train_ui.py` | Тонкая обёртка `from train_ui.app import main; main()` |
| `tests/test_protocol.py` | Тесты encode/decode |
| `tests/test_models.py` | Тесты реестра (scan/delete/meta) |
| `tests/test_parameter_widget.py` | Тесты ×2/÷2 логики (без GUI, чистая функция) |
| `tests/test_evaluator.py` | Тест eval-прогона (короткий, 1 эпизод, мало дней) |

Не меняется: `rl/*`, `python/*`, `train.py`, `src/*`.

## 4. Компоненты UI (макет по ТЗ §9)

Окно 1280×800, QSplitter'ы:

- **Левая панель (~300px)**: QTreeWidget/QTableWidget списка моделей (чекбокс, имя,
  дата, steps, best_reward), кнопки «Удалить выбранные», «Обновить», «Оценить»
  (eval-прогон выбранной).
- **Центр**: блок параметров (сетка ParameterWidget: подпись, QDoubleSpinBox, кнопки
  ×2/÷2), кнопки «Запустить обучение», «Остановить», «Сбросить значения по умолчанию»,
  поле «Имя запуска».
- **Правая панель (~280px)**: статистика выбранной модели — метрики обучения
  (steps, best_reward, episodes, финальные losses из meta.json) + игровая статистика
  (средние дни, люди, постройки — из eval, кнопка «Пересчитать»).
- **Низ (~30% высоты)**: над консолью QProgressBar + текстовый статус
  («Эпоха/Rollout N, FPS, осталось ~X мин»); консоль — QPlainTextEdit (read-only,
  цветные HTML-строки: info — обычный, warn — жёлтый, error — красный), кнопки
  «Очистить», «Сохранить в файл».

### Параметры (все из Config, по ТЗ §2)

| Параметр | Поле | Тип | Диапазон | Tooltip (кратко) |
|---|---|---|---|---|
| Имя запуска | name | text | — | Подкаталог в models/, где сохраняются чекпоинты |
| Шагов обучения | total_timesteps | int | 1e4..1e9 | Сколько timesteps PPO-обучения выполнить |
| Параллельных окружений | n_envs | int | 1..256 | Сколько сред бежит одновременно (больше — быстрее rollout) |
| Шагов на rollout | n_steps | int | 128..65536 | Длина собираемого батча переходов на окружение |
| Размер батча | batch_size | int | 256..262144 | Размер мини-батча для PPO-обновления |
| Эпох на rollout | n_epochs | int | 1..50 | Сколько раз PPO перебирает собранный батч |
| Коэф. обучения | learning_rate | float | 1e-6..1e-1 | Шаг оптимизатора AdamW |
| Дисконт | gamma | float | 0.9..0.9999 | Вес будущих наград |
| GAE lambda | gae_lambda | float | 0.8..1.0 | БIAS/VARIANCE в оценке GAE |
| Clip range | clip_range | float | 0.05..0.5 | Ограничение изменения политики за шаг PPO |
| Ent coef | ent_coef | float | 0..0.1 | Вес энтропии (исследование) |
| VF coef | vf_coef | float | 0.1..2.0 | Вес функции ценности |
| Max grad norm | max_grad_norm | float | 0.1..10.0 | Обрезка градиента |
| Слой сети | net_arch | int | 64..1024 | Размер скрытых слоёв (2 слоя) |
| Seed | seed | int | 0..2^31 | Случайное зерно |
| Размер карты | map_size | int | 100..500 | Размер мира в ячейках |
| AMP | use_amp | check | — | Полупrecise-арифметика (bfloat16) на GPU |
| torch.compile | torch_compile | check | — | Компиляция модели (первый запуск дольше) |

×2/÷2: целые — round, минимум 1; дробные — round(x, 8), минимум 1e-9.

### Поведение

- **Запуск**: сборка Config из полей → запуск `python train_ui/worker.py --config <tmp.json>`
  через QProcess. Кнопка «Запустить» блокируется на время обучения (защита от
  повторного запуска по ТЗ §8), «Остановить» — активна.
- **Остановка**: `{"cmd":"stop"}` в stdin; таймаут 15 с → `QProcess.kill()`.
- **Прогресс**: `progress`-сообщения → QProgressBar (done/total) + текст
  «{done:,}/{total:,} · FPS {fps:,.0f} · best {best_reward:.1f} · осталось ~{eta}».
  ETA = (total-done)/fps. При `done` — прогресс 100%, зелёный.
- **Логи**: `log`/`error`-сообщения → консоль с автопрокруткой (если пользователь
  не скроллит вверх).
- **Модели**: реестр сканирует `~/colony_runs/models/*/` — каталоги с `final_model.pt`
  или `checkpoint_*.pt`. Удаление — подтверждение QMessageBox (необратимо),
  физическое удаление каталога.
- **Статистика**: клик по строке → meta.json (steps, best_reward, episodes, время,
  параметры запуска) + кнопка «Оценить» запускает eval в отдельном процессе
  (`worker.py --eval <model> --episodes N --max-days D`) и показывает средние
  days/people/bases.
- **Сохранение состояния** (ТЗ §7): config.json — параметры, имя запуска, геометрия
  окна, выбранная модель.
- **Тестируемость** (ТЗ §7): `objectName` у всех элементов; чистая логика
  (protocol, models, ×2/÷2, evaluator) без Qt — тестируется pytest'ом.

## 5. Обработка ошибок

- Crash процесса (код ≠ 0, не остановка) → в консоль `error`, кнопки возвращаются.
- Таймаут остановки → kill + сообщение.
- Ошибка eval → сообщение в консоль, статистика не меняется.
- Пустой список моделей → подсказка «Запустите обучение, чтобы получить модели».

## 6. Тесты

- `test_protocol.py` — roundtrip encode/decode для всех типов, битая строка → ошибка.
- `test_models.py` — создание фейковых каталогов моделей, scan находит, meta.json
  читается, delete удаляет; папки без final_model.pt игнорируются.
- `test_parameter_widget.py` — `scale_value(v, factor)` для int/float, границы.
- `test_evaluator.py` — `run_eval` на заведомо-случайной модели, 1 эпизод,
  max_days=50: возвращает dict с days/people/bases ≥ 0. (Нужен собранный colony_cpp;
  пропуск, если pyd не найден.)

## 7. Вне рамок (YAGNI)

- TensorBoard-графики в UI (уже есть в логах).
- Мультизапуск (несколько обучений одновременно) — одно за раз.
- Веб-интерфейс, тёмная тема, локализация кроме русского.
