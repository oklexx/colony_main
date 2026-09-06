# 02 — CODE REVIEW

## Детальный анализ кода

---

## 1. Стиль и читаемость

### Позитив
- Все файлы используют `from ____ import annotations` для PEP 604 union syntax
- Consistent naming: `snake_case` для функций/переменных, `PascalCase` для классов
- Docstrings присутствуют на ключевых классах и методах
- Заголовки файлов с описанием назначения

### Проблемы

#### 1.1. `main_window.py` — 1505 строк (CRITICAL)
Файл смешивает:
- UI layout (~500 строк)
- Training control logic (~300 строк)
- Watch/Eval subprocess management (~200 строк)
- Curriculum management (~150 строк)
- Config serialization (~100 строк)
- Stats display (~100 строк)

**Рекомендация:** Разделить на `training_controller.py`, `curriculum_manager.py`, `eval_controller.py`.

#### 1.2. Магические числа
- `main_window.py:185`: `root.addLayout(right, 5)` — stretch factor без пояснения
- `main_window.py:199`: `metrics_main_panel.setFixedWidth(290)` — захардкоженная ширина
- `action_loop_widget.py:140`: `total_envs` всегда 8 в вычислении loop_rate

#### 1.3. Дублирование стилей
`KLStatusWidget._update_status()` и `_update_progress_bar()` содержат идентичные `setStyleSheet()` вызовы с разницей только в цвете. Должен быть общий helper-метод.

---

## 2. Обработка ошибок

### 2.1. `_update_progress` в `main_window.py:1174-1231`

```python
# Строка 1188: AttributeError если m не имеет ent_coef
ent_coef=m.ent_coef if hasattr(m, 'ent_coef') else 0.005

# Строка 1194: AttributeError если m не имеет stage
if hasattr(m, 'stage') and m.stage is not None:

# Строка 1205: AttributeError если m не имеет loop_detected
if hasattr(m, 'loop_detected') and m.loop_detected:

# Строка 1220: AttributeError если m не имеет returns
if hasattr(m, 'returns') and m.returns:
```

Проблема: `ProgressMsg` не содержит `stage`, `returns`, `ent_coef`. `hasattr` проверяет наличие атрибута, но на dataclass это всегда True (атрибут есть, даже если не задан). Нужно проверять конкретные поля ProgressMsg.

### 2.2. `QuickActionsWidget._on_pause_clicked()` (строка 180)
```python
self.log("[QuickActions] Pause training requested")  # AttributeError
```
`QuickActionsWidget` не имеет атрибута `log`. То же в `_on_resume_clicked()`, `_on_boost_clicked()`, `_on_reset_clicked()`.

### 2.3. `worker.py:373-375` — command_queue создаётся ДВАЖДЫ
```python
command_queue = None
if hasattr(args, 'command_queue') and args.command_queue:  # Всегда False!
    command_queue = _queue.Queue(maxsize=100)
# ...
rc = run_train(..., command_queue=_queue.Queue(maxsize=command_queue_size) if command_queue_size > 0 else None)
```
Аргумент `command_queue` не добавлен в argparse, поэтому `hasattr(args, 'command_queue')` всегда False. Второй `_queue.Queue` создаётся в вызове `run_train`, но не подключён к stdin-потоку.

---

## 3. Безопасность

### 3.1. Path traversal
`curriculum_table` данные загружаются из JSON без валидации (`_cur_load` в `main_window.py:1394-1409`). Валидация есть (`_validate_curriculum`), но только на монотонность порогов — нет проверки типов значений.

### 3.2. `os.environ.get("COLONY_DEBUG")` (cpp_env.py:135)
Выводит содержимое `RewardConfig` в stdout. В production может раскрыть внутренние параметры.

### 3.3. `torch.load` с `weights_only=False`
Используется в `evaluator.py:57`, `worker.py:267`, `ppo.py:232`. Потенциальная уязвимость при загрузке недоверенных чекпоинтов.

---

## 4. Производительность кода

### 4.1. `LoopDetector.update_batch()` — O(n*m)
```python
for data in action_data:        # O(n) environments
    state.action_sequence.pop(0)  # O(m) — list shift!
```
`list.pop(0)` — O(n) операция. Для 100 элементов и 8 envs — приемлемо, но при масштабировании на 2048 envs будет проблема. Использовать `collections.deque`.

### 4.2. `_calculate_action_distribution()` в `async_trainer.py:131-160`
Метод итерирует `loop_detector._state` и `state.action_sequence` — приватные атрибуты другого класса. Нарушение инкапсуляции.

### 4.3. `_poll_pid_alive()` (main_window.py:992-1008)
Вызывает `tasklist` через subprocess каждые 250ms. На Windows это ~2-5ms на вызов. При 4 таймерах (training, eval, watch, messages) это ~10-20ms/сек — приемлемо, но можно оптимизировать через `psutil`.

---

## 5. Сериализация

### 5.1. `ProgressMsg.to_dict()` (protocol.py:65-83)
```python
"curriculum_next_at_step": int(self.curriculum_next_at_step) if self.curriculum_next_at_step else None,
```
Если `curriculum_next_at_step = 0`, то `if 0` → False → None. Нулевой шаг — валидное значение.

### 5.2. `CommandMsg.type` (protocol.py:135)
```python
type: MsgType = field(default=MsgType.LOG, init=False)
```
CommandMsg переиспользует `MsgType.LOG` вместо отдельного типа. При decode: `decode()` обрабатывает `"command"` отдельно (строка 222), но `encode()` создаёт `{"type": "command"}`. Несоответствие типов.

---

## 6. Файлы конфигурации

| Файл | Оценка | Проблемы |
|------|--------|----------|
| `configs/bases.json` | OK | 33 здания, полные описания |
| `configs/events.json` | OK | 46 событий, корректные ссылки |
| `configs/reward.json` | OK | Совпадает с `RewardConfig` defaults |
| `rl/config.py` | OK | Dataclass с validation |

---

## 7. Статический анализ (py_compile)

Все 12 ключевых файлов компилируются без ошибок.

### Результаты тестов
- `test_protocol.py`: 13/14 PASSED, 1 FAILED (`test_encode_stop` — ожидает `{"cmd": "stop"}`, получает `{"type": "command", "cmd": "stop", "payload": {...}}`)
- `test_parameter_widget.py`: 14/14 PASSED

---

## Summary: Критические находки

| # | Severity | File:Line | Issue |
|---|----------|-----------|-------|
| 1 | **Critical** | `main_window.py:1226` | `self.dashboard` не определён — AttributeError |
| 2 | **Critical** | `quick_actions_widget.py:180` | `self.log()` не определён — AttributeError |
| 3 | **Critical** | `quick_actions_widget.py:270` | Бесконечная рекурсия `_send_command` |
| 4 | **High** | `main_window.py:1194-1221` | Обращение к несуществующим полям ProgressMsg |
| 5 | **High** | `worker.py:373` | `command_queue` не подключён к stdin |
| 6 | **Medium** | `async_trainer.py:46-48` | Action names не совпадают с реальными |
| 7 | **Medium** | `loop_detector.py:84` | `list.pop(0)` O(n) вместо deque |
| 8 | **Low** | `protocol.py:135` | CommandMsg.type = MsgType.LOG |
