# 07 — FINDINGS AND RECOMMENDATIONS

## Полный список выявленных проблем

---

## Critical (Блокируют использование)

| # | File:Line | Issue | Рекомендация |
|---|-----------|-------|--------------|
| C1 | `main_window.py:1226` | `self.dashboard.update_data(...)` — атрибут `dashboard` не определён. `TrainingDashboardWidget` не импортируется и не создаётся. | Добавить import и инициализацию в `_build_ui()`, либо удалить ссылку |
| C2 | `quick_actions_widget.py:180,200,222,237,281,289` | `self.log(...)` — QuickActionsWidget не имеет атрибута `log`. Все обработчики кнопок крашат с AttributeError. | Добавить `log` callback при инициализации, либо использовать `print()` |
| C3 | `quick_actions_widget.py:270-271` | `_send_command("reset_curriculum", ...)` вызывает сам себя → RecursionError. | Убрать рекурсивный вызов, заменить на прямую emit сигнала |
| C4 | `main_window.py:1194-1221` | `_update_progress()` обращается к `m.stage`, `m.progress_percent`, `m.available_actions`, `m.next_stage_at_step`, `m.upcoming_stages`, `m.returns` — поля ProgressMsg не существуют. | Добавить поля в ProgressMsg, либо передавать через отдельный dict |
| C5 | `worker.py:373-375` | `command_queue` не подключён к stdin-потоку. Аргумент `command_queue` не в argparse. Команды с UI не доходят до training loop. | Добавить `--command-queue` в argparse или использовать file-based команды |
| C6 | `quick_actions_widget.py:186,208` | `self.findChild(QLabel, "status_label")` — QLabel не имеет `objectName`. Всегда возвращает None. | Либо задать objectName при создании, либо хранить ссылку на label |
| C7 | `main_window.py:1118` | `_msg_timer.start(250)` запускается до проверки `if self._train_pid`. Если proc не запустился, таймер будет poll мёртвый файл. | Переместить `start()` после проверки pid |

---

## High (Серьёзные проблемы)

| # | File:Line | Issue | Рекомендация |
|---|-----------|-------|--------------|
| H1 | `async_trainer.py:46-48` | Action names захардкожены (INITIALIZE, MOVE_RIGHT...) и не совпадают с реальными (DAY, WEEK, BUILD_*). Loop detector получает неправильные имена. | Использовать `self.em.env._action_names` или передавать из C++ |
| H2 | `async_trainer.py:173-187` | Попытка чтения `self.em.buffer.actions` — формат зависит от типа buffer (RolloutBuffer vs _TensorRolloutBuffer). Может крашить. | Использовать публичный API buffer или отдельный action tracker |
| H3 | `env_manager.py:263-286` | `get_allowed_buildings_for_stage()` возвращает INITIALIZE, MOVE_RIGHT... — имена не существуют в C++ env. Curriculum progress отображает мусор. | Привязать к реальному action space из C++ |
| H4 | `async_trainer.py:264-293` | `progress_callback` отправляет `TrainMetrics` (base), расширенные данные (`ui_metrics` dict) собираются, но **не передаются** callback. | Либо добавить поля в TrainMetrics, либо callback должен принимать dict |
| H5 | `loop_detector.py:154-157` | `envs_with_loops` считает `action_count >= threshold` — это **общее** количество действий, не consecutive. Любой активный env помечается как "in loop". | Использовать отдельный `consecutive_count` per env |

---

## Medium (Проблемы качества)

| # | File:Line | Issue | Рекомендация |
|---|-----------|-------|--------------|
| M1 | `curriculum_progress_widget.py:119-125` | Прогресс-бар красный при progress > 70%. Должен быть primary/blue. | Заменить `red` на `#2196F3` |
| M2 | `return_statistics_widget.py:138-142` | `style_max` = `"background-color: #4caf50;"` → дублирование property в f-string. | Убрать `background-color:` из style_max |
| M3 | `protocol.py:135` | `CommandMsg.type = MsgType.LOG` — вводит в заблуждение. | Добавить `MsgType.COMMAND = "command"` |
| M4 | `loop_detector.py:84` | `list.pop(0)` O(n) — не масштабируется. | Использовать `collections.deque(maxlen=100)` |
| M5 | `action_loop_widget.py:140` | `total_envs` хардкожен = 8. Не обновляется из n_envs. | Передавать n_envs через set_loop_status |
| M6 | `test_protocol.py:99` | `test_encode_stop` устарел — ожидает другой формат. | Обновить assertion |
| M7 | `main_window.py:992-1008` | `_poll_pid_alive()` использует `tasklist` (Windows-only). | Добавить fallback для Linux/macOS |
| M8 | `ppo.py:180` | `logits, values = self.model(flat, obs)` — для hybrid `obs` это minimap, но переменная названа `obs`. | Переименовать в `minimap` для ясности |

---

## Low (Замечания)

| # | File:Line | Issue | Рекомендация |
|---|-----------|-------|--------------|
| L1 | `dashboard_widget.py:101-121` | `_setup_plots()` перезаписывает `self._plot_widget` 3 раза. | Использовать отдельные переменные для каждого графика |
| L2 | `dashboard_widget.py:163-171` | `_scroll_view()` вызывает `setXRange` на кривых, а не на PlotWidget. | Исправить вызов |
| L3 | `async_trainer.py:106` | `_ep_returns` растёт бесконечно. | Добавить `maxlen` или периодическую очистку |
| L4 | `kl_status_widget.py:20` | `kl_optimal = 0.25` задан, но не используется. | Исправить или удалить |
| L5 | `main_window.py:1128` | Bare `except:` — глотает все исключения. | Заменить на `except Exception:` |
| L6 | `evaluator.py:57` | `torch.load(..., weights_only=False)` — потенциальная уязвимость. | Использовать `weights_only=True` где возможно |

---

## Чек-лист проверки

- [ ] C1: TrainingDashboardWidget подключён
- [ ] C2: QuickActionsWidget.log() работает
- [ ] C3: Нет бесконечной рекурсии
- [ ] C4: ProgressMsg содержит все нужные поля
- [ ] C5: Command queue подключён к stdin
- [ ] C6: status_label найден через findChild
- [ ] C7: Таймер запускается после проверки pid
- [ ] H1: Action names совпадают с C++
- [ ] H2: Buffer actions читаются безопасно
- [ ] H3: get_allowed_buildings возвращает реальные имена
- [ ] H4: Расширенные данные передаются в callback
- [ ] H5: Loop detector считает consecutive, не total
- [ ] M1: Прогресс-бар синий при высоком прогрессе
- [ ] M2: Нет дублирования style property
- [ ] M3: CommandMsg имеет свой MsgType
- [ ] M4: LoopDetector использует deque
- [ ] M5: total_envs не хардкожен
- [ ] M6: test_encode_stop исправлен
- [ ] M7: _poll_pid_alive кроссплатформенный
- [ ] M8: hybrid переменные названы понятно

---

## Приоритеты исправлений

### Неделя 1 (Critical + High)
1. Исправить QuickActionsWidget (C2, C3, C6) — 2 часа
2. Подключить TrainingDashboardWidget (C1) — 1 час
3. Добавить поля в ProgressMsg (C4) — 2 часа
4. Подключить command queue (C5) — 3 часа
5. Исправить action names (H1) — 1 час
6. Исправить loop detector counting (H5) — 1 час

### Неделя 2 (Medium + Low)
7. Исправить curriculum buildings (H3) — 2 часа
8. Передавать расширенные данные в callback (H4) — 2 часа
9. Исправить buffer actions reading (H2) — 2 часа
10. Исправить M1-M8 — 3 часа
11. Исправить L1-L6 — 2 часа
12. Добавить тесты — 4 часа

**Общая оценка: ~25 часов работы**
