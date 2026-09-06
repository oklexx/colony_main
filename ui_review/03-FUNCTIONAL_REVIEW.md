# 03 — FUNCTIONAL REVIEW

## Проверка функциональности компонентов UI

---

## 1. KLStatusWidget

**Файл:** `train_ui/kl_status_widget.py`

### Что делает
Отображает цветовой индикатор KL divergence:
- `KL < 0.01` → LOW (orange)
- `0.01 ≤ KL < 0.025` → OK (lime)
- `0.025 ≤ KL < 0.08` → OPTIMAL (turquoise)
- `KL ≥ 0.08` → HIGH (red)

Прогресс-бар нормализует KL в диапазон 0.01-0.08.

### Статус: **Частично работает**

### Проблемы
1. **Некорректная логика статусов** (строки 66-93): Диапазоны пересекаются:
   - `kl < 0.01` → LOW
   - `kl < 0.025` → OK
   - `kl < 0.08` → OPTIMAL
   - иначе → HIGH
   
   Но `kl_optimal = 0.25` (строка 20) задаётся, но **не используется** в `_update_status()`! Вместо этого используется хардкод `0.025` и `0.08`.

2. **Прогресс-бар заливает фон цветом статуса** (строки 98-141): При высоком KL фон красный, chunk тоже красный — визуально неразличимо.

3. **ent_coef не обновляется** извне — виджет показывает статичное значение 0.005.

### Интеграция с MainWindow
`main_window.py:1187-1191`:
```python
if m.kl is not None:
    self.kl_status_widget.update(
        kl=float(m.kl),
        ent_coef=m.ent_coef if hasattr(m, 'ent_coef') else 0.005
    )
```
`ProgressMsg` не имеет поля `ent_coef` → всегда показывает 0.005.

---

## 2. CurriculumProgressWidget

**Файл:** `train_ui/curriculum_progress_widget.py`

### Что делает
Отображает текущий этап curriculum, прогресс-бар, доступные действия, upcoming stages.

### Статус: **Не работает корректно**

### Проблемы
1. **Прогресс-бар красный при высоком прогрессе** (строки 113-125):
   ```python
   if progress_percent < 0.3:
       color = orange  # OK
   elif progress_percent < 0.7:
       color = lime    # OK
   else:
       color = red     # НЕТ! Должен быть blue/primary
   ```

2. **`_update_upcoming_stages` не вызывается** из `update()` при отсутствии `upcoming_stages` в данных.

3. **Данные не приходят из worker** — `ProgressMsg` не содержит `stage`, `progress_percent`, `available_actions`. В `main_window.py:1194-1202` эти данные берутся из `hasattr(m, 'stage')`, что всегда True на dataclass.

### Интеграция
Данные для этого виджета **никогда не приходят** из worker-процесса в корректном формате. `AsyncTrainer.progress_callback` отправляет `TrainMetrics` (без curriculum полей), а расширенные данные (`ui_metrics` dict) собираются, но **не передаются** в callback.

---

## 3. ActionLoopWidget

**Файл:** `train_ui/action_loop_widget.py`

### Что делает
Отображает предупреждения при обнаружении циклов действий (PRESERVE loop и т.д.).

### Статус: **Работает визуально, но данные некорректны**

### Проблемы
1. **Loop rate всегда вычисляется с total_envs=8** (строка 140):
   ```python
   pct = (self._envs_with_loops / max(total_envs, 1)) * 100
   ```
   `total_envs` по умолчанию 8, не обновляется извне.

2. **`consecutive_count` всегда 10** (`main_window.py:1209`):
   ```python
   consecutive_count=10,  # Хардкод!
   ```

3. **LoopDetector не отслеживает реальные последовательности** корректно:
   - `async_trainer.py:173-187`: Пытается получить действия из buffer, но обращается к `self.em.buffer.actions` как к 2D тензору, хотя buffer может быть `_TensorRolloutBuffer` с другим форматом.
   - Action names захардкожены в `async_trainer.py:46-48` и не совпадают с реальными (DAY, WEEK, BUILD_*).

---

## 4. ReturnStatisticsWidget

**Файл:** `train_ui/return_statistics_widget.py`

### Что делает
Показывает avg/median/max/min возвратов по эпизодам.

### Статус: **Частично работает**

### Проблемы
1. **Нет гистограммы** — в спецификации (04-UI_COMPONENTS.md) требовалась гистограмма распределения через pyqtgraph. Реализованы только текстовые лейблы.

2. **Данные не приходят** — `ProgressMsg` не содержит `returns`. `main_window.py:1220-1221`:
   ```python
   if hasattr(m, 'returns') and m.returns:
       self.return_statistics_widget.update_statistics(returns=list(m.returns))
   ```
   `ProgressMsg` не имеет атрибута `returns` → никогда не обновляется.

3. **Style bug** (строка 138-142): `style_max` содержит `background-color: #4caf50;` (с точкой с запятой), но передаётся в f-string:
   ```python
   self._max_label.setStyleSheet(f"""
       font-size: 11pt;
       padding: 5px;
       background-color: {style_max};
   """)
   ```
   Это создаёт `background-color: background-color: #4caf50;;` — двойной property.

---

## 5. QuickActionsWidget

**Файл:** `train_ui/quick_actions_widget.py`

### Что делает
Панель кнопок: Pause/Resume, Boost Entropy ×2, Reset Curriculum.

### Статус: **НЕ РАБОТАЕТ — все кнопки крашат**

### Критические баги
1. **`self.log()` не определён** (строки 180, 200, 222, 237, 281, 289):
   ```python
   self.log("[QuickActions] Pause training requested")  # AttributeError
   ```
   Виджет не наследует и не имеет атрибута `log`.

2. **Бесконечная рекурсия** (строки 270-271):
   ```python
   elif cmd == "reset_curriculum":
       stage = payload.get("stage", 0)
       self._send_command("reset_curriculum", {"stage": stage})  # Вызывает сам себя!
   ```

3. **`_on_pause_clicked`** использует `self.findChild(QLabel, "status_label")` (строка 186) — но QLabel не имеет `objectName = "status_label"`. `findChild` вернёт None.

4. **Сигналы не подключены** к MainWindow — `cmd_boost_entropy`, `cmd_pause_training`, `cmd_resume_training`, `cmd_stop_training` определены, но нигде не `connect()`.

---

## 6. TrainingDashboardWidget

**Файл:** `train_ui/dashboard_widget.py`

### Статус: **НЕ ПОДКЛЮЧЁН**

`TrainingDashboardWidget` определён, но:
- Не импортируется в `main_window.py`
- Не создаётся в `MainWindow._build_ui()`
- Ссылка `self.dashboard.update_data()` в `main_window.py:1226` вызовет `AttributeError`

### Проблемы в самом коде
1. `_setup_plots()` (строки 101-121) создаёт графики на `self._plot_widget`, но `_add_plot_to_splitter()` создаёт **отдельный** `self._plot_widget` для каждого графика, перезаписывая предыдущий. В итоге только последний график (action_plot) доступен.

2. `_scroll_view()` (строки 163-171) пытается вызвать `setXRange` на кривых, а не на PlotWidget.

---

## 7. Headless-режим

**Реализован в:** `worker.py`

### Статус: **Работает**

```bash
python train_ui/worker.py --config config.json --name run_001 --output msg.jsonl
```

- Training запускается без UI
- Логи пишутся в JSONL
- Команды читаются с stdin

### Проблема
`command_queue` не подключён к stdin-потоку (`_watch_stdin`). Команды с stdin попадают в `command_queue`, но `run_train()` создаёт **свой** `_queue.Queue` и передаёт его в `AsyncTrainer`, который его **не использует** ( нет code для обработки команд в training loop).

---

## 8. Обучение через UI

### Статус: **Запускается, но виджеты не обновляются**

`_start_training()` (`main_window.py:1057-1129`) корректно:
- Собирает конфиг
- Запускает worker-процесс
- Запускает QTimer для polling
- Читает JSONL-файл

`_update_progress()` (`main_window.py:1174-1231`) пытается обновить виджеты, но:
- `self.dashboard` → AttributeError (КРИТИЧЕСКИЙ)
- `m.stage`, `m.returns` → AttributeError (Высокий)
- `m.ent_coef` → hasattr проверка, но всегда 0.005 (Средний)

---

## Summary

| Компонент | Статус | Блокеры |
|-----------|--------|---------|
| KLStatusWidget | Частично | ent_coef всегда 0.005 |
| CurriculumProgressWidget | Не работает | Данные не приходят из worker |
| ActionLoopWidget | Частично | Хардкод consecutive=10, неправильные action names |
| ReturnStatisticsWidget | Не работает | Данные не приходят, нет гистограммы |
| QuickActionsWidget | **КРАШ** | self.log() не определён, рекурсия |
| TrainingDashboardWidget | **НЕ ПОДКЛЮЧЁН** | Не импортируется, self.dashboard не существует |
| Headless-режим | Работает | command_queue не подключён |
| Запуск обучения | Работает | Виджеты не обновляются |
