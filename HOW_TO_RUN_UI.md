# 🚀 Как запустить UI с новыми улучшениями

## ⚡ Быстрый старт

### 1. Запустить UI (если уже интегрировано)
```bash
cd C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main
python run_train_ui.py
```

**Если интерфейс открылся без ошибок** — все виджеты успешно интегрированы! ✓

---

### 2. Если UI НЕ запускается или виджеты отсутствуют

Новые виджеты созданы, но могут еще не подключены к MainWindow. Нужно:

#### Шаг 1: Проверить импорты
```bash
python -c "from train_ui.kl_status_widget import KLStatusWidget; print('OK')"
```

Ожидаемый результат: `OK`

#### Шаг 2: Добавить импорты в main_window.py

Откройте `train_ui/main_window.py` и добавьте в imports (строка ~1-50):

```python
from train_ui.kl_status_widget import KLStatusWidget
from train_ui.curriculum_progress_widget import CurriculumProgressWidget
from train_ui.action_loop_widget import ActionLoopWidget
from train_ui.return_statistics_widget import ReturnStatisticsWidget
from train_ui.quick_actions_widget import QuickActionsWidget
from train_ui.dashboard_widget import TrainingDashboardWidget
```

#### Шаг 3: Создать контейнер для виджетов

В `MainWindow.__init__()`, после создания layout, добавьте:

```python
# Container for new widgets
self._new_widgets_frame = QWidget()
self._new_widgets_layout = QVBoxLayout(self._new_widgets_frame)
self._new_widgets_frame.setStyleSheet("""
    QFrame {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
    }
""")

# Add widgets (order matters - top to bottom)
self.kl_status = KLStatusWidget()
self._new_widgets_layout.addWidget(self.kl_status)

self.curriculum_progress = CurriculumProgressWidget()
self._new_widgets_layout.addWidget(self.curriculum_progress)

self.action_loop = ActionLoopWidget()
self._new_widgets_layout.addWidget(self.action_loop)

self.return_stats = ReturnStatisticsWidget()
self._new_widgets_layout.addWidget(self.return_stats)

self.quick_actions = QuickActionsWidget()
self._new_widgets_layout.addWidget(self.quick_actions)

self.dashboard = TrainingDashboardWidget()
self._new_widgets_layout.addWidget(self.dashboard)

# Add frame to main layout
self.main_layout.insertWidget(0, self._new_widgets_frame)  # Insert at top
```

#### Шаг 4: Подключить обновления к worker signals

В методе `connect_worker()` или аналогичном, добавьте:

```python
# Connect to progress updates
def on_progress_update(data):
    """Update all new widgets with training data."""
    # Update KL Status
    if 'kl' in data and 'entropy' in data:
        self.kl_status.update(
            kl=data['kl'],
            ent_coef=data.get('ent_coef', 0.005)
        )
    
    # Update Curriculum Progress
    if 'stage' in data:
        progress_data = {
            'stage': data['stage'],
            'progress_percent': data.get('progress_percent', 0),
            'available_actions': data.get('available_actions', ''),
            'next_stage_at_step': data.get('next_stage_at_step'),
            'upcoming_stages': data.get('upcoming_stages', [])
        }
        self.curriculum_progress.update(progress_data)
    
    # Update Action Loop Widget
    if 'loop_detected' in data:
        self.action_loop.set_loop_status(
            loop_detected=data['loop_detected'],
            action_name=data.get('loop_action_name'),
            consecutive_count=10,  # or calculate from history
            threshold=3,
            envs_with_loops=data.get('envs_with_loops', 0),
            total_envs=8
        )
    
    # Update Return Statistics (if returns data available)
    if 'returns' in data:
        self.return_stats.update_statistics(returns=data['returns'])
    
    # Update Dashboard plots
    top_actions = data.get('top_actions', {})
    self.dashboard.update_data(
        kl=data.get('kl'),
        entropy=data.get('entropy'),
        top_actions=top_actions
    )

# Connect signal
self.worker.progress_signal.connect(on_progress_update)
```

---

## 🎯 Запуск с конфигурацией

### Вариант 1: Интерактивный (с UI)
```bash
python run_train_ui.py
```

Откроется окно с:
- ✅ KL статусом (цветной индикатор)
- ✅ Прогрессом куррикулума
- ✅ Алертами на циклы действий
- ✅ Статистикой эпизодов
- ✅ Панелью быстрых действий (pause/resume/boost)
- ✅ Интерактивными графиками

### Вариант 2: Через конфиг
```bash
python run_train_ui.py --config path/to/config.json
```

### Вариант 3: Headless (без UI, только worker)
```bash
python train_ui/worker.py --config path/to/config.json --name my_run
```

---

## 🧪 Проверка работы виджетов

После запуска UI проверьте:

1. **KLStatusWidget**: Должен показывать KL value и цвет (зеленый/оранжевый/красный)
2. **CurriculumProgressWidget**: Показывает "Stage: X/3" и прогресс бар
3. **ActionLoopWidget**: Сначала скрыт, появляется при детекции циклов
4. **ReturnStatisticsWidget**: 4 метрики (avg, med, max, min returns)
5. **QuickActionsWidget**: Кнопки Pause/Resume/Boost/Reset активны
6. **TrainingDashboardWidget**: Графики KL и Entropy обновляются

---

## 🐛 Устранение проблем

### Проблема: ModuleNotFoundError
**Решение**: Убедитесь что `requirements.txt` обновлен:
```bash
pip install -r requirements.txt
```

### Проблема: Import error в виджетах
**Решение**: Проверьте зависимости:
```bash
python -c "import pyqtgraph; import PySide6; print('OK')"
```

### Проблема: Виджеты не появляются
**Решение**: Убедитесь что добавлены в main_layout:
```python
self.main_layout.insertWidget(0, self._new_widgets_frame)
```

---

## 📚 Дополнительные ресурсы

- `docs/ui-improvements/IMPLEMENTATION_COMPLETE.md` — полная документация
- `docs/ui-improvements/FINAL_STATUS.md` — статус реализации
- `test_ui_improvements.py` — тестовая СУИта новых модулей

---

**Если UI работает, но виджеты не подключены** — просто добавьте код из Шага 3 в main_window.py и перезапустите! 🎉
