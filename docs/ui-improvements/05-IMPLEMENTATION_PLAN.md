# 📅 05-IMPLEMENTATION_PLAN.md — План реализации по фазам

## 🎯 Общая стратегия

**Подход:** Поэтапная реализация с тестированием после каждой фазы. Каждая фаза должна быть **production-ready** до перехода к следующей.

**Оценка времени:** Все оценки для одного разработчика. Команда из 2 человек может сократить время вдвое.

---

## 📋 ФАЗА 1: Core Infrastructure (День 1-2)

**Цель:** Создать базовую инфраструктуру для передачи данных и команд между worker и UI.

### День 1

#### Утро (4 часа): Расширение Protocol
```bash
# ТЗ раздел: train_ui/protocol.py

[ ] Add new fields to ProgressMsg:
    - top_actions: Dict[str, float]
    - loop_detected: bool
    - loop_action_name: Optional[str]
    - envs_with_loops: int
    - curriculum_stage_active: int
    - curriculum_next_at_step: Optional[int]

[ ] Add CommandMsg class:
    - cmd: str
    - payload: Optional[Dict]
    - Implement encode/decode

[ ] Update _REQUIRED dict in protocol.py
    - Add required fields for new message types

[ ] Test encoding/decoding with sample data
```

#### День (4 часа): Dual-channel Communication
```bash
# ТЗ раздел: train_ui/worker.py

[ ] Create command_queue parameter in run_train() signature
[ ] Implement separate threads for metrics and commands processing
    - Thread 1: Read from metrics_queue → send to UI
    - Thread 2: Read from command_queue → execute commands

[ ] Add headless mode detection:
    - def run_train(..., headless=False)
    - If headless=True, skip queue creation
    - Direct training execution

[ ] Test with --headless flag on server

[ ] Documentation: Update README.md with --headless usage
```

### День 2

#### Утро (4 часа): LoopDetector Module
```bash
# ТЗ раздел: rl/loop_detector.py (NEW FILE)

[ ] Create LoopDetector class:
    - __init__(threshold_config=None)
    - _state: Dict[int, Tuple[str, int]]
    - _history: List[Dict] (last 1000 updates)

[ ] Implement update_batch() method:
    - Process multiple envs at once
    - Track consecutive actions per env
    - Return dict of alerts (env_idx -> action_name or None)

[ ] Implement get_stats() method:
    - Calculate total_envs, envs_with_loops
    - Compute max/avg consecutive counts
    - Calculate loop_rate_percent

[ ] Test with mock data (generate synthetic loops)
```

#### День (4 часа): AsyncTrainer Integration
```bash
# ТЗ раздел: rl/async_trainer.py

[ ] Add LoopDetector instantiation in __init__():
    self.loop_detector = LoopDetector()

[ ] Modify _collect_rollout() method:
    - Call loop_detector.update_batch() after action sampling
    - Extract alerts from results
    - Add to rollout_info dict

[ ] Create auto-boost mechanism:
    - detect_and_boost_entropy() called every 10k steps
    - Boost ent_coef by ×1.3 max cap at 0.05
    - Log the change with reason

[ ] Modify progress_callback to send loop stats:
    - Add loop_detected, loop_action_name, envs_with_loops
    - Get curriculum progress via em.get_curriculum_progress()

[ ] Test end-to-end training run with monitoring
```

---

## 📋 ФАЗА 2: Metrics Collection & Display (День 3-4)

**Цель:** Реализовать сбор метрик и создание всех UI виджетов.

### День 3

#### Утро (4 часа): EnvManager Extensions
```bash
# ТЗ раздел: rl/env_manager.py

[ ] Add get_allowed_buildings_for_stage(stage_id) method:
    - Implement default curriculum schedule
    - Return list of building names per stage

[ ] Add get_curriculum_progress(current_step) method:
    - Calculate progress_percent to next transition
    - Find next_stage_step from schedule
    - Format available_actions string (first 5 buildings)

[ ] Modify _should_advance_curriculum():
    - Update last_stage_change_step after promotion
    - Log curriculum transitions

[ ] Test with actual training run
```

#### День (4 часа): Top Actions Collection
```bash
# ТЗ раздел: rl/async_trainer.py

[ ] Add action name mapping in EnvManager or AsyncTrainer:
    - List[str] action_names = [...]
    - Or property: @property def action_names(self) -> List[str]

[ ] Modify _collect_rollout() to count action frequencies:
    - torch.bincount(actions.flatten(), minlength=n_actions)
    - Create list of (count, name) pairs
    - Sort and take top 5

[ ] Calculate percentages for each top action:
    - total_actions = sum(all counts)
    - percentage = count / total_actions * 100
    - Round to 4 decimal places

[ ] Send top_actions dict in progress_callback:
    - Format: {"PRESERVE": 0.48, "BUILD_HOUSE": 0.12, ...}

[ ] Test with debug output (print action distribution)
```

### День 4

#### Утро (4 часа): KLStatusWidget + CurriculumProgressWidget
```bash
# ТЗ раздел: train_ui/kl_status_widget.py (NEW FILE)

[ ] Create KLStatusWidget class:
    - Status label with color coding (LOW/OPTIMAL/HIGH)
    - Progress bar (0.01-0.8 normalized to 0-100%)
    - Info label with exact values + ent_coef

[ ] Implement status calculation logic:
    - kl < 0.01 → LOW (orange)
    - 0.01 <= kl < 0.025 → OK (lime)
    - 0.025 <= kl < 0.08 → OPTIMAL (turquoise)
    - kl >= 0.08 → HIGH (red)

[ ] Test with different KL values (simulate updates)

[ ] ---
# ТЗ раздел: train_ui/curriculum_progress_widget.py (NEW FILE)

[ ] Create CurriculumProgressWidget class:
    - Stage badge "Stage: 0/3"
    - Progress bar to next transition
    - Upcoming stages list (max 3)
    - Available actions info

[ ] Implement update() method with all parameters
[ ] Test with mock curriculum schedule data
```

#### День (4 часа): ActionLoopWidget + ReturnStatisticsWidget
```bash
# ТЗ раздел: train_ui/action_loop_widget.py (NEW FILE)

[ ] Create ActionLoopWidget class:
    - Alert label (hidden by default)
    - Action name display
    - Consecutive count and threshold info
    - Statistics grid for history window

[ ] Implement set_loop_status() method:
    - Show/hide alert based on active flag
    - Color code by severity (orange vs red)
    - Format count/threshold text

[ ] Test with loop detection scenarios

[ ] ---
# ТЗ раздел: train_ui/return_statistics_widget.py (NEW FILE)

[ ] Create ReturnStatisticsWidget class:
    - 4 labels: avg, median, max, min
    - Histogram area (pyqtgraph or fallback)

[ ] Implement update_statistics() method:
    - Calculate mean, median (sorted[n//2])
    - Compute max/min from list
    - Normalize returns for histogram display

[ ] Optional: Install pyqtgraph for histogram
    pip install pyqtgraph numpy

[ ] Test with sample return data distribution
```

---

## 📋 ФАЗА 3: Advanced Interactivity (День 5-6)

**Цель:** Добавить Quick Actions, интерактивные графики, асинхронные обновления.

### День 5

#### Утро (4 часа): Quick Actions Panel
```bash
# ТЗ раздел: train_ui/main_window.py + protocol.py

[ ] Extend Protocol with more commands:
    - CMD_PAUSE_TRAINING: {"save_checkpoint": bool}
    - CMD_RESUME_TRAINING: {}
    - CMD_STOP_TRAINING: {"final_save": bool}

[ ] Create QuickActionsWidget class:
    - Pause/Resume buttons
    - Boost Entropy ×2 button
    - Reset Curriculum button

[ ] Implement _send_command() method:
    - Encode command using P.encode_command(cmd, payload)
    - Put into command_queue via UI queue mechanism

[ ] Handle commands in worker thread:
    - boost_entropy: ppo_model.ent_coef = min(coef * 2.0, 0.05)
    - reset_curriculum: em.set_curriculum_stage(0)
    - pause/resume: Toggle training state flag
    - stop_training: Set stop_event.is_set()

[ ] Test each command individually
```

#### День (4 часа): Interactive Training Dashboard
```bash
# ТЗ раздел: train_ui/dashboard_widget.py (NEW FILE)

[ ] Install pyqtgraph if not present:
    pip install pyqtgraph

[ ] Create TrainingDashboard widget class:
    - KL divergence plot with target line
    - Entropy plot
    - Action distribution stacked bar chart

[ ] Implement update_data() method:
    - Append new data point to curve
    - Auto-scroll X axis to show last 1000 steps
    - Enable zoom/pan (view_box.setMouseEnabled)

[ ] Add interactivity:
    - Mouse wheel for zoom
    - Drag for pan
    - Hover tooltips (pyqtgraph built-in)

[ ] Test real-time updates during training

[ ] Optional: Save plot to file on training end
```

---

## 📋 ФАЗА 4: Polish & Optimization (День 7)

**Цель:** Оптимизация производительности, обработка ошибок, документация.

### День 7 (Полный день - 8 часов)

#### Утро (4 часа): Error Handling & Logging
```bash
# ТЗ раздел: train_ui/worker.py + protocol.py

[ ] Wrap all progress_callback calls in try/except:
    - Log errors but don't crash training
    - Use structured logging format

[ ] Add queue overflow protection:
    - Use multiprocessing.Queue with maxsize=1000
    - Handle get_nowait() with Empty exception

[ ] Implement message validation:
    - Check for corrupted JSON before decode
    - Validate required fields in each msg type
    - Log invalid messages but continue processing

[ ] Add comprehensive logging:
    - [Worker] Info messages for training progress
    - [LoopDetector] Auto-boost events with timestamps
    - [UI] Command received messages
    - [Error] Exception stack traces
```

#### День (4 часа): Performance Optimization & Documentation
```bash
[ ] Optimize message sending frequency:
    - Send full stats every 1000 steps (not every step)
    - Send only delta changes for critical metrics (KL, loops)
    - Batch multiple updates into single message if possible

[ ] Memory profiling:
    - Monitor heap usage during long training runs
    - Ensure no memory leaks in UI threads
    - Limit history window sizes appropriately

[ ] Write/update documentation:
    - README.md: Quick start guide with examples
    - API docs for new classes (docstrings)
    - Architecture diagram (optional, Mermaid.js format)

[ ] Add configuration options:
    - --curriculum-auto-promotion: Enable/disable auto-advancement
    - --loop-detection-threshold: Override default thresholds
    - --metrics-interval: Steps between full progress updates

[ ] Final testing suite:
    - Run complete training with all features enabled
    - Test headless mode
    - Test each Quick Action command
    - Verify color coding accuracy
    - Check widget responsiveness under load

[ ] Code review checklist:
    - Type hints on all new methods
    - Docstrings following Google style
    - Consistent naming conventions
    - No dead code or TODOs
```

---

## 📊 ОЦЕНКА ПРОГРЕССА (Metrics)

### Код по фазам
| Фаза | Строк кода | Файлов | Классов | Методов |
|------|------------|--------|---------|---------|
| P1 | ~800 | 3 | 4 | 25 |
| P2 | ~1500 | 7 | 9 | 45 |
| P3 | ~600 | 2 | 3 | 15 |
| P4 | ~200 | - | - | 8 |
| **ИТОГО** | **~3100** | **12** | **16** | **93** |

### Время по фазам
| Фаза | Дней | Часов | % проекта |
|------|------|-------|-----------|
| P1 | 2 | 16 | 20% |
| P2 | 2 | 16 | 25% |
| P3 | 1 | 8 | 10% |
| P4 | 1 | 8 | 10% |
| **ИТОГО** | **6** | **48** | **65%** |

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Release Testing
- [ ] Run training without UI (headless)
- [ ] Run training with UI in full window
- [ ] Test resize/minimize during training
- [ ] Verify all Quick Actions work correctly
- [ ] Check loop detection triggers at correct thresholds
- [ ] Confirm curriculum transitions log correctly
- [ ] Validate KL status color coding accuracy

### Documentation
- [ ] README.md updated with new features
- [ ] API docstrings complete
- [ ] Architecture diagram added (optional)
- [ ] Example training configs included

### Performance Benchmarks
```bash
# Expected performance targets:
- UI FPS ≥ 30 (no lag during updates)
- Worker overhead < 2% of total training time
- Memory usage ≤ 8GB (per process)
- Command latency < 100ms
```

---

## 🎯 КРИТЕРИИ ПРИЕМКИ (Acceptance Criteria)

### P1 - Critical Features
1. ✅ UI показывает top-5 действий с процентами (обновление каждые 2048 шагов)
2. ✅ KL Status Widget отображает цветной индикатор (LOW/OPTIMAL/HIGH)
3. ✅ CurriculumProgressWidget показывает текущий stage и прогресс до перехода
4. ✅ ActionLoopWidget предупреждает когда >30% envs в loops
5. ✅ Dual-channel communication работает без блокировок

### P2 - Display Features  
6. ✅ ReturnStatisticsWidget вычисляет avg/med/max/min за последние 50 эпизодов
7. ✅ Quick Actions Panel позволяет boost entropy, reset curriculum, pause/resume
8. ✅ Interactive dashboard с pyqtgraph (zoom/pan/hover)
9. ✅ Headless mode работает корректно без UI

### P3 - Advanced Features (Optional)
10. Профили настроек (экспорт/import JSON)
11. Сравнение multiple runs side-by-side
12. GPU/memory monitoring в статус бар

---

## 🔧 ЗАВИСИМОСТИ И ПОТРЕБНОСТИ

### Новые зависимости (если нужны):
```bash
pip install pyqtgraph numpy  # Для графиков (опционально, не критично)
```

### System requirements:
- Python 3.8+
- PySide6 (already present)
- multiprocessing support (standard library)
- 4GB+ RAM recommended for full training with UI

---

## 📝 ЗАПИСЬ РЕШЕНИЙ (Decision Log)

| Дата | Решение | Обоснование | Альтернативы рассмотрены |
|------|---------|-------------|--------------------------|
| 2026-09-04 | Отложить профили настроек на P3 | Фокус на критичных фичах для отладки | Экспорт/import в P1 |
| 2026-09-04 | Отдельная command_queue | Разделение Concerns, нет конфликтов приоритетов | Общая двусторонняя очередь |
| 2026-09-04 | UI загружает curriculum из конфига | Экономия трафика между процессами | Worker отправляет каждый шаг |
| 2026-09-04 | Headless режим необходим | Для серверов/Docker/CI pipelines | Только десктопное приложение |
| 2026-09-04 | Hybrid history (worker→current, UI→history) | UI строит long-term statistics гибче | Worker хранит всё |
| 2026-09-04 | PyQtGraph для графиков | Real-time performance лучше matplotlib | Matplotlib с FigureCanvasQTAgg |

---

## ✅ ФИНАЛЬНЫЕ ШАГИ ПОСЛЕ РЕАЛИЗАЦИИ

1. **Unit Testing** - Написать тесты для новых классов (LoopDetector, widgets)
2. **Integration Testing** - End-to-end training с полным UI
3. **Documentation** - README.md, API docstrings
4. **Performance Testing** - 1M+ step run monitoring
5. **User Manual** - Quick start guide для пользователей
6. **Release Notes** - Список изменений для version bump

---

## 🎉 УСПЕШНОЕ ЗАВЕРШЕНИЕ

**Ожидаемый результат:**
- Полностью рабочий UI с мониторингом всех критичных метрик
- Автоматическая детекция action loops и предупреждения
- Интерактивное управление обучением через Quick Actions
- Headless режим для серверов/CID/CI
- Production-ready код с error handling и logging

**Продолжительность:** 6 рабочих дней (48 часов кодинга)  
**Сложность:** Medium-High (интеграция multiprocessing, real-time UI updates)  
**Риск:** Low-Medium (если тестировать поэтапно по фазам)
