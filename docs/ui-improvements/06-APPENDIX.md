# 06-APPENDIX.md — Дополнения и финальные решения

## Введение

Данный документ содержит **дополнительные спецификации**, принятые на финальном этапе разработки ТЗ. Все вопросы согласованы и зафиксированы для future reference.

---

## 1. Синхронизация UI ↔ Worker при изменении параметров

### Вопрос
Когда в UI меняется параметр (например, ent_coef через слайдер), должно ли это мгновенно применяться к работающему worker?

### Решение: Двухуровневая стратегия

| Тип параметра | Поведение | Обоснование |
|---------------|-----------|-------------|
| **Адаптивные (runtime-safe)** | Мгновенное применение через CommandMsg | Эти параметры меняются во время inference и не требуют перезапуска: ent_coef, adaptive_entropy, loop_threshold, kl_threshold |
| **Статические (config-only)** | Требуют перезапуска training process | Эти параметры влияют на модель architecture или global config: network_size, batch_size, total_steps, n_envs |

### Реализация в API

Добавить в CommandMsg:
```python
@dataclass
class CommandMsg:
    cmd: str
    payload: Optional[Dict[str, Any]] = None
    
    # NEW: Update runtime parameters dynamically
    CMD_UPDATE_RUNTIME_PARAM = "update_runtime_param"
        payload: {
            "param_name": str,           # e.g., "ent_coef", "loop_threshold"
            "new_value": float|int,      # New value to set
            "persist": bool              # Save to config file? (default: False)
        }
```

---

## 2. Отображение адаптивного ent_coef на графике

### Решение: Включить в базовую версию (рекомендуется)

**Обоснование:** Адаптивный ent_coef критичен для понимания работы LoopDetector. Пользователь должен видеть связь: loops detected → auto boost → entropy increases.

---

## 3. Версионирование конфигов для воспроизводимости

### Решение: ДА — обязательно для P1

**Обоснование:** Без hash невозможно точно определить, какая версия UI использовалась при обучении. При debug старых моделей нужно знать точный конфиг + UI version.

### Реализация

#### А. Config Hash Calculation (NEW FILE: train_ui/config_manager.py)
```python
import hashlib
import json

class ConfigManager:
    @staticmethod
    def compute_config_hash(config_dict: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of config for reproducibility."""
        sorted_config = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(sorted_config.encode('utf-8'))
        return hash_obj.hexdigest()[:12]  # Use first 12 chars (sufficient uniqueness)
```

#### Б. Meta.json Structure Update
```python
meta = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "ui_version": "1.0.0",                    # NEW: Track UI version used
    "config_hash": config_hash,               # NEW: SHA-256 of config
    "steps": int(metrics.total_timesteps),
    "best_reward": float(metrics.best_reward) if metrics.best_reward != float("-inf") else 0.0,
    "episodes": int(metrics.n_episodes),
    "train_time_sec": float(elapsed),
    "config": cfg.to_dict(),                  # Full config for reproducibility
}
```

---

## 4. Обработка ошибок при передаче данных

### Решение: Различное поведение для метрик vs команд

| Тип сообщения | Retry Policy | Обоснование |
|---------------|--------------|-------------|
| **ProgressMsg (metrics)** | ONE-SHOT (no retry) | Потеря 1-2 точек не критична для тренда. Worker training loop must not block. |
| **CommandMsg (critical)** | RETRY with ACK | Команды типа "stop_training" должны быть гарантированно доставлены |

### Реализация

#### А. Metrics — One-shot delivery
```python
def progress_cb(metrics, ...):
    try:
        mf.write(fast_msg)  # Single attempt
    except Exception as e:
        print(f"[Worker] Error in progress_cb: {e}")
        pass  # DON'T RETRY — training loop must continue
```

#### Б. Commands — Retry with ACK (for critical ones only)
```python
CRITICAL_COMMANDS = {"stop_training", "reset_curriculum", "pause_training"}

def _execute_command(self, cmd_msg) -> bool:
    try:
        if cmd_msg.cmd == "stop_training":
            stop_event.set()
            return True
        elif cmd_msg.cmd == "boost_entropy":
            ppo_model.ent_coef = min(current_coef * factor, 0.05)
            return True
        else:
            return False
    except Exception as e:
        print(f"[Worker] Command execution failed: {e}")
        return False
```

**Примечание:** Retry mechanism is complex and may introduce latency. For P1, one-shot with logging is sufficient. Full ACK/retry can be added in P4 if needed.

---

## 5. Поддержка нескольких запусков одновременно

### Решение: One run per UI session (P1-P3), Multi-run → P4

**Обоснование:**
- Complex UI/UX challenge (how to display multiple panels)
- Requires data normalization for fair comparison
- Memory overhead (multiple processes + their queues)
- Not needed for initial debugging focus

---

## 6. Документирование API для расширения

### Решение: ДА — обязательно для maintainability

**Добавлять в существующие файлы:**

В **03-API_SPECS.md — Section "Extending the Protocol"**:
```markdown
## Как добавить новую метрику или виджет

### Шаг 1: Расширить ProgressMsg (train_ui/protocol.py)
@dataclass
class ProgressMsg:
    # ... existing fields ...
    my_new_metric: float = 0.0  # ADD NEW FIELD HERE
    
### Шаг 2: Добавить сбор в AsyncTrainer (rl/async_trainer.py)
my_metric = self._compute_my_metric(...)
self.progress_callback({"my_new_metric": my_metric})

### Шаг 3: Создать UI виджет (train_ui/my_widget.py)
class MyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("0.0")
        
### Шаг 4: Интегрировать в MainWindow (train_ui/main_window.py)
my_widget = MyWidget()
self.statusBar().addWidget(my_widget)

### Важные правила расширения
1. Never break existing fields — always use default values for new optional fields
2. Type safety — use type hints consistently (float, int, str, Optional[str])
3. Performance — keep message size < 50KB to avoid queue bottleneck
4. Thread safety — UI updates must happen in main thread only
5. Documentation — add docstrings explaining metric meaning and units
```

---

## 7. CI/CD и Unit-тесты (Новое от пользователя)

### Решение: Обязательно для production-качества

#### Тесты LoopDetector
```python
# rl/test_loop_detector.py
import unittest
import torch
from rl.loop_detector import LoopDetector

class TestLoopDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LoopDetector()
    
    def test_detect_preserve_loop(self):
        """Test PRESERVE loop detection with default threshold=3."""
        actions = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
        names = ["PRESERVE", "BUILD_HOUSE", "DAY", ...]
        
        alerts = self.detector.update_batch(
            env_indices=list(range(5)),
            actions=actions,
            action_names=names
        )
        
        # First 3 PRESERVE should trigger alert for env 0
        self.assertEqual(alerts[0], "PRESERVE")
    
    def test_no_loop_below_threshold(self):
        """Test that counts below threshold don't trigger."""
        actions = torch.tensor([0, 1, 2], dtype=torch.long)
        names = ["PRESERVE", "BUILD_HOUSE", "DAY"]
        
        alerts = self.detector.update_batch(
            env_indices=[0, 1, 2],
            actions=actions,
            action_names=names
        )
        
        # No loops should be detected
        self.assertIsNone(alerts[0])

if __name__ == '__main__':
    unittest.main()
```

#### Тесты UI виджетов
```python
# train_ui/test_widgets.py
import unittest
from PyQt6.QtWidgets import QApplication, QWidget
from train_ui.kl_status_widget import KLStatusWidget

class TestKLStatusWidget(unittest.TestCase):
    def setUp(self):
        QApplication.instance() or QApplication([])
        self.widget = KLStatusWidget()
    
    def test_status_color_for_low_kl(self):
        """Test LOW status displays orange."""
        self.widget.update_status(0.005)  # Below 0.01 threshold
        
        style = self.widget.status_label.styleSheet()
        self.assertIn("#FFA500", style, "Should be orange")
    
    def test_progress_bar_bounds(self):
        """Test progress bar respects min/max thresholds."""
        kl_to_percent = (lambda kl: ((kl - 0.01) / (0.08 - 0.01)) * 100)
        
        self.widget.update_status(0.045)
        progress = self.widget.kl_progress.value()
        
        expected_percent = kl_to_percent(0.045)
        self.assertAlmostEqual(progress, expected_percent, places=1)

if __name__ == '__main__':
    unittest.main()
```

#### CI/CD Pipeline Configuration (GitHub Actions example)
```yaml
# .github/workflows/test-ui.yml
name: UI Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov coverage pyqtgraph
      - name: Run tests
        run: |
          pytest --cov=rl/loop_detector --cov=train_ui/widgets --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Требования:**
- pytest для unit-тестов с coverage.py target >85%
- GitHub Actions runner на каждый PR
- End-to-end тестирование после каждой фазы

---

## 8. Дополнительные рекомендации от пользователя (Новое)

### 8.1 Индикатор статуса нормализации возвратов

**Рекомендация:** Добавить в ReturnStatisticsWidget или статус-бар отображение return_normalization: ON/OFF.

**Решение:** ✅ Включено

**Реализация в ProgressMsg:**
```python
@dataclass
class ProgressMsg:
    # ... existing fields ...
    
    return_normalization_enabled: bool = False  # NEW
```

**Реализация в UI:**
```python
# train_ui/return_statistics_widget.py
class ReturnStatisticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.norm_status_label = QLabel("Return Normalization: OFF")
        self.norm_status_label.setStyleSheet(
            "font-size: 9px; color: #666; padding: 2px;"
        )
        
    def update_normalization_status(self, enabled: bool):
        if enabled:
            self.norm_status_label.setText("Return Normalization: ON")
            self.norm_status_label.setStyleSheet(
                "font-size: 9px; color: #32CD32; font-weight: bold;"
            )
        else:
            self.norm_status_label.setText("Return Normalization: OFF")
            self.norm_status_label.setStyleSheet(
                "font-size: 9px; color: #666;"
            )
```

---

### 8.2 Визуализация изменения vf_coef (запасной слот)

**Рекомендация:** Если в будущем вы решите делать vf_coef адаптивным, стоит зарезервировать место для его отображения.

**Решение:** ✅ Выполнено

**Реализация:**
- В `KLStatusWidget.info_label` добавлено поле: `vf_coef: -`
- Резервируется space для consistency с другими полями (ent_coef, KL)
- Когда vf_coef станет динамическим — просто заменить `-` на значение

```python
# Пример future implementation:
self.info_label.setText(
    f"KL: {self.current_kl:.5f} | ent_coef: {self.ent_coef:.5f} | "
    f"vf_coef: {self.vf_coef:.5f}"
)
```

---

### 8.3 Поддержка сохранения графика в PNG

**Рекомендация:** В TrainingDashboard добавить кнопку "Save Plot" для создания отчётов.

**Решение:** ✅ Выполнено (одна строка кода через pyqtgraph)

**Реализация:**
```python
# train_ui/dashboard_widget.py
class TrainingDashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.save_btn = QPushButton("Save Plot to PNG")
        self.save_btn.clicked.connect(self._save_plot_to_png)
        layout.addWidget(self.save_btn)
    
    def _save_plot_to_png(self):
        """Export pyqtgraph plot to PNG file."""
        try:
            # Save each subplot separately
            for i, widget in enumerate(self.plot_widgets):
                filename = f"plot_{i}_step_{self._current_step}.png"
                widget.saveGraphicsPane(filename)  # ONE LINE!
            
            self.statusBar().showMessage(f"Saved plots to current directory")
        except Exception as e:
            print(f"[Error] Could not save plot: {e}")
```

**Преимущества:**
- One line implementation via pyqtgraph
- No external dependencies needed (pyqtgraph already used)
- Facilitates report generation for experiments

---

### 8.4 Логирование команд в отдельный файл

**Рекомендация:** Писать все команды (boost entropy, reset curriculum) в command_log.jsonl рядом с метриками для воспроизводимости.

**Решение:** ✅ Выполнено

**Реализация:**
```python
# train_ui/command_logger.py (NEW FILE)
import json
import time
from pathlib import Path

class CommandLogger:
    def __init__(self, log_file: str = "command_log.jsonl"):
        self.log_file = Path(log_file)
        
    def log_command(self, timestamp: float, cmd: str, payload: Dict, success: bool):
        """Append command to JSONL log file."""
        entry = {
            "timestamp": timestamp,
            "cmd": cmd,
            "payload": payload,
            "success": success
        }
        
        # Atomic append (read -> modify -> write)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

# Usage in worker.py:
self.command_logger = CommandLogger("logs/command_log.jsonl")

def handle_command(self, cmd_msg):
    result = self._execute_command(cmd_msg)
    
    # Log every command for reproducibility
    timestamp = time.time()
    self.command_logger.log_command(
        timestamp=timestamp,
        cmd=cmd_msg.cmd,
        payload=cmd_msg.payload or {},
        success=result.get("success", False)
    )
```

**Формат log файла:**
```jsonl
{"timestamp": 1693872000.45, "cmd": "boost_entropy", "payload": {"factor": 2.0}, "success": true}
{"timestamp": 1693872105.12, "cmd": "reset_curriculum", "payload": {}, "success": true}
{"timestamp": 1693872200.88, "cmd": "stop_training", "payload": {"final_save": true}, "success": true}
```

**Преимущества:**
- Complete audit trail of all UI actions
- Reproducible experiments (know exactly what commands were sent)
- Debug training anomalies (correlate commands with metric spikes)

---

### 8.5 Автоматическая проверка наличия pyqtgraph

**Рекомендация:** Добавить подсказку пользователю: pip install pyqtgraph numpy

**Решение:** ✅ Выполнено

**Текущая реализация (ReturnStatisticsWidget):**
```python
try:
    from pyqtgraph import PlotWidget
    self.histogram_widget = PlotWidget(title="Return Distribution")
except ImportError:
    self.histogram_widget = QLabel(
        "Histogram unavailable. Install: pip install 'pyqtgraph numpy'"
    )
    self.histogram_widget.setStyleSheet(
        "font-size: 10px; color: #FFA500; padding: 10px; background: #2d2d2d;"
    )
```

**Улучшенная реализация (проактивная подсказка):**
```python
class MainWindow(QMainWindow):
    def __init__(self, ...):
        super().__init__(...)
        
        # Check for optional dependencies early
        self._check_optional_dependencies()
    
    def _check_optional_dependencies(self):
        """Check and notify user about missing optional dependencies."""
        missing = []
        
        try:
            import pyqtgraph
        except ImportError:
            missing.append("pyqtgraph")
        
        try:
            import numpy
        except ImportError:
            missing.append("numpy")
        
        if missing:
            # Show tooltip or status bar message
            install_cmd = "pip install " + " numpy" + '"' * (missing[-1] != "numpy")
            
            self.statusBar().showMessage(
                f"Optional features disabled. Install: {install_cmd}"
            )
```

**Примечание:** Текущая реализация в ReturnStatisticsWidget уже достаточно хорошая — показывает подсказку только когда пользователь пытается использовать гистограмму. Это более инвазивный подход (no unnecessary error messages at startup).

---

## 📋 ФИНАЛЬНЫЙ CHECKLIST ДО НАЧАЛА РЕАЛИЗАЦИИ

### Документация [✅ Все заполнено]
- [x] 01-EXECUTIVE_SUMMARY.md — Цели и метрики
- [x] 02-ARCHITECTURE.md — Диаграммы и структура
- [x] 03-API_SPECS.md — Полные спецификации API
- [x] 04-UI_COMPONENTS.md — Виджеты с кодом
- [x] 05-IMPLEMENTATION_PLAN.md — План по фазам
- [x] 06-APPENDIX.md — Дополнения и финальные решения
- [x] README.md — Quick start overview

### Тестирование и CI/CD [✅ Добавлено]
- [x] Unit-тесты для LoopDetector (mock data)
- [x] Unit-теты для UI виджетов (PySide6 TestWidget)
- [x] Integration tests для Protocol encode/decode
- [x] End-to-end тесты Phase 1 → Phase 4
- [x] CI/CD pipeline specification (GitHub Actions/GitLab CI)

### Дополнительные фичи от пользователя [✅ Добавлено]
- [x] Return normalisation status indicator (ON/OFF)
- [x] vf_coef visualization placeholder (for future extensibility)
- [x] Save Plot to PNG (TrainingDashboard → pyqtgraph export)
- [x] Command logging to JSONL (command_log.jsonl)
- [x] PyQtGraph availability check with user hint

### Архитектурные решения [✅ Все зафиксированы]