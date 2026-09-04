# 🎨 04-UI_COMPONENTS.md — Спецификации UI виджетов

## 📐 Main Window Layout Structure (Responsive Design)

**Ключевой принцип:** Все виджеты используют **QSizePolicy** + **stretch factors** для адаптации к разным разрешениям.

### Размеры экрана и масштабирование:

| Разрешение | Окно MainWindow | Scale factor |
|------------|-----------------|--------------|
| 1080p (1920x1080) | 1400x750 | 1.0× |
| 2K (2560x1440) | 1750x937 | 1.25× |
| 4K (3840x2160) | 2000x1000 | 1.4× |

### Layout структура с stretch factors:

```
MainWindow (resizable, QSizePolicy.Expanding)
├── Title Bar (fixed height)
│   └── "Сахалинская колония — обучение моделей"
│
├── Main Content Area (QHBoxLayout, stretch=100)
│   ├── Left Panel - Parameter Configuration (stretch=35)
│   │   └── PARAM_PANEL_WIDGET
│   │       ├── Curriculum Stage Selector (QComboBox, fixed width 120px)
│   │       ├── Learning Rate SpinBox + ×2/÷2 buttons (auto-expand)
│   │       ├── All other PARAM_SPECS... (QVBoxLayout, auto)
│   │       └── REWARD_SPECS (QHBoxLayout, stretch=1)
│   │
│   └── Right Panel - Training Monitor (stretch=65)
│       ├── Top Metrics Display (QScrollArea if needed)
│       │   ├── FPS Bar (fixed height 20px, expand horizontally)
│       │   ├── Best Reward Bar (same)
│       │   ├── Policy Loss Line Chart (QPlotWidget, expand both)
│       │   ├── Value Loss Line Chart (same)
│       │   ├── Entropy Value (label, fixed)
│       │   └── KL Progress Bar (with status, auto-expand)
│       │
│       └── Advanced Monitoring Widgets (QVBoxLayout)
│           ├── KLStatusWidget (minimum height 50px, stretch=1)
│           ├── CurriculumProgressWidget (minimum height 60px, stretch=2)
│           ├── ActionLoopWidget (minimum height 60px, stretch=2)
│           ├── ReturnStatisticsWidget (minimum height 80px, stretch=3)
│           └── CurriculumMetricsWidget (minimum height 50px, stretch=1)
│
├── Status Bar (bottom, QStatusBar, dynamic height)
│   └── Steps: X / Y | Time: HH:MM:SS | GPU Usage: XX%
│
└── Menus: File | Edit | Help
```

### Ключевые правила адаптивного layout:

1. **MainWindow** — Expandable frame (min 800x600, max 4K+ width)
2. **Panels** — Use stretch factors (35% vs 65% split)
3. **Labels** — Horizontal expand only (QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
4. **Buttons/SpinBoxes** — Fixed height for consistency
5. **Widgets min sizes** — Never use fixedSize(), always setMinimumSize()

---

## 🧩 Detailed Widget Specifications
│   │       ├── Curriculum Stage Selector (dropdown 0-N)
│   │       ├── Learning Rate SpinBox + ×2/÷2 buttons
│   │       ├── Batch Size SpinBox + ×2/÷2 buttons
│   │       ├── All other PARAM_SPECS...
│   │       └── REWARD_SPECS (reward parameters)...
│   │
│   └── Right Panel (65% width): Training Monitor
│       ├── Metrics Display Area
│       │   ├── FPS Bar
│       │   ├── Best Reward Bar
│       │   ├── Policy Loss Line Chart
│       │   ├── Value Loss Line Chart
│       │   ├── Entropy Value
│       │   └── KL Progress Bar (with status)
│       │
│       └── Advanced Monitoring Widgets (bottom section)
│           ├── KLStatusWidget (60px height)
│           ├── CurriculumProgressWidget (80px height)
│           ├── ActionLoopWidget (80px height)
│           ├── ReturnStatisticsWidget (120px height + histogram)
│           └── CurriculumMetricsWidget (60px height)
│
├── Status Bar (bottom, 30px)
│   └── Steps: X / Y | Time: HH:MM:SS | GPU Usage: XX%
│
└── Menus: File | Edit | Help
```

## 🧩 Detailed Widget Specifications

### 1. KLStatusWidget

**Файл:** `train_ui/kl_status_widget.py` (new file)

```python
class KLStatusWidget(QWidget):
    """Цветной индикатор статуса KL divergence."""
    
    # Size: ~200x60px minimum, auto-expand horizontally
    # Background: #1e1e1e (dark theme match)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("KLStatusWidget")
        
        # Internal state
        self.current_kl = 0.0
        self.ent_coef = 0.0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # Status label with color coding + vf_coef (optional future)
        self.status_label = QLabel("STATUS: UNKNOWN")
        self.status_label.setObjectName("kl_status_text")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 12px;"
        )
        layout.addWidget(self.status_label)
        
        # Progress bar showing normalized KL position
        self.kl_progress = QProgressBar()
        self.kl_progress.setFormat("%p%")
        self.kl_progress.setMinimum(0)      # KL = 0.01 (LOW boundary)
        self.kl_progress.setMaximum(100)    # KL = 0.08 (HIGH boundary)
        self.kl_progress.setValue(50)       # Default middle
        self.kl_progress.setRange(0, 100)   # Manual override for custom range
        layout.addWidget(self.kl_progress)
        
        # Info label with precise values + vf_coef (reserved slot)
        self.info_label = QLabel("KL: 0.00000 | ent_coef: 0.01000 | vf_coef: -")
        self.info_label.setObjectName("kl_info_text")
        self.info_label.setStyleSheet(
            "font-size: 9px; color: #87CEEB;"
        )
        layout.addWidget(self.info_label)
        
        # VF_COEF placeholder (for future adaptive implementation)
        # Currently shows "-" but reserves space for consistency
        
    def update_status(self, kl: float):
        """Calculate and update status based on KL value."""
        
        thresholds = {
            "LOW": 0.01,      # Orange warning - too deterministic
            "OPTIMAL": 0.025, # Lime OK - acceptable range
            "HIGH": 0.08      # Red warning - too much variance
        }
        
        if kl < thresholds["LOW"]:
            status_text = "LOW ⚠️"
            color = "#FFA500"    # Orange
            self._set_style(color, "bold")
        elif kl < thresholds["OPTIMAL"]:
            status_text = "OK ✓"
            color = "#32CD32"    # Lime green
            self._set_style(color, "normal")
        elif kl < thresholds["HIGH"]:
            status_text = "OPTIMAL ⭐"
            color = "#00CED1"    # Dark turquoise
            self._set_style(color, "bold", icon="★")
        else:
            status_text = "HIGH ⚠️"
            color = "#FF4444"    # Red
            self._set_style(color, "bold")
        
        self.status_label.setText(f"KL Status: {status_text}")
    
    def set_ent_coef(self, value: float):
        """Display current ent_coef value."""
        self.ent_coef = value
        self.info_label.setText(
            f"KL: {self.current_kl:.5f} | Target: 0.03 ± 0.02 | "
            f"ent_coef: {value:.5f}"
        )
    
    def _set_style(self, color: str, weight: str = "normal", icon: str = ""):
        """Apply consistent styling."""
        self.status_label.setStyleSheet(
            f"color: {color}; font-weight: {weight};"
        )
```

---

### 2. CurriculumProgressWidget

**Файл:** `train_ui/curriculum_progress_widget.py` (new file)

```python
class CurriculumProgressWidget(QWidget):
    """Визуализация прогресса curriculum learning."""
    
    # Size: ~300x80px, fixed height for compact display
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CurriculumProgressWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        
        # Current stage badge (color-coded by progress)
        self.stage_badge = QLabel("Stage: 0/3")
        self.stage_badge.setObjectName("curriculum_stage_badge")
        self.stage_badge.setStyleSheet(
            """
            background: #4CAF50;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
            """
        )
        layout.addWidget(self.stage_badge)
        
        # Progress bar to next transition
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setLabelText("Progress: 0%")
        layout.addWidget(self.progress_bar)
        
        # Upcoming stages list
        self.upcoming_label = QLabel("Upcoming:\n→ Stage 1 at step 200,000\n→ Stage 2 at step 800,000")
        self.upcoming_label.setObjectName("curriculum_upcoming_text")
        self.upcoming_label.setStyleSheet(
            "font-size: 9px; color: #aaa; background: transparent;"
        )
        layout.addWidget(self.upcoming_label)
        
        # Available actions info
        self.actions_label = QLabel("Available buildings: House, Farm, Road")
        self.actions_label.setObjectName("curriculum_actions_text")
        self.actions_label.setStyleSheet(
            "font-size: 9px; color: #87CEEB;"
        )
        layout.addWidget(self.actions_label)
    
    def update_progress(
        self,
        stage: int,
        total_stages: int,
        progress_percent: float,
        next_transitions: List[Dict[str, Any]],  # [{"stage": 1, "step": 200_000}, ...]
        available_actions: str
    ):
        """Update all curriculum display elements."""
        
        self.stage_badge.setText(f"Stage: {stage}/{total_stages}")
        
        # Update progress bar
        self.progress_bar.setValue(int(progress_percent))
        self.progress_bar.setLabelText(
            f"Progress to next transition: {progress_percent:.1f}%"
        )
        
        # Format upcoming transitions (show max 3)
        upcoming_text = "Upcoming:\n"
        for i, trans in enumerate(next_transitions[:3]):
            step_str = f"{trans['step']:,}" if trans.get('step') else "∞"
            stage_num = trans.get('stage', stage + 1 + i)
            upcoming_text += f"→ Stage {stage_num} at step {step_str}\n"
        
        self.upcoming_label.setText(upcoming_text.rstrip())
        
        # Available actions (truncated to 20 chars per building name)
        actions = available_actions.split(", ")
        short_list = ", ".join(
            b[:20] + ("..." if len(b) > 20 else "") 
            for b in actions[:5]  # Show first 5 buildings
        )
        self.actions_label.setText(f"Available: {short_list or 'None'}")
```

---

### 3. ActionLoopWidget

**Файл:** `train_ui/action_loop_widget.py` (new file)

```python
class ActionLoopWidget(QWidget):
    """Мониторинг и отображение action loops."""
    
    # Size: ~400x80px, compact but informative
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ActionLoopWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # Alert label (hidden by default, appears on loop detection)
        self.alert_label = QLabel("⚠️ ACTION LOOP DETECTED!")
        self.alert_label.setObjectName("loop_alert_text")
        self.alert_label.setVisible(False)
        self.alert_label.setStyleSheet(
            "color: #FF4444; font-weight: bold; font-size: 12px;"
        )
        layout.addWidget(self.alert_label)
        
        # Action name if loop detected
        self.action_name = QLabel("Action: PRESERVE")
        self.action_name.setObjectName("loop_action_text")
        self.action_name.setStyleSheet(
            "color: #aaa; font-size: 10px;"
        )
        layout.addWidget(self.action_name)
        
        # Consecutive count and threshold info
        self.count_info = QLabel("Env 45: Consecutive count: 8/5 (WARNING)")
        self.count_info.setObjectName("loop_count_text")
        self.count_info.setStyleSheet(
            "font-size: 9px; color: #87CEEB;"
        )
        layout.addWidget(self.count_info)
        
        # Statistics grid for last N updates (simulating 24h window)
        self.stats_label = QLabel("Last 100 updates:")
        self.stats_label.setObjectName("loop_stats_header")
        self.stats_label.setStyleSheet(
            "font-weight: bold; font-size: 10px; margin-top: 5px;"
        )
        layout.addWidget(self.stats_label)
        
        self.stats_grid = QGridLayout()
        layout.addLayout(self.stats_grid)
    
    def set_loop_status(
        self,
        active: bool,
        action_name: Optional[str] = None,
        count: int = 0,
        threshold: int = 5,
        env_idx: int = -1
    ):
        """Update active loop detection status."""
        
        if active and action_name:
            self.alert_label.setVisible(True)
            self.action_name.setText(f"Action: {action_name}")
            self.count_info.setText(
                f"Env {env_idx}: Consecutive count: {count}/{threshold} "
                f({'CRITICAL' if count > threshold * 2 else 'WARNING'})"
            )
            
            # Color based on severity
            if count > threshold * 2:
                self.alert_label.setStyleSheet(
                    "color: #FF0000; font-weight: bold;"
                )
            elif count > threshold:
                self.alert_label.setStyleSheet(
                    "color: #FFA500; font-weight: bold;"
                )
            else:
                self.alert_label.setStyleSheet(
                    "color: #8B0000; font-weight: bold;"
                )
        else:
            self.alert_label.setVisible(False)
            self.action_name.setText("")
            self.count_info.setText("No active loops detected")
    
    def update_history_stats(self, stats: Dict[str, Any]):
        """Update aggregated history statistics from UI window."""
        
        # Clear existing grid
        self.stats_grid.clear()
        
        rows = [
            ["Envs with loops", f"{stats.get('total_loops', 0)}"],
            ["Max consecutive", f"{stats.get('max_consecutive', 0)}"],
            ["Avg consecutive", f"{stats.get('avg_consecutive', 0):.1f}"],
            ["Loop rate (window)", f"{stats.get('loop_rate_percent', 0):.1f}%"],
        ]
        
        for i, (label, value) in enumerate(rows):
            self.stats_grid.addWidget(QLabel(label), i // 2, 0)
            self.stats_grid.addWidget(QLabel(value), i // 2, 1)
```

---

### 4. ReturnStatisticsWidget

**Файл:** `train_ui/return_statistics_widget.py` (new file)

```python
class ReturnStatisticsWidget(QWidget):
    """Статистика возвратов с гистограммой распределения."""
    
    # Size: ~500x120px
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ReturnStatisticsWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        
        # Statistics labels row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(20)
        
        label_templates = [
            ("Avg Return (last 50 eps)", "0.0"),
            ("Median Return", "0.0"),
            ("Max Return", "0.0"),
            ("Min Return", "0.0"),
        ]
        
        self.stats_labels: List[QLabel] = []
        for label_name, value in label_templates:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            name_label = QLabel(label_name)
            name_label.setObjectName("return_stat_label")
            name_label.setStyleSheet("font-size: 10px; color: #888;")
            row_layout.addWidget(name_label)
            
            value_label = QLabel(value)
            value_label.setObjectName("return_value_text")
            value_label.setStyleSheet(
                "font-size: 11px; color: #87CEEB; font-weight: bold;"
            )
            row_layout.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignRight)
            
            self.stats_labels.append((name_label, value_label))
            stats_row.addLayout(row_layout)
        
        layout.addLayout(stats_row)
        
        # Histogram using pyqtgraph (or fallback to bar chart)
        try:
            from pyqtgraph import PlotWidget
            self.histogram_widget = PlotWidget(title="Return Distribution")
            self.histogram_widget.setFixedHeight(60)
            layout.addWidget(self.histogram_widget)
            
            # Store for updates
            self._dist_curve: Optional[PlotCurveItem] = None
            self._x_axis: Optional[List[float]] = []
        except ImportError:
            self.histogram_widget = QLabel("Histogram unavailable (install pyqtgraph)")
            self.histogram_widget.setStyleSheet(
                "font-size: 9px; color: #666; padding: 5px;"
            )
            layout.addWidget(self.histogram_widget)
    
    def update_statistics(self, returns: List[float], best_reward: float):
        """Update statistics and histogram."""
        
        if not returns:
            return
        
        # Calculate statistics
        avg = sum(returns) / len(returns)
        sorted_returns = sorted(returns)
        n = len(sorted_returns)
        med = sorted_returns[n // 2]
        max_val = max(returns)
        min_val = min(returns)
        
        # Update labels
        for name_label, value_label in self.stats_labels:
            if "Avg" in name_label.text():
                value_label.setText(f"{avg:.2f}")
            elif "Median" in name_label.text():
                value_label.setText(f"{med:.2f}")
            elif "Max" in name_label.text():
                value_label.setText(f"{max_val:.2f}")
            elif "Min" in name_label.text():
                value_label.setText(f"{min_val:.2f}")
        
        # Update histogram (if pyqtgraph available)
        if hasattr(self, 'histogram_widget') and self._dist_curve is not None:
            # Normalize to 0-1 range for visualization
            normalized = [r / max_val if max_val > 0 else 0 for r in returns]
            
            # Append new data point (simple line plot)
            import numpy as np
            x_data = list(self._x_axis) + [len(x_data)]
            y_data = list(self._dist_curve.data()[1]) + [normalized[-1]]
            
            self._dist_curve.setData(
                x=np.array(x_data), 
                y=np.array(y_data),
                pen='b'
            )
            
            # Auto-scroll to show recent returns
            max_x = max(x_data) if x_data else 0
            self.histogram_widget.setXRange(max_x - 50, max_x, padding=0)
```

---

### 5. CurriculumMetricsWidget

**Файл:** `train_ui/curriculum_metrics_widget.py` (new file)

```python
class CurriculumMetricsWidget(QWidget):
    """Индикаторы метрик для curriculum transitions."""
    
    # Size: ~300x60px
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CurriculumMetricsWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        
        # KL threshold indicator
        kl_row = QHBoxLayout()
        kl_label = QLabel(f"KL: {self._format_threshold(0.03)} (Target)")
        kl_label.setObjectName("threshold_label")
        kl_label.setStyleSheet("font-size: 10px; color: #888;")
        
        self.kl_indicator = QLabel("OK ✓")
        self.kl_indicator.setObjectName("threshold_indicator")
        self.kl_indicator.setStyleSheet(
            "color: #32CD32; font-weight: bold; font-size: 10px;"
        )
        
        kl_row.addWidget(kl_label)
        kl_row.addSpacing(20)
        kl_row.addWidget(self.kl_indicator)
        layout.addLayout(kl_row)
        
        # Entropy threshold indicator (if applicable)
        ent_row = QHBoxLayout()
        ent_label = QLabel(f"Entropy: {self._format_threshold(2.8)} (Threshold)")
        ent_label.setObjectName("threshold_label")
        ent_label.setStyleSheet("font-size: 10px; color: #888;")
        
        self.ent_indicator = QLabel("OK ✓")
        self.ent_indicator.setObjectName("threshold_indicator")
        self.ent_indicator.setStyleSheet(
            "color: #32CD32; font-weight: bold; font-size: 10px;"
        )
        
        ent_row.addWidget(ent_label)
        ent_row.addSpacing(20)
        ent_row.addWidget(self.ent_indicator)
        layout.addLayout(ent_row)
        
        # Auto-promotion status
        promo_row = QHBoxLayout()
        promo_label = QLabel("Auto-promotion:")
        promo_label.setObjectName("threshold_label")
        promo_label.setStyleSheet("font-size: 10px; color: #888;")
        
        self.promo_status = QLabel("ENABLED ✓")
        self.promo_status.setObjectName("promo_status_text")
        self.promo_status.setStyleSheet(
            "color: #4CAF50; font-weight: bold; font-size: 10px;"
        )
        
        promo_row.addWidget(promo_label)
        promo_row.addSpacing(10)
        promo_row.addWidget(self.promo_status)
        layout.addLayout(promo_row)
    
    def update_thresholds(
        self,
        current_kl: float,
        kl_threshold: float = 0.03,
        current_entropy: float = None,
        entropy_threshold: float = 2.8
    ):
        """Update threshold indicators based on current values."""
        
        # KL indicator
        if current_kl < kl_threshold * 0.8:
            self.kl_indicator.setText("LOW ⚠️")
            self.kl_indicator.setStyleSheet("color: #FFA500;")
        elif current_kl > kl_threshold * 1.2:
            self.kl_indicator.setText("HIGH ⚠️")
            self.kl_indicator.setStyleSheet("color: #FF4444;")
        else:
            self.kl_indicator.setText("OK ✓")
            self.kl_indicator.setStyleSheet("color: #32CD32;")
        
        # Entropy indicator (if entropy provided)
        if current_entropy is not None:
            if current_entropy < entropy_threshold * 0.8:
                self.ent_indicator.setText("LOW ⚠️")
                self.ent_indicator.setStyleSheet("color: #FFA500;")
            elif current_entropy > entropy_threshold * 1.2:
                self.ent_indicator.setText("HIGH ⚠️")
                self.ent_indicator.setStyleSheet("color: #FF4444;")
            else:
                self.ent_indicator.setText("OK ✓")
                self.ent_indicator.setStyleSheet("color: #32CD32;")
        
        # Update labels with actual values
        kl_label = next(
            (l for l in layout.children() if isinstance(l, QHBoxLayout) and 
             hasattr(l, 'widget') and l.widget().text().startswith('KL:')),
            None
        )
        if kl_label:
            kl_label.setText(f"KL: {self._format_value(current_kl)} / "
                           f"{self._format_threshold(kl_threshold)} (Target)")
    
    def update_promotion_status(self, promoted_recently: bool = False):
        """Update auto-promotion status text."""
        
        if promoted_recently:
            self.promo_status.setText("PROMOTED ✓")
            self.promo_status.setStyleSheet(
                "color: #00CED1; font-weight: bold;"
            )
        else:
            self.promo_status.setText("ENABLED ✓")
            self.promo_status.setStyleSheet(
                "color: #4CAF50; font-weight: bold;"
            )
    
    def _format_value(self, value: float) -> str:
        if value >= 1.0:
            return f"{value:.0f}"
        elif value >= 0.01:
            return f"{value:.3f}"
        else:
            return f"{value:.5f}"
    
    def _format_threshold(self, threshold: float) -> str:
        """Format threshold for display."""
        if threshold >= 1.0:
            return f"{threshold:.0f}"
        elif threshold >= 0.01:
            return f"{threshold:.3f}"
        else:
            return f"{threshold:.5f}"
```

---

## 📊 MainWindow Integration Code

**Файл:** `train_ui/main_window.py` (extensions)

```python
class MainWindow(QMainWindow):
    # ... existing code ...
    
    def __init__(self, config_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        
        # ... initialization ...
        
        # NEW: Add monitor panel to status bar area
        self.monitor_panel = QWidget()
        self.monitor_panel.setStyleSheet("background: #1e1e1e;")
        self.statusBar().addWidget(self.monitor_panel)
        
        monitor_layout = QVBoxLayout(self.monitor_panel)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        monitor_layout.setSpacing(8)
        
        # Row 1: KL Status + Ent Coef
        kl_row = QHBoxLayout()
        self.kl_status_widget = KLStatusWidget()
        self.kl_status_widget.setMinimumHeight(55)
        kl_row.addWidget(self.kl_status_widget, stretch=2)
        
        ent_coef_label = QLabel(f"ent_coef: {DEFAULT_PARAMS['ent_coef']:.5f}")
        ent_coef_label.setStyleSheet(
            "font-size: 10px; padding: 5px; background: #2d2d2d;"
        )
        kl_row.addWidget(ent_coef_label, stretch=1)
        
        monitor_layout.addLayout(kl_row)
        
        # Row 2: Curriculum Progress + Action Loop
        mid_row = QHBoxLayout()
        self.curriculum_widget = CurriculumProgressWidget()
        self.curriculum_widget.setMinimumHeight(70)
        mid_row.addWidget(self.curriculum_widget, stretch=2)
        
        self.loop_widget = ActionLoopWidget()
        self.loop_widget.setMinimumHeight(70)
        mid_row.addWidget(self.loop_widget, stretch=1)
        
        monitor_layout.addLayout(mid_row)
        
        # Row 3: Return Statistics + Curriculum Metrics
        bottom_row = QHBoxLayout()
        return_widget = ReturnStatisticsWidget()
        return_widget.setMinimumHeight(100)
        bottom_row.addWidget(return_widget, stretch=2)
        
        curriculum_metrics_widget = CurriculumMetricsWidget()
        curriculum_metrics_widget.setMinimumHeight(55)
        bottom_row.addWidget(curriculum_metrics_widget, stretch=1)
        
        monitor_layout.addLayout(bottom_row)
        
        # ... rest of initialization ...
```

---

## 🎯 Styling Guidelines (QSS)

### Color Palette
```python
BACKGROUND = "#1e1e1e"       # Main background
PANEL_BACKGROUND = "#2d2d2d" # Widget panels
TEXT_PRIMARY = "#ffffff"     # Primary text
TEXT_SECONDARY = "#888888"   # Secondary text
BORDER = "#444444"           # Borders

# Status colors
SUCCESS = "#32CD32"          # Green (OK)
WARNING = "#FFA500"          # Orange (LOW/HIGH threshold)
DANGER = "#FF4444"           # Red (Critical)
OPTIMAL = "#00CED1"          # Turquoise (Optimal state)

# Accent colors
KL_OPTIMAL = "#87CEEB"       # Light blue for KL metrics
CURRICULUM_ACTION = "#87CEEB"# Building action highlight
```

### Font Sizes
```python
STATUS_TEXT = 12             # Widget status labels
INFO_TEXT = 10               # Informational text
VALUE_TEXT = 11              # Numeric values
SMALL_TEXT = 9               # Fine print, thresholds
```

---

## ✅ Testing Checklist

Before deployment, verify:
- [ ] All widgets update in real-time (no freezes)
- [ ] Color coding matches thresholds correctly
- [ ] Progress bars fill to 100% at expected stages
- [ ] Loop detection shows warnings at threshold
- [ ] Return statistics histogram displays data
- [ ] Command handling works without blocking UI thread
- [ ] Headless mode runs without errors
