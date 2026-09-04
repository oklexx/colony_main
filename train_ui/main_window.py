from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget, QSizePolicy,
)

from train_ui import protocol as P
from train_ui.models import ModelInfo, ModelRegistry
from train_ui.parameter_widget import (
    PARAM_SPECS, REWARD_SPECS, ParamSpec, scale_value, spec_for,
)
from train_ui.kl_status_widget import KLStatusWidget
from train_ui.curriculum_progress_widget import CurriculumProgressWidget
from train_ui.action_loop_widget import ActionLoopWidget
from train_ui.return_statistics_widget import ReturnStatisticsWidget
from train_ui.quick_actions_widget import QuickActionsWidget

CONFIG_VERSION = 5

CONFIG_PATH = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    / "sakhalin_colony_ui" / "config.json"
)

LEVEL_COLORS = {"info": "#D4D4D4", "warn": "#CE9178", "error": "#F44747"}

DEFAULT_PARAMS: Dict[str, Any] = {
    "name": "run_001",
    "use_amp": False,
    "torch_compile": False,
    **{s.key: s.default for s in PARAM_SPECS},
    **{s.key: s.default for s in REWARD_SPECS},
}

STYLE_SMALL = "font-size:10px;"
STYLE_BTN_SM = "font-size:10px; padding:3px 6px; min-height:18px;"
STYLE_TITLE = "font-weight:bold; font-size:10px;"


class ParameterRow(QWidget):
    """Compact parameter: label + spinbox + ×2/÷2."""

    value_changed = Signal(str, object)

    def __init__(self, spec: ParamSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName(f"param_{spec.key}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.label = QLabel(spec.label)
        self.label.setObjectName(f"label_{spec.key}")
        self.label.setToolTip(spec.tooltip)
        self.label.setFixedWidth(90)
        self.label.setStyleSheet(STYLE_SMALL)

        if spec.is_int:
            self.spin = QSpinBox()
            self.spin.setRange(int(spec.min), int(spec.max))
            self.spin.setSingleStep(int(spec.step) or 1)
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(spec.min, spec.max)
            if spec.min > 0:
                decimals = max(
                    0, min(8, int(math.ceil(-math.log10(spec.min))) + 1))
            else:
                decimals = 6
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep(spec.step)

        self.spin.setObjectName(f"spin_{spec.key}")
        self.spin.setToolTip(spec.tooltip)
        self.spin.setValue(float(spec.default))
        self.spin.setFixedWidth(80)
        self.spin.setStyleSheet(STYLE_SMALL)
        self.spin.setGroupSeparatorShown(True)
        self.spin.valueChanged.connect(self._on_changed)

        self.btn_x2 = QPushButton("×2")
        self.btn_x2.setFixedSize(18, 14)
        self.btn_x2.setStyleSheet("font-size:8px; padding:0;")
        self.btn_x2.clicked.connect(lambda: self._scale(2))

        self.btn_d2 = QPushButton("÷2")
        self.btn_d2.setFixedSize(18, 14)
        self.btn_d2.setStyleSheet("font-size:8px; padding:0;")
        self.btn_d2.clicked.connect(lambda: self._scale(0.5))

        layout.addWidget(self.label)
        layout.addWidget(self.spin)
        layout.addWidget(self.btn_x2)
        layout.addWidget(self.btn_d2)

    def _on_changed(self, _v):
        self.value_changed.emit(self.spec.key, self.value())

    def _scale(self, factor: float):
        new = scale_value(self.value(), factor, is_int=self.spec.is_int)
        new = max(self.spec.min, min(self.spec.max, float(new)))
        self.spin.blockSignals(True)
        self.spin.setValue(new)
        self.spin.blockSignals(False)
        self.value_changed.emit(self.spec.key, self.value())

    def value(self):
        v = self.spin.value()
        return int(v) if self.spec.is_int else float(v)

    def set_value(self, v):
        self.spin.blockSignals(True)
        self.spin.setValue(float(v))
        self.spin.blockSignals(False)


class MainWindow(QMainWindow):
    train_finished = Signal(int)

    def __init__(self, config: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сахалинская колония — обучение моделей")
        self.setMinimumSize(980, 600)
        self.setObjectName("main_window")

        self.config = config or {}
        self.registry = ModelRegistry()
        self._train_pid: Optional[int] = None
        self._msg_file: Optional[str] = None
        self._msg_offset: int = 0
        self._msg_timer: Optional[QTimer] = None
        self._eval_pid: Optional[int] = None
        self._eval_msg: Optional[str] = None
        self._eval_offset: int = 0
        self._eval_timer: Optional[QTimer] = None
        self._eval_model: Optional[ModelInfo] = None
        self._eval_done: bool = False
        self._watch_pid: Optional[int] = None
        self._watch_timer: Optional[QTimer] = None
        self._watch_start_time: float = 0
        self._auto_scroll = True
        self._reward_history: List[float] = []

        self._build_ui()
        self._restore_state()
        self._watch_stage_mode_changed()
        self.refresh_models()

    # ────────────────────── UI ──────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        left = QVBoxLayout()
        left.setSpacing(2)
        right = QVBoxLayout()
        right.setSpacing(2)

        root.addLayout(left, 1)
        root.addLayout(right, 3)

        self._build_left(left)
        self._build_right(right)
        self._build_menubar()

    # ── LEFT COLUMN ──

    def _build_left(self, layout: QVBoxLayout):
        # Models
        m_row = QHBoxLayout()
        m_row.setSpacing(2)
        m_row.addWidget(QLabel("Модель:"), 0)
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        m_row.addWidget(self.model_combo, 1)
        layout.addLayout(m_row)

        # Model name input
        n_row = QHBoxLayout()
        n_row.setSpacing(2)
        n_row.addWidget(QLabel("Имя:"), 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("run_001")
        self.name_edit.setFixedWidth(140)
        self.name_edit.setStyleSheet(STYLE_SMALL)
        n_row.addWidget(self.name_edit, 1)
        n_row.addStretch(1)
        layout.addLayout(n_row)

        # Model buttons
        mb = QHBoxLayout()
        mb.setSpacing(2)
        self.btn_delete = QPushButton("Удалить")
        self.btn_delete.setStyleSheet(STYLE_BTN_SM)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_watch = QPushButton("Наблюдать")
        self.btn_watch.setStyleSheet(STYLE_BTN_SM)
        self.btn_watch.clicked.connect(self._watch_selected)
        self.btn_resume = QPushButton("Дообучить")
        self.btn_resume.setStyleSheet(STYLE_BTN_SM)
        self.btn_resume.clicked.connect(self._resume_training)
        mb.addWidget(self.btn_delete)
        mb.addWidget(self.btn_watch)
        mb.addWidget(self.btn_resume)
        layout.addLayout(mb)

        # Stats
        self._stat_labels: Dict[str, QLabel] = {}
        sg = QGridLayout()
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setSpacing(0)
        for i, (k, t) in enumerate([
            ("steps", "Шаги"), ("best_reward", "Лучшая"),
            ("episodes", "Эпизоды"), ("train_time", "Время"),
            ("eval_days", "Дни eval"), ("eval_people", "Люди eval"),
        ]):
            r, c = divmod(i, 2)
            nl = QLabel(t)
            nl.setStyleSheet("font-size:10px; color:#aaa;")
            vl = QLabel("—")
            vl.setObjectName(f"stat_{k}")
            vl.setStyleSheet("font-size:11px;")
            sg.addWidget(nl, r * 2, c)
            sg.addWidget(vl, r * 2 + 1, c)
            self._stat_labels[k] = vl
        layout.addLayout(sg)

        # Observation
        self._obs_labels: Dict[str, QLabel] = {}
        og = QGridLayout()
        og.setContentsMargins(0, 0, 0, 0)
        og.setSpacing(0)
        for i, (k, t) in enumerate([
            ("day", "День"), ("action", "Действие"),
            ("reward", "Награда"), ("people", "Люди"),
            ("money", "Деньги"),
        ]):
            r, c = divmod(i, 2)
            nl = QLabel(t)
            nl.setStyleSheet("font-size:10px; color:#aaa;")
            vl = QLabel("—")
            vl.setObjectName(f"obs_{k}")
            vl.setStyleSheet("font-size:11px;")
            og.addWidget(nl, r * 2, c)
            og.addWidget(vl, r * 2 + 1, c)
            self._obs_labels[k] = vl
        layout.addLayout(og)

        # Parameters (1-column)
        pg = QGridLayout()
        pg.setSpacing(2)
        pg.setContentsMargins(0, 0, 0, 0)
        self.param_rows: Dict[str, ParameterRow] = {}
        for i, spec in enumerate(PARAM_SPECS):
            row = ParameterRow(spec)
            row.value_changed.connect(lambda _k, _v: self._on_param_changed())
            self.param_rows[spec.key] = row
            pg.addWidget(row, i, 0)
        layout.addLayout(pg)

    # ── RIGHT COLUMN ──

    def _build_right(self, layout: QVBoxLayout):
        # ── Top row: Curriculum+Watch (left) ‖ Rewards (right) ──
        top_h = QHBoxLayout()
        top_h.setSpacing(6)

        # --- Left: Autocurriculum + Watch stage stacked vertically ---
        left_v = QVBoxLayout()
        left_v.setSpacing(4)
        left_v.setContentsMargins(0, 0, 0, 0)

        cur_box = QGroupBox("Автокурикулум (расписание этапов)")
        cur_box.setStyleSheet("font-weight:bold; font-size:10px;")
        cur_lay = QVBoxLayout(cur_box)
        cur_lay.setSpacing(3)
        cur_lay.setContentsMargins(4, 8, 4, 4)

        cur_prof = QHBoxLayout()
        cur_prof.setSpacing(4)
        cur_prof.addWidget(QLabel("Профиль:"))
        self.cur_profile_combo = QComboBox()
        self.cur_profile_combo.addItems(
            ["Отключён", "Стандартный", "Быстрый", "Медленный"])
        self.cur_profile_combo.setCurrentIndex(1)
        self.cur_profile_combo.setFixedWidth(95)
        self.cur_profile_combo.setStyleSheet(STYLE_SMALL)
        self.cur_profile_combo.currentIndexChanged.connect(
            self._cur_profile_changed)
        cur_prof.addWidget(self.cur_profile_combo)
        cur_prof.addStretch(1)
        cur_lay.addLayout(cur_prof)

        self.curriculum_table = QTableWidget(0, 2)
        self.curriculum_table.setObjectName("curriculum_table")
        self.curriculum_table.setHorizontalHeaderLabels(
            ["Порог шагов", "Этап (1-3)"])
        self.curriculum_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.curriculum_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.curriculum_table.setMinimumHeight(100)
        self.curriculum_table.setMaximumHeight(130)
        self.curriculum_table.setStyleSheet("font-size:10px;")
        cur_lay.addWidget(self.curriculum_table)

        cur_btns = QHBoxLayout()
        cur_btns.setSpacing(4)
        self.btn_cur_add = QPushButton("Добавить")
        self.btn_cur_add.setStyleSheet(STYLE_BTN_SM)
        self.btn_cur_add.clicked.connect(self._cur_add_row)
        self.btn_cur_del = QPushButton("Удалить")
        self.btn_cur_del.setStyleSheet(STYLE_BTN_SM)
        self.btn_cur_del.clicked.connect(self._cur_del_row)
        self.btn_cur_clear = QPushButton("Очистить")
        self.btn_cur_clear.setStyleSheet(STYLE_BTN_SM)
        self.btn_cur_clear.clicked.connect(self._cur_clear)
        self.btn_cur_save = QPushButton("Сохранить")
        self.btn_cur_save.setStyleSheet(STYLE_BTN_SM)
        self.btn_cur_save.clicked.connect(self._cur_save)
        self.btn_cur_load = QPushButton("Загрузить")
        self.btn_cur_load.setStyleSheet(STYLE_BTN_SM)
        self.btn_cur_load.clicked.connect(self._cur_load)
        cur_btns.addWidget(self.btn_cur_add)
        cur_btns.addWidget(self.btn_cur_del)
        cur_btns.addWidget(self.btn_cur_clear)
        cur_btns.addWidget(self.btn_cur_save)
        cur_btns.addWidget(self.btn_cur_load)
        cur_lay.addLayout(cur_btns)

        left_v.addWidget(cur_box)

        # Watch stage block (same width, below)
        watch_box = QGroupBox("Этап курикулума")
        watch_box.setStyleSheet("font-weight:bold; font-size:10px;")
        watch_lay = QVBoxLayout(watch_box)
        watch_lay.setSpacing(4)
        watch_lay.setContentsMargins(4, 8, 4, 4)

        self.watch_use_model_stage_chk = QCheckBox(
            "Использовать stage модели")
        self.watch_use_model_stage_chk.setStyleSheet(STYLE_SMALL)
        self.watch_use_model_stage_chk.setChecked(True)
        self.watch_use_model_stage_chk.toggled.connect(
            self._watch_stage_mode_changed)
        watch_lay.addWidget(self.watch_use_model_stage_chk)

        self.watch_override_stage_chk = QCheckBox("Override stage")
        self.watch_override_stage_chk.setStyleSheet(STYLE_SMALL)
        self.watch_override_stage_chk.setChecked(False)
        self.watch_override_stage_chk.setEnabled(False)
        self.watch_override_stage_chk.toggled.connect(
            self._watch_stage_mode_changed)
        watch_lay.addWidget(self.watch_override_stage_chk)

        watch_lay.addWidget(QLabel("Stage:"))
        self.stage_combo = QComboBox()
        self.stage_combo.addItems([
            "0 — все здания", "1 — базовые",
            "2 — +средние", "3 — полные"])
        self.stage_combo.setCurrentIndex(2)
        self.stage_combo.setFixedWidth(110)
        self.stage_combo.setStyleSheet(STYLE_SMALL)
        watch_lay.addWidget(self.stage_combo)

        self.watch_stage_info_label = QLabel(
            "Stage из meta.json модели.")
        self.watch_stage_info_label.setStyleSheet(
            "font-size:9px; color:#808080;")
        self.watch_stage_info_label.setWordWrap(True)
        watch_lay.addWidget(self.watch_stage_info_label)

        left_v.addWidget(watch_box)
        top_h.addLayout(left_v, 3)

        # --- Right: Rewards single column ---
        rw_box = QVBoxLayout()
        rw_box.setSpacing(0)
        rw_box.setContentsMargins(0, 0, 0, 0)
        rw_box.addWidget(QLabel("Награды"), 0)
        rw_v = QVBoxLayout()
        rw_v.setSpacing(2)
        self.reward_rows: Dict[str, ParameterRow] = {}
        for spec in REWARD_SPECS:
            row = ParameterRow(spec)
            row.value_changed.connect(lambda _k, _v: self._on_param_changed())
            self.reward_rows[spec.key] = row
            rw_v.addWidget(row)
        rw_box.addLayout(rw_v)
        rw_wrap = QWidget()
        rw_wrap.setLayout(rw_box)
        rw_wrap.setMaximumWidth(250)
        top_h.addWidget(rw_wrap, 1)

        layout.addLayout(top_h)

        # ── Stage + Progress ──
        r_prog = QHBoxLayout()
        r_prog.setSpacing(3)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ожидание…")
        self.progress_bar.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setFixedHeight(16)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size:9px; color:#aaa;")
        self.status_label.setFixedWidth(180)
        r_prog.addWidget(self.progress_bar, 1)
        r_prog.addWidget(self.status_label, 0)
        layout.addLayout(r_prog)

        # ── AMP / compile + Buttons ──
        r3 = QHBoxLayout()
        r3.setSpacing(3)

        self.chk_amp = QCheckBox("AMP (bfloat16)")
        self.chk_amp.setStyleSheet(STYLE_SMALL)
        self.chk_amp.setChecked(bool(self.config.get("use_amp", False)))
        self.chk_compile = QCheckBox("torch.compile")
        self.chk_compile.setStyleSheet(STYLE_SMALL)
        self.chk_compile.setChecked(
            bool(self.config.get("torch_compile", False)))

        self.btn_reset = QPushButton("Сбросить")
        self.btn_reset.setStyleSheet(STYLE_BTN_SM)
        self.btn_reset.clicked.connect(self._reset_params)
        self.btn_start = QPushButton("Запустить обучение")
        self.btn_start.setStyleSheet(STYLE_BTN_SM)
        self.btn_start.clicked.connect(lambda: self._start_training())
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setStyleSheet(STYLE_BTN_SM)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_reset_default = QPushButton("По умолчанию")
        self.btn_reset_default.setStyleSheet(STYLE_BTN_SM)
        self.btn_reset_default.clicked.connect(self._reset_to_defaults)

        r3.addWidget(self.chk_amp, 0)
        r3.addWidget(self.chk_compile, 0)
        r3.addWidget(self.btn_reset, 0)
        r3.addWidget(self.btn_start, 1)
        r3.addWidget(self.btn_stop, 0)
        r3.addWidget(self.btn_reset_default, 0)
        r3.addWidget(self.status_label, 0)
        layout.addLayout(r3)

        # ── Console ──
        ct = QHBoxLayout()
        ct.setSpacing(2)
        self.btn_copy_log = QPushButton("Копировать")
        self.btn_copy_log.setStyleSheet(STYLE_BTN_SM)
        self.btn_copy_log.clicked.connect(self._copy_log)
        self.btn_clear_log = QPushButton("Очистить")
        self.btn_clear_log.setStyleSheet(STYLE_BTN_SM)
        self.btn_clear_log.clicked.connect(self._clear_log)
        ct.addStretch(1)
        ct.addWidget(self.btn_copy_log)
        ct.addWidget(self.btn_clear_log)
        layout.addLayout(ct)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setObjectName("console")
        self.console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.console.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace; font-size:10px;")
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.console.verticalScrollBar().rangeChanged.connect(self._on_scroll_range)
        self.console.verticalScrollBar().valueChanged.connect(self._on_scroll_value)
        layout.addWidget(self.console, 1)

        # ── NEW WIDGETS PANEL (UI Improvements) ──
        new_panel = QWidget()
        new_panel.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 2px solid #4a90d9;
                border-radius: 8px;
                padding: 8px;
                margin: 5px;
            }
            QLabel { color: #ffffff; font-size: 9px; }
            QProgressBar { 
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                text-align: center;
            }
        """)
        new_layout = QVBoxLayout(new_panel)
        new_layout.setSpacing(2)
        
        # KL Status Widget
        self.kl_status_widget = KLStatusWidget()
        new_layout.addWidget(self.kl_status_widget, 1)
        
        # Curriculum Progress Widget
        self.curriculum_progress_widget = CurriculumProgressWidget()
        new_layout.addWidget(self.curriculum_progress_widget, 0)
        
        # Action Loop Widget
        self.action_loop_widget = ActionLoopWidget()
        new_layout.addWidget(self.action_loop_widget, 0)
        
        # Return Statistics Widget
        self.return_statistics_widget = ReturnStatisticsWidget()
        new_layout.addWidget(self.return_statistics_widget, 1)
        
        # Quick Actions Widget
        self.quick_actions_widget = QuickActionsWidget()
        new_layout.addWidget(self.quick_actions_widget, 0)

        layout.insertWidget(1, new_panel)  # Insert below parameter panel

    # ── menubar ──

    def _build_menubar(self):
        mb = self.menuBar()
        mm = mb.addMenu("Модель")
        mm.addAction("Оценить").triggered.connect(self._eval_selected)
        mm.addAction("Наблюдать").triggered.connect(self._watch_selected)
        mm.addAction("Дообучить").triggered.connect(self._resume_training)
        mm.addSeparator()
        mm.addAction("Удалить").triggered.connect(self._delete_selected)
        mm.addAction("Обновить").triggered.connect(self.refresh_models)

    # ────────────────────── logging ──────────────────────

    def log(self, level: str, message: str):
        color = LEVEL_COLORS.get(level, "#D4D4D4")
        self.console.appendHtml(
            f'<span style="color:{color};">{message}</span>')
        if self._auto_scroll:
            sb = self.console.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_scroll_range(self, _lo, _hi):
        sb = self.console.verticalScrollBar()
        if sb.value() >= sb.maximum() - 4:
            self._auto_scroll = True

    def _on_scroll_value(self, v):
        sb = self.console.verticalScrollBar()
        self._auto_scroll = v >= sb.maximum() - 4

    def _copy_log(self):
        text = self.console.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.log("info", "Скопировано")

    def _clear_log(self):
        self.console.clear()

    # ────────────────────── models ──────────────────────

    def refresh_models(self):
        prev = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("— нет моделей —")
        for m in self.registry.scan():
            self.model_combo.addItem(m.name)
        if prev and self.model_combo.findText(prev) >= 0:
            self.model_combo.setCurrentText(prev)
        self.model_combo.blockSignals(False)
        self._on_model_changed()

    def _on_model_changed(self):
        self._update_stats()

    def _selected_model(self) -> Optional[ModelInfo]:
        name = self.model_combo.currentText()
        if name in ("— нет моделей —", ""):
            return None
        return self.registry.get(name)

    def _update_stats(self):
        m = self._selected_model()
        if m is None:
            for lbl in self._stat_labels.values():
                lbl.setText("—")
            return
        self._stat_labels["steps"].setText(f"{m.steps:,}")
        self._stat_labels["best_reward"].setText(f"{m.best_reward:.2f}")
        self._stat_labels["episodes"].setText(str(m.episodes))
        self._stat_labels["train_time"].setText(f"{m.train_time_sec:.0f}s")
        ev = m.eval or {}
        self._stat_labels["eval_days"].setText(
            f"{ev.get('days', 0):.1f}" if ev else "—")
        self._stat_labels["eval_people"].setText(
            f"{ev.get('people', 0):.1f}" if ev else "—")

    def _update_obs(self, data: Dict[str, Any]):
        for key in ("day", "action", "reward", "people", "money"):
            lbl = self._obs_labels.get(key)
            if lbl is None:
                continue
            v = data.get(key)
            if v is None:
                lbl.setText("—")
            elif key == "reward":
                lbl.setText(f"{v:+.2f}")
            else:
                lbl.setText(str(v))

    # ────────────────────── eval / watch / resume ──────────────────────

    def _delete_selected(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Удаление", "Выберите модель")
            return
        msg = QMessageBox()
        msg.setWindowTitle("Подтверждение")
        msg.setText(f"Удалить «{m.name}»?")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            self.registry.delete(m.name)
            self.log("info", f"Удалена: {m.name}")
        except Exception as e:
            self.log("error", f"Ошибка: {e}")
        self.refresh_models()

    def _eval_selected(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Оценка", "Выберите модель")
            return
        import sys as _sys, tempfile
        eval_msg = os.path.join(
            tempfile.gettempdir(),
            f"colony_eval_{int(time.time() * 1000)}.jsonl")
        if os.path.exists(eval_msg):
            os.unlink(eval_msg)
        import subprocess as _sp
        proc = _sp.Popen(
            [_sys.executable, "-u",
             str(Path(__file__).resolve().parent / "worker.py"),
             "--eval-model", str(m.model_file),
             "--episodes", "5", "--max-days", "1000",
             "--device", "cpu", "--output", eval_msg],
            cwd=str(Path(__file__).resolve().parent.parent),
            creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        self._eval_pid = proc.pid
        self._eval_msg = eval_msg
        self._eval_offset = 0
        self._eval_model = m
        self._eval_done = False
        self._eval_timer = QTimer(self)
        self._eval_timer.timeout.connect(self._poll_eval)
        self._eval_timer.start(250)
        self.log("info", f"Eval: {m.name} (pid={self._eval_pid})")

    def _read_eval_file(self):
        if self._eval_msg is None:
            return
        try:
            size = os.path.getsize(self._eval_msg)
        except OSError:
            return
        if size <= self._eval_offset:
            return
        try:
            with open(self._eval_msg, "r", encoding="utf-8") as f:
                f.seek(self._eval_offset)
                data = f.read()
                self._eval_offset = f.tell()
        except (OSError, UnicodeDecodeError):
            return
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            if t == "eval_result":
                m = self._eval_model
                if m:
                    self.registry.save_eval(m.name, {
                        "days": d.get("days", 0),
                        "people": d.get("people", 0),
                        "bases": d.get("bases", 0),
                        "avg_return": d.get("avg_return", 0),
                    })
                    self.log("info",
                             f"Eval {m.name}: days={d.get('days', 0):.1f}")
                    self._update_stats()
                    self._eval_done = True
            elif t == "log":
                self.log(d.get("level", "info"), d.get("message", ""))
            elif t == "error":
                self.log("error", d.get("message", ""))

    def _poll_eval(self):
        self._read_eval_file()
        if self._eval_pid and not self._poll_pid_alive(self._eval_pid):
            self._read_eval_file()
            if not self._eval_done:
                self.log("error", "Eval завершился с ошибкой")
            self._cleanup_eval()
            self.refresh_models()

    def _cleanup_eval(self):
        if self._eval_timer:
            self._eval_timer.stop()
            self._eval_timer = None
        if self._eval_msg:
            try:
                os.unlink(self._eval_msg)
            except OSError:
                pass
            self._eval_msg = None
        self._eval_pid = None
        self._eval_model = None
        self._eval_done = False

    def _watch_selected(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Наблюдение", "Выберите модель")
            return
        if self._watch_pid:
            self._stop_watch()
            return
        import sys as _sys, tempfile, subprocess as _sp
        watch_msg = os.path.join(
            tempfile.gettempdir(),
            f"colony_watch_{int(time.time() * 1000)}.log")
        args = [_sys.executable, "-u",
                str(Path(__file__).resolve().parent.parent / "watch_champion.py"),
                "--model-dir", str(m.model_file.parent),
                "--episodes", "1000000",
                "--max-steps", "1000000",
                "--speed", "5", "--device", "cpu",
                "--log-file", watch_msg,
                "--visual"]
        proc = _sp.Popen(
            args,
            cwd=str(Path(__file__).resolve().parent.parent),
            creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        self._watch_pid = proc.pid
        self._watch_start_time = time.time()
        self._watch_log = watch_msg
        self._watch_log_offset = 0
        self._reward_history = []
        for lbl in self._obs_labels.values():
            lbl.setText("—")
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._poll_watch)
        self._watch_timer.start(500)
        self.log("info", f"Наблюдение: {m.name} (pid={self._watch_pid})")

    def _poll_watch(self):
        self._read_watch_log()
        if self._watch_pid and time.time() - self._watch_start_time > 2.0:
            if not self._poll_pid_alive(self._watch_pid):
                self._read_watch_log()
                self._cleanup_watch()
                self.log("info", "Наблюдение завершено")

    def _read_watch_log(self):
        if not getattr(self, "_watch_log", None):
            return
        try:
            size = os.path.getsize(self._watch_log)
        except OSError:
            return
        if size <= self._watch_log_offset:
            return
        try:
            with open(self._watch_log, "r", encoding="utf-8") as f:
                f.seek(self._watch_log_offset)
                data = f.read()
                self._watch_log_offset = f.tell()
        except (OSError, UnicodeDecodeError):
            return
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                    if d.get("type") == "step":
                        self._update_obs(d)
                        self._reward_history.append(d.get("reward", 0.0))
                        continue
                    elif d.get("type") == "log":
                        self.log("info", d.get("message", ""))
                        continue
                    elif d.get("type") == "error":
                        self.log("error", d.get("message", ""))
                        continue
                    elif d.get("type") == "done":
                        self.log("info", d.get("message", "Завершено"))
                        continue
                except json.JSONDecodeError:
                    pass
            self.log("info", line)

    def _stop_watch(self):
        if not self._watch_pid:
            return
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self._watch_pid)],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        self._cleanup_watch()
        self.log("info", "Наблюдение остановлено")

    def _cleanup_watch(self):
        if self._watch_timer:
            self._watch_timer.stop()
            self._watch_timer = None
        if getattr(self, "_watch_log", None):
            try:
                os.unlink(self._watch_log)
            except OSError:
                pass
            self._watch_log = None
        self._watch_pid = None

    def _resume_training(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Дообучение", "Выберите модель")
            return
        if not m.model_file.exists():
            self.log("error", f"Файл не найден: {m.model_file}")
            return
        self.log("info", f"Дообучение: {m.name}")
        self._start_training(resume_model=m.model_file)

    def _poll_pid_alive(self, pid: int) -> bool:
        if not pid:
            return False
        try:
            import subprocess as _sp
            _sp.run(["tasklist", "/FI", f"PID eq {pid}"],
                    capture_output=True, timeout=5)
        except Exception:
            return False
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x00100000, 0, pid)
        if not h:
            return False
        r = k.WaitForSingleObject(h, 0)
        k.CloseHandle(h)
        return r != 0x0

    # ────────────────────── training ──────────────────────

    def _on_param_changed(self):
        pass

    def _randomize_seed(self):
        spec = spec_for("seed")
        self.param_rows["seed"].set_value(
            random.randint(int(spec.min), int(spec.max)))

    def _next_run_name(self) -> str:
        models = self.registry.scan()
        max_num = 0
        for m in models:
            if m.name.startswith("run_"):
                try:
                    num = int(m.name.split("_", 1)[1].split("_")[0])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    pass
        return f"run_{max_num + 1:03d}"

    def _collect_config(self) -> Dict[str, Any]:
        name = self.name_edit.text().strip()
        if not name:
            name = self._next_run_name()
        cfg: Dict[str, Any] = {
            "name": name,
            "use_amp": self.chk_amp.isChecked(),
            "torch_compile": self.chk_compile.isChecked(),
        }
        for k, r in self.param_rows.items():
            cfg[k] = r.value()
        for k, r in self.reward_rows.items():
            cfg[k] = r.value()
        net = int(cfg.pop("net_arch", 256))
        cfg["net_arch"] = [net, net]
        cur = self._get_curriculum_data()
        if cur:
            ok, msg = self._validate_curriculum(cur)
            if not ok:
                QMessageBox.warning(self, "Курикулум", msg)
                cur = []
        cfg["curriculum_schedule"] = cur
        cfg["curriculum_stage"] = self.stage_combo.currentIndex()
        return cfg

    def _start_training(self, resume_model: Optional[Path] = None):
        if self._train_pid:
            self.log("warn", "Уже запущено")
            return
        self._randomize_seed()
        cfg = self._collect_config()
        if resume_model:
            cfg["name"] = cfg["name"] + "_ft"
        existing = self.registry.get(cfg["name"])
        if existing and not resume_model:
            ret = QMessageBox.question(
                self, "Дубликат имени",
                f"Модель «{cfg['name']}» уже существует.\nПерезаписать?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(cfg, tmp, ensure_ascii=False)
        tmp.close()
        msg_file = os.path.join(
            tempfile.gettempdir(),
            f"colony_ui_{int(time.time() * 1000)}.jsonl")
        import sys as _sys, subprocess as _sp
        proc = _sp.Popen(
            [_sys.executable, "-u",
             str(Path(__file__).resolve().parent / "worker.py"),
             "--config", tmp.name, "--name", cfg["name"],
             "--output", msg_file],
            cwd=str(Path(__file__).resolve().parent.parent),
            creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        self._train_pid = proc.pid
        self._msg_file = msg_file
        self._msg_offset = 0
        self._msg_timer = QTimer(self)
        self._msg_timer.timeout.connect(self._poll_training)
        self._msg_timer.start(250)
        self._set_training_ui(True)
        self.progress_bar.setValue(0)
        self.log("info",
                 f"Запуск: {cfg['name']} steps={cfg['total_timesteps']:,}")

    def _poll_training(self):
        self._poll_messages()
        if self._train_pid and not self._poll_pid_alive(self._train_pid):
            self._poll_messages()
            self._on_train_finished(0)

    def _poll_messages(self):
        if not self._msg_file:
            return
        try:
            size = os.path.getsize(self._msg_file)
        except OSError:
            return
        if size <= self._msg_offset:
            return
        try:
            with open(self._msg_file, "r", encoding="utf-8") as f:
                f.seek(self._msg_offset)
                data = f.read()
                self._msg_offset = f.tell()
        except (OSError, UnicodeDecodeError):
            return
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = P.decode(line)
            except ValueError:
                continue
            if isinstance(msg, P.ReadyMsg):
                self.log("info", "Процесс готов")
            elif isinstance(msg, P.LogMsg):
                self.log(msg.level, msg.message)
            elif isinstance(msg, P.ProgressMsg):
                self._update_progress(msg)
            elif isinstance(msg, P.SavedMsg):
                self.log("info", f"Чекпоинт: {msg.path}")
            elif isinstance(msg, P.DoneMsg):
                self.log("info",
                         f"Завершено: {msg.total:,} шагов, "
                         f"best={msg.best_reward:.2f}")
            elif isinstance(msg, P.ErrorMsg):
                self.log("error", msg.message)

    def _update_progress(self, m: P.ProgressMsg):
        total = max(m.total, 1)
        self.progress_bar.setValue(min(1000, int(1000 * m.done / total)))
        self.progress_bar.setFormat(f"{m.done:,}/{m.total:,}")
        fps = m.fps if m.fps > 0 else 0.0
        eta = ""
        if fps > 0 and m.done < m.total:
            eta = f" ~{int((m.total - m.done) / fps // 60)}мин"
        self.status_label.setText(
            f"FPS {fps:.0f} best {m.best_reward:.1f} ep{m.episodes}{eta}")
        
        # ── UPDATE NEW WIDGETS ──
        # KL Status Widget
        if m.kl is not None:
            self.kl_status_widget.update(
                kl=float(m.kl),
                ent_coef=m.ent_coef if hasattr(m, 'ent_coef') else 0.005
            )
        
        # Curriculum Progress Widget
        if hasattr(m, 'stage') and m.stage is not None:
            stage_info = {
                'stage': int(m.stage) if m.stage >= 0 else 0,
                'progress_percent': getattr(m, 'progress_percent', 50.0),
                'available_actions': getattr(m, 'available_actions', ''),
                'next_stage_at_step': getattr(m, 'next_stage_at_step', None),
                'upcoming_stages': getattr(m, 'upcoming_stages', []),
            }
            self.curriculum_progress_widget.update(stage_info)
        
        # Action Loop Widget
        if hasattr(m, 'loop_detected') and m.loop_detected:
            self.action_loop_widget.set_loop_status(
                loop_detected=True,
                action_name=getattr(m, 'loop_action_name', 'UNKNOWN'),
                consecutive_count=10,
                threshold=3,
                envs_with_loops=getattr(m, 'envs_with_loops', 0),
                total_envs=8
            )
        else:
            self.action_loop_widget.set_loop_status(
                loop_detected=False
            )
        
        # Return Statistics Widget (if returns data available)
        if hasattr(m, 'returns') and m.returns:
            self.return_statistics_widget.update_statistics(returns=list(m.returns))
        
        # Dashboard Widget (update every 100 steps to avoid overdraw)
        if m.done % 100 == 0:
            top_actions = getattr(m, 'top_actions', {})
            self.dashboard.update_data(
                kl=float(m.kl) if hasattr(m, 'kl') and m.kl is not None else None,
                entropy=float(m.entropy) if hasattr(m, 'entropy') and m.entropy is not None else None,
                top_actions=top_actions if isinstance(top_actions, dict) else {}
            )

    def _stop_training(self):
        if not self._train_pid:
            return
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(self._train_pid)],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        self._cleanup_training()
        self._set_training_ui(False)
        self.status_label.setText("Остановлено")

    def _on_train_finished(self, code: int = 0):
        self._cleanup_training()
        self._set_training_ui(False)
        if code == 0:
            self.progress_bar.setValue(1000)
            self.status_label.setText("Завершено")
        else:
            self.log("error", f"Код {code}")
        self.refresh_models()
        self.train_finished.emit(code)

    def _cleanup_training(self):
        if self._msg_timer:
            self._msg_timer.stop()
            self._msg_timer = None
        if self._msg_file:
            try:
                os.unlink(self._msg_file)
            except OSError:
                pass
            self._msg_file = None
        self._train_pid = None

    def _set_training_ui(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if not running:
            self.progress_bar.setFormat("Ожидание…")
            self.progress_bar.setValue(0)

    # ────────────────────── params ──────────────────────

    def _reset_params(self):
        for k, r in self.param_rows.items():
            r.set_value(DEFAULT_PARAMS[k])
        for k, r in self.reward_rows.items():
            r.set_value(DEFAULT_PARAMS[k])
        self.log("info", "Сброшено")

    def _reset_to_defaults(self):
        self._reset_params()
        self.stage_combo.setCurrentIndex(2)
        self.chk_amp.setChecked(False)
        self.chk_compile.setChecked(False)
        self._cur_clear()
        self._load_data_to_table([[0, 1], [5_000_000, 2], [15_000_000, 3]])
        self.log("info", "По умолчанию")

    # ────────────────────── curriculum ──────────────────────

    CURRICULUM_PROFILES: Dict[str, List[List[int]]] = {
        "Отключён": [],
        "Стандартный": [[0, 1], [5_000_000, 2], [15_000_000, 3]],
        "Быстрый": [[0, 2], [3_000_000, 3]],
        "Медленный": [[0, 1], [10_000_000, 2], [30_000_000, 3]],
    }

    def _table_to_data(self) -> List[List[int]]:
        data: List[List[int]] = []
        for r in range(self.curriculum_table.rowCount()):
            try:
                th = int(self.curriculum_table.item(r, 0).text())
                st = int(self.curriculum_table.item(r, 1).text())
                data.append([th, st])
            except (AttributeError, ValueError):
                continue
        return data

    def _load_data_to_table(self, data: List[List[int]]):
        self.curriculum_table.setRowCount(0)
        for th, st in sorted(data, key=lambda x: x[0]):
            row = self.curriculum_table.rowCount()
            self.curriculum_table.insertRow(row)
            self.curriculum_table.setItem(row, 0, QTableWidgetItem(str(th)))
            self.curriculum_table.setItem(row, 1, QTableWidgetItem(str(st)))

    def _cur_add_row(self, threshold: int = 0, stage: int = 1):
        data = self._table_to_data()
        data.append([threshold, stage])
        self._load_data_to_table(data)

    def _cur_del_row(self):
        row = self.curriculum_table.currentRow()
        if row >= 0:
            self.curriculum_table.removeRow(row)

    def _cur_clear(self):
        self.curriculum_table.setRowCount(0)

    def _cur_profile_changed(self, _idx: int):
        name = self.cur_profile_combo.currentText()
        profile = self.CURRICULUM_PROFILES.get(name, [])
        self._load_data_to_table(profile)

    def _get_curriculum_data(self) -> List[List[int]]:
        return self._table_to_data()

    def _validate_curriculum(self, data) -> tuple[bool, str]:
        for i in range(1, len(data)):
            if data[i][0] <= data[i - 1][0]:
                return False, f"Порог {data[i][0]:,} ≤ {data[i-1][0]:,}"
        return True, "OK"

    def _cur_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить", "curriculum.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._get_curriculum_data(), f, indent=2)
        self.log("info", f"Сохранено: {path}")

    def _cur_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            parsed = [[int(a), int(b)] for a, b in data
                       if isinstance(a, (int, float)) and isinstance(b, (int, float))]
            self._load_data_to_table(parsed)
            self.log("info", f"Загружено: {path}")
        except Exception as e:
            self.log("error", f"Ошибка: {e}")

    def _watch_stage_mode_changed(self):
        use_model = self.watch_use_model_stage_chk.isChecked()
        override = self.watch_override_stage_chk.isChecked()
        self.watch_override_stage_chk.setEnabled(use_model)
        self.stage_combo.setEnabled(not use_model or override)
        if use_model and not override:
            self.watch_stage_info_label.setText(
                "Stage берётся из best_model.meta.json модели.")
        elif use_model and override:
            self.watch_stage_info_label.setText(
                "Stage: переопределение из списка (модель → override).")
        else:
            self.watch_stage_info_label.setText(
                "Stage: только из списка (модель игнорируется).")

    # ────────────────────── state ──────────────────────

    def save_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "use_amp": self.chk_amp.isChecked(),
            "torch_compile": self.chk_compile.isChecked(),
            "model_name": self.name_edit.text().strip(),
            "config_version": CONFIG_VERSION,
        }
        m = self._selected_model()
        if m:
            state["selected_model"] = m.name
        for k, r in self.param_rows.items():
            state[k] = r.value()
        for k, r in self.reward_rows.items():
            state[k] = r.value()
        state["stage"] = self.stage_combo.currentIndex()
        state["curriculum_schedule"] = self._get_curriculum_data()
        return state

    def _restore_state(self):
        cfg = self.config
        self.chk_amp.setChecked(bool(cfg.get("use_amp", False)))
        self.chk_compile.setChecked(bool(cfg.get("torch_compile", False)))
        saved_name = cfg.get("model_name", "")
        if saved_name:
            self.name_edit.setText(saved_name)
        for k, r in self.param_rows.items():
            v = cfg.get(k, DEFAULT_PARAMS.get(k))
            if isinstance(v, (list, tuple)):
                v = v[0] if v else DEFAULT_PARAMS.get(k, 0)
            if v is not None:
                r.set_value(float(v))
        cfg_ver = cfg.get("config_version", 0)
        for k, r in self.reward_rows.items():
            # Reward params always come from code defaults (REWARD_SPECS).
            # Saved config only overrides if config_version is HIGHER
            # (meaning user explicitly changed them in a newer UI version).
            if cfg_ver > CONFIG_VERSION:
                v = cfg.get(k, DEFAULT_PARAMS.get(k))
            else:
                v = DEFAULT_PARAMS.get(k)
            if isinstance(v, (list, tuple)):
                v = v[0] if v else DEFAULT_PARAMS.get(k, 0)
            if v is not None:
                r.set_value(float(v))
        stage = cfg.get("stage")
        if stage is not None:
            self.stage_combo.setCurrentIndex(int(stage))
        cur = cfg.get("curriculum_schedule")
        if cur and isinstance(cur, list):
            parsed = [[int(a), int(b)] for a, b in cur
                      if isinstance(a, (int, float)) and isinstance(b, (int, float))]
        else:
            parsed = [[0, 1], [5_000_000, 2], [15_000_000, 3]]
        self._load_data_to_table(parsed)
        sel = cfg.get("selected_model")
        if sel:
            idx = self.model_combo.findText(sel)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)


def save_config(state: Dict[str, Any]):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Ошибка сохранения: {e}")


def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
