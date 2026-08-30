from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QProcess, QTimer, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from train_ui import protocol as P
from train_ui.models import ModelInfo, ModelRegistry
from train_ui.parameter_widget import PARAM_SPECS, ParamSpec, scale_value, spec_for

CONFIG_PATH = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "sakhalin_colony_ui" / "config.json"

LEVEL_COLORS = {"info": "#D4D4D4", "warn": "#CE9178", "error": "#F44747"}

DEFAULT_PARAMS: Dict[str, Any] = {
    "name": "run_001",
    "use_amp": False,
    "torch_compile": False,
    **{s.key: s.default for s in PARAM_SPECS},
}


class ParameterRow(QWidget):
    """One parameter: label + spinbox + ×2/÷2 buttons + tooltip."""

    value_changed = Signal(str, object)

    def __init__(self, spec: ParamSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName(f"param_{spec.key}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(4)

        self.label = QLabel(spec.label)
        self.label.setObjectName(f"label_{spec.key}")
        self.label.setToolTip(spec.tooltip)

        if spec.is_int:
            self.spin = QSpinBox()
            self.spin.setRange(int(spec.min), int(spec.max))
            self.spin.setSingleStep(int(spec.step) or 1)
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(spec.min, spec.max)
            if spec.min > 0:
                decimals = max(0, min(8, int(math.ceil(-math.log10(spec.min))) + 1))
            else:
                decimals = 6
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep(spec.step)

        self.spin.setObjectName(f"spin_{spec.key}")
        self.spin.setToolTip(spec.tooltip)
        self.spin.setValue(float(spec.default))
        self.spin.valueChanged.connect(self._on_changed)

        self.btn_x2 = QPushButton("×2")
        self.btn_x2.setObjectName(f"x2_{spec.key}")
        self.btn_x2.setFixedWidth(36)
        self.btn_x2.setToolTip("Умножить значение на 2")
        self.btn_x2.clicked.connect(lambda: self._scale(2))

        self.btn_d2 = QPushButton("÷2")
        self.btn_d2.setObjectName(f"d2_{spec.key}")
        self.btn_d2.setFixedWidth(36)
        self.btn_d2.setToolTip("Разделить значение на 2")
        self.btn_d2.clicked.connect(lambda: self._scale(0.5))

        layout.addWidget(self.label, 1)
        layout.addWidget(self.spin, 2)
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
        self.resize(1280, 800)
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
        self._pending_stop_timer: Optional[QTimer] = None

        self._build_ui()
        self._restore_state()
        self.refresh_models()

    # ---------- UI construction ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top_split = QSplitter(Qt.Horizontal)
        top_split.setObjectName("top_split")

        self._build_left_panel(top_split)
        self._build_center_panel(top_split)
        self._build_right_panel(top_split)

        top_split.setSizes([320, 560, 320])

        bottom = QSplitter(Qt.Vertical)
        bottom.setObjectName("bottom_split")

        progress_box = QGroupBox("Прогресс обучения")
        progress_box.setObjectName("progress_box")
        p_layout = QVBoxLayout(progress_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Ожидание запуска…")
        self.status_label.setObjectName("status_label")
        p_layout.addWidget(self.progress_bar)
        p_layout.addWidget(self.status_label)

        console_box = QGroupBox("Консоль")
        console_box.setObjectName("console_box")
        c_layout = QVBoxLayout(console_box)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.verticalScrollBar().rangeChanged.connect(self._on_scroll_range)
        self.console.verticalScrollBar().valueChanged.connect(self._on_scroll_value)
        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.clicked.connect(self.console.clear)
        self.btn_save_log = QPushButton("Сохранить в файл")
        self.btn_save_log.setObjectName("btn_save_log")
        self.btn_save_log.clicked.connect(self._save_log)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_save_log)
        c_layout.addWidget(self.console)
        c_layout.addLayout(btn_row)

        bottom.addWidget(progress_box)
        bottom.addWidget(console_box)
        bottom.setSizes([120, 280])

        root.addWidget(top_split, 3)
        root.addWidget(bottom, 2)

    def _build_left_panel(self, splitter: QSplitter):
        box = QGroupBox("Сохранённые модели")
        box.setObjectName("models_box")
        layout = QVBoxLayout(box)

        self.model_table = QTableWidget(0, 5)
        self.model_table.setObjectName("model_table")
        self.model_table.setHorizontalHeaderLabels(
            ["", "Имя", "Дата", "Шаги", "Лучшая награда"]
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_table.cellClicked.connect(self._on_model_clicked)

        sel_row = QHBoxLayout()
        self.chk_select_all = QCheckBox("Выбрать все")
        self.chk_select_all.setObjectName("chk_select_all")
        self.chk_select_all.toggled.connect(self._toggle_select_all)
        sel_row.addWidget(self.chk_select_all)
        sel_row.addStretch(1)

        btns = QHBoxLayout()
        self.btn_delete = QPushButton("Удалить выбранные")
        self.btn_delete.setObjectName("btn_delete")
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.setObjectName("btn_refresh")
        self.btn_refresh.clicked.connect(self.refresh_models)
        self.btn_eval = QPushButton("Оценить")
        self.btn_eval.setObjectName("btn_eval")
        self.btn_eval.clicked.connect(self._eval_selected)
        self.btn_watch = QPushButton("Наблюдать")
        self.btn_watch.setObjectName("btn_watch")
        self.btn_watch.setToolTip("Запустить игру с выбранной моделью и показывать ход в консоли")
        self.btn_watch.clicked.connect(self._watch_selected)
        self.btn_resume = QPushButton("Дообучить")
        self.btn_resume.setObjectName("btn_resume")
        self.btn_resume.setToolTip("Запустить обучение с весами выбранной модели (fine-tuning)")
        self.btn_resume.clicked.connect(self._resume_training)
        btns.addWidget(self.btn_delete)
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_eval)
        btns.addWidget(self.btn_watch)
        btns.addWidget(self.btn_resume)

        self.models_hint = QLabel("Запустите обучение, чтобы получить модели")
        self.models_hint.setObjectName("models_hint")

        layout.addWidget(self.model_table, 1)
        layout.addLayout(sel_row)
        layout.addLayout(btns)
        layout.addWidget(self.models_hint)
        splitter.addWidget(box)

    def _build_center_panel(self, splitter: QSplitter):
        box = QGroupBox("Параметры обучения")
        box.setObjectName("params_box")
        layout = QVBoxLayout(box)

        name_row = QHBoxLayout()
        name_lbl = QLabel("Имя запуска:")
        name_lbl.setObjectName("label_name")
        name_lbl.setToolTip("Имя каталога для сохранения модели в ~/colony_runs/models/")
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("name_edit")
        self.name_edit.setText(str(self.config.get("name", DEFAULT_PARAMS["name"])))
        name_row.addWidget(name_lbl)
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.param_rows: Dict[str, ParameterRow] = {}
        for i, spec in enumerate(PARAM_SPECS):
            row = ParameterRow(spec)
            row.value_changed.connect(lambda _k, _v: self._on_param_changed)
            self.param_rows[spec.key] = row
            grid.addWidget(row, i, 0)
            if spec.key == "seed":
                self.chk_random_seed = QCheckBox("🎲 random")
                self.chk_random_seed.setObjectName("chk_random_seed")
                self.chk_random_seed.setToolTip(
                    "При активации сразу генерирует случайный сид. "
                    "При каждом запуске обучения сид будет перегенерирован заново."
                )
                self.chk_random_seed.toggled.connect(self._on_random_seed_toggled)
                grid.addWidget(self.chk_random_seed, i, 1)

        checks = QHBoxLayout()
        self.chk_amp = QCheckBox("AMP (bfloat16)")
        self.chk_amp.setObjectName("chk_amp")
        self.chk_amp.setToolTip("Включить смешанную точность (bfloat16) на GPU")
        self.chk_amp.setChecked(bool(self.config.get("use_amp", DEFAULT_PARAMS["use_amp"])))
        self.chk_compile = QCheckBox("torch.compile")
        self.chk_compile.setObjectName("chk_compile")
        self.chk_compile.setToolTip("Включить компиляцию модели (первый запуск медленнее)")
        self.chk_compile.setChecked(bool(self.config.get("torch_compile", DEFAULT_PARAMS["torch_compile"])))
        checks.addWidget(self.chk_amp)
        checks.addWidget(self.chk_compile)
        checks.addStretch(1)

        layout.addLayout(grid)
        layout.addLayout(checks)

        # --- Curriculum ---
        curr_box = QGroupBox("Автокурикулум (расписание этапов)")
        curr_box.setObjectName("curriculum_box")
        curr_layout = QVBoxLayout(curr_box)

        self.curriculum_table = QTableWidget(0, 2)
        self.curriculum_table.setObjectName("curriculum_table")
        self.curriculum_table.setHorizontalHeaderLabels(["Порог шагов", "Этап (1-3)"])
        self.curriculum_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.curriculum_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.curriculum_table.setMinimumHeight(120)
        self.curriculum_table.setMaximumHeight(200)
        curr_layout.addWidget(self.curriculum_table)

        curr_btns = QHBoxLayout()
        self.btn_cur_add = QPushButton("Добавить")
        self.btn_cur_add.setObjectName("btn_cur_add")
        self.btn_cur_add.clicked.connect(self._cur_add_row)
        self.btn_cur_del = QPushButton("Удалить")
        self.btn_cur_del.setObjectName("btn_cur_del")
        self.btn_cur_del.clicked.connect(self._cur_del_row)
        self.btn_cur_clear = QPushButton("Очистить")
        self.btn_cur_clear.setObjectName("btn_cur_clear")
        self.btn_cur_clear.clicked.connect(self._cur_clear)
        curr_btns.addWidget(self.btn_cur_add)
        curr_btns.addWidget(self.btn_cur_del)
        curr_btns.addWidget(self.btn_cur_clear)
        curr_btns.addStretch(1)
        curr_layout.addLayout(curr_btns)

        curr_profile_row = QHBoxLayout()
        profile_lbl = QLabel("Профиль:")
        profile_lbl.setObjectName("curriculum_profile_label")
        self.curriculum_profile_combo = QComboBox()
        self.curriculum_profile_combo.setObjectName("curriculum_profile_combo")
        self.curriculum_profile_combo.addItems(["Отключён", "Стандартный", "Быстрый", "Медленный"])
        self.curriculum_profile_combo.setCurrentIndex(1)
        self.curriculum_profile_combo.currentIndexChanged.connect(self._cur_profile_changed)
        curr_profile_row.addWidget(profile_lbl)
        curr_profile_row.addWidget(self.curriculum_profile_combo)
        curr_profile_row.addStretch(1)
        curr_layout.addLayout(curr_profile_row)

        curr_save_row = QHBoxLayout()
        self.btn_cur_save = QPushButton("Сохранить")
        self.btn_cur_save.setObjectName("btn_cur_save")
        self.btn_cur_save.setToolTip("Сохранить расписание в JSON-файл")
        self.btn_cur_save.clicked.connect(self._cur_save)
        self.btn_cur_load = QPushButton("Загрузить")
        self.btn_cur_load.setObjectName("btn_cur_load")
        self.btn_cur_load.setToolTip("Загрузить расписание из JSON-файла")
        self.btn_cur_load.clicked.connect(self._cur_load)
        curr_save_row.addWidget(self.btn_cur_save)
        curr_save_row.addWidget(self.btn_cur_load)
        curr_save_row.addStretch(1)
        curr_layout.addLayout(curr_save_row)

        layout.addWidget(curr_box)

        actions = QHBoxLayout()
        self.btn_start = QPushButton("Запустить обучение")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setToolTip("Собрать параметры и запустить обучение в отдельном процессе")
        self.btn_start.clicked.connect(lambda _checked=False: self._start_training())
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Корректно остановить обучение (сохранит текущую модель)")
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_reset = QPushButton("Сбросить значения по умолчанию")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setToolTip("Восстановить значения параметров из конфигурации по умолчанию")
        self.btn_reset.clicked.connect(self._reset_params)
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_stop)
        actions.addWidget(self.btn_reset)
        layout.addLayout(actions)

        splitter.addWidget(box)

    def _build_right_panel(self, splitter: QSplitter):
        box = QGroupBox("Статистика модели")
        box.setObjectName("stats_box")
        layout = QVBoxLayout(box)

        self.stats_placeholder = QLabel("Выберите модель из списка")
        self.stats_placeholder.setObjectName("stats_placeholder")
        self.stats_placeholder.setAlignment(Qt.AlignCenter)

        self.stats_grid = QGridLayout()
        self.stats_grid.setSpacing(4)
        self._stat_labels: Dict[str, QLabel] = {}
        for key, label in [
            ("steps", "Шаги"),
            ("best_reward", "Лучшая награда"),
            ("episodes", "Эпизоды"),
            ("train_time_sec", "Время обучения, с"),
            ("eval_days", "Средние дни (eval)"),
            ("eval_people", "Средние люди (eval)"),
            ("eval_bases", "Средние постройки (eval)"),
        ]:
            lbl = QLabel(label + ":")
            val = QLabel("—")
            val.setObjectName(f"stat_{key}")
            self._stat_labels[key] = val
            self.stats_grid.addWidget(lbl, len(self._stat_labels) - 1, 0)
            self.stats_grid.addWidget(val, len(self._stat_labels) - 1, 1)

        self.btn_recalc = QPushButton("Пересчитать eval")
        self.btn_recalc.setObjectName("btn_recalc")
        self.btn_recalc.setToolTip("Повторно запустить eval-прогон выбранной модели")
        self.btn_recalc.setEnabled(False)
        self.btn_recalc.clicked.connect(self._eval_selected)

        layout.addWidget(self.stats_placeholder)
        layout.addLayout(self.stats_grid)
        layout.addWidget(self.btn_recalc)

        watch_box = QGroupBox("Наблюдение (live)")
        watch_box.setObjectName("watch_box")
        w_layout = QVBoxLayout(watch_box)
        self._watch_labels: Dict[str, QLabel] = {}
        for key, label in [
            ("day", "День"),
            ("action", "Действие"),
            ("reward", "Награда"),
            ("total_reward", "Суммарная награда"),
            ("people", "Люди"),
            ("bases", "Постройки"),
            ("money", "Деньги"),
        ]:
            lbl = QLabel(label + ":")
            val = QLabel("—")
            val.setObjectName(f"watch_{key}")
            self._watch_labels[key] = val
            w_layout.addWidget(lbl)
            w_layout.addWidget(val)
        layout.addWidget(watch_box)

        # --- Watch stage override ---
        watch_stage_box = QGroupBox("Этап курикулума при наблюдении")
        watch_stage_box.setObjectName("watch_stage_box")
        watch_stage_layout = QVBoxLayout(watch_stage_box)

        self.watch_use_model_stage_chk = QCheckBox("Использовать stage модели (из meta.json)")
        self.watch_use_model_stage_chk.setObjectName("watch_use_model_stage_chk")
        self.watch_use_model_stage_chk.setChecked(True)
        self.watch_use_model_stage_chk.toggled.connect(self._watch_stage_mode_changed)
        watch_stage_layout.addWidget(self.watch_use_model_stage_chk)

        self.watch_override_stage_chk = QCheckBox("Переопределить stage")
        self.watch_override_stage_chk.setObjectName("watch_override_stage_chk")
        self.watch_override_stage_chk.setChecked(False)
        self.watch_override_stage_chk.setEnabled(False)
        self.watch_override_stage_chk.toggled.connect(self._watch_stage_mode_changed)
        watch_stage_layout.addWidget(self.watch_override_stage_chk)

        stage_row = QHBoxLayout()
        stage_lbl = QLabel("Stage:")
        self.watch_stage_combo = QComboBox()
        self.watch_stage_combo.setObjectName("watch_stage_combo")
        self.watch_stage_combo.addItems([
            "0 — все здания",
            "1 — базовые",
            "2 — +средние",
            "3 — полные",
        ])
        self.watch_stage_combo.setCurrentIndex(2)
        stage_row.addWidget(stage_lbl)
        stage_row.addWidget(self.watch_stage_combo)
        stage_row.addStretch(1)
        watch_stage_layout.addLayout(stage_row)

        self.watch_stage_info_label = QLabel("")
        self.watch_stage_info_label.setObjectName("watch_stage_info_label")
        self.watch_stage_info_label.setWordWrap(True)
        self.watch_stage_info_label.setStyleSheet("color: #808080; font-size: 11px;")
        watch_stage_layout.addWidget(self.watch_stage_info_label)

        watch_stage_layout.addStretch(1)
        layout.addWidget(watch_stage_box)

        self.watch_visual_chk = QCheckBox("Визуальный режим (GUI окно)")
        self.watch_visual_chk.setObjectName("watch_visual_chk")
        self.watch_visual_chk.setChecked(False)
        self.watch_visual_chk.setToolTip("Открыть raylib-окно с визуализацией игры")
        layout.addWidget(self.watch_visual_chk)

        layout.addStretch(1)
        splitter.addWidget(box)

    # ---------- curriculum ----------

    CURRICULUM_PROFILES: Dict[str, List[List[int]]] = {
        "Отключён": [],
        "Стандартный": [[0, 1], [5_000_000, 2], [15_000_000, 3]],
        "Быстрый": [[0, 2], [3_000_000, 3]],
        "Медленный": [[0, 1], [10_000_000, 2], [30_000_000, 3]],
    }

    def _cur_add_row(self, threshold: int = 0, stage: int = 1):
        row = self.curriculum_table.rowCount()
        self.curriculum_table.insertRow(row)
        self.curriculum_table.setItem(row, 0, QTableWidgetItem(str(threshold)))
        self.curriculum_table.setItem(row, 1, QTableWidgetItem(str(stage)))

    def _cur_del_row(self):
        row = self.curriculum_table.currentRow()
        if row >= 0:
            self.curriculum_table.removeRow(row)

    def _cur_clear(self):
        self.curriculum_table.setRowCount(0)

    def _cur_profile_changed(self, _idx: int):
        name = self.curriculum_profile_combo.currentText()
        profile = self.CURRICULUM_PROFILES.get(name, [])
        self._cur_clear()
        for th, st in profile:
            self._cur_add_row(th, st)

    def _cur_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить расписание", "curriculum.json", "JSON (*.json)"
        )
        if not path:
            return
        data = self._get_curriculum_data()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.log("info", f"Расписание сохранено: {path}")

    def _cur_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить расписание", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                self.log("error", "Файл должен содержать список пар [порог, этап]")
                return
            self._cur_clear()
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    self._cur_add_row(int(item[0]), int(item[1]))
            self.log("info", f"Расписание загружено: {path}")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            self.log("error", f"Не удалось загрузить: {e}")

    def _get_curriculum_data(self) -> List[List[int]]:
        data: List[List[int]] = []
        for row in range(self.curriculum_table.rowCount()):
            th_item = self.curriculum_table.item(row, 0)
            st_item = self.curriculum_table.item(row, 1)
            if th_item and st_item:
                try:
                    th = int(th_item.text())
                    st = int(st_item.text())
                    if 1 <= st <= 3:
                        data.append([th, st])
                except ValueError:
                    continue
        data.sort(key=lambda x: x[0])
        return data

    def _validate_curriculum(self, data: List[List[int]]) -> tuple[bool, str]:
        for i in range(1, len(data)):
            if data[i][0] <= data[i - 1][0]:
                return False, f"Порог {data[i][0]:,} должен быть больше предыдущего {data[i-1][0]:,}"
        return True, "OK"

    # ---------- helpers ----------

    def log(self, level: str, message: str):
        color = LEVEL_COLORS.get(level, "#D4D4D4")
        self.console.appendHtml(f'<span style="color:{color};">{message}</span>')
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

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить лог", "train.log", "Текст (*.log *.txt);;Все файлы (*)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.console.toPlainText())
        self.log("info", f"Лог сохранён: {path}")

    # ---------- models ----------

    def refresh_models(self):
        self.model_table.setRowCount(0)
        models = self.registry.scan()
        self.models_hint.setVisible(not models)
        for m in models:
            r = self.model_table.rowCount()
            self.model_table.insertRow(r)
            cb_item = QTableWidgetItem()
            cb_item.setFlags(cb_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            cb_item.setCheckState(Qt.CheckState.Unchecked)
            self.model_table.setItem(r, 0, cb_item)
            self.model_table.setItem(r, 1, QTableWidgetItem(m.name))
            date_str = m.created.strftime("%Y-%m-%d %H:%M") if m.created else "—"
            self.model_table.setItem(r, 2, QTableWidgetItem(date_str))
            self.model_table.setItem(r, 3, QTableWidgetItem(f"{m.steps:,}"))
            self.model_table.setItem(r, 4, QTableWidgetItem(f"{m.best_reward:.2f}"))
        self.config["selected_model"] = self.config.get("selected_model")
        self._update_stats_panel()

    def _selected_model(self) -> Optional[ModelInfo]:
        r = self.model_table.currentRow()
        if r < 0:
            return None
        name = self.model_table.item(r, 1).text()
        return self.registry.get(name)

    def _on_model_clicked(self, row: int, _col: int):
        self._update_stats_panel()
        self._update_watch_stage_label()

    def _toggle_select_all(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for r in range(self.model_table.rowCount()):
            item = self.model_table.item(r, 0)
            if item:
                item.setCheckState(state)

    def _update_stats_panel(self):
        m = self._selected_model()
        if m is None:
            self.stats_placeholder.setVisible(True)
            for lbl in self._stat_labels.values():
                lbl.setText("—")
            self.btn_recalc.setEnabled(False)
            return
        self.stats_placeholder.setVisible(False)
        self._stat_labels["steps"].setText(f"{m.steps:,}")
        self._stat_labels["best_reward"].setText(f"{m.best_reward:.2f}")
        self._stat_labels["episodes"].setText(f"{m.episodes}")
        self._stat_labels["train_time_sec"].setText(f"{m.train_time_sec:.1f}")
        ev = m.eval or {}
        self._stat_labels["eval_days"].setText(f"{ev.get('days', 0):.1f}" if ev else "—")
        self._stat_labels["eval_people"].setText(f"{ev.get('people', 0):.1f}" if ev else "—")
        self._stat_labels["eval_bases"].setText(f"{ev.get('bases', 0):.1f}" if ev else "—")
        self.btn_recalc.setEnabled(True)

    def _delete_selected(self):
        names = []
        for r in range(self.model_table.rowCount()):
            item = self.model_table.item(r, 0)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(self.model_table.item(r, 1).text())
        if not names:
            QMessageBox.information(self, "Удаление", "Не выбраны модели")
            return
        msg = QMessageBox()
        msg.setWindowTitle("Подтверждение удаления")
        msg.setText(f"Удалить {len(names)} модель(и)?")
        msg.setInformativeText("Действие необратимо: файлы будут удалены с диска.\n" +
                               "\n".join(names[:10]) + ("\n…" if len(names) > 10 else ""))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self.registry.delete_many(names)
            self.log("info", f"Удалено моделей: {len(removed)}")
        except Exception as e:
            self.log("error", f"Ошибка удаления: {e}")
        self.refresh_models()

    def _eval_selected(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Оценка", "Сначала выберите модель из списка")
            return
        import sys as _sys
        import tempfile
        eval_msg = os.path.join(tempfile.gettempdir(), f"colony_eval_{int(time.time()*1000)}.jsonl")
        if os.path.exists(eval_msg):
            os.unlink(eval_msg)
        args = [
            "-u",
            str(Path(__file__).resolve().parent / "worker.py"),
            "--eval-model", str(m.model_file),
            "--episodes", "5",
            "--max-days", "1000",
            "--device", "cpu",
            "--output", eval_msg,
        ]
        ok, pid = QProcess.startDetached(_sys.executable, args, str(Path(__file__).resolve().parent.parent))
        if not ok:
            self.log("error", "Не удалось запустить eval")
            self.btn_eval.setEnabled(True)
            return
        self.btn_eval.setEnabled(False)
        self._eval_pid = pid
        self._eval_msg = eval_msg
        self._eval_offset = 0
        self._eval_model = m
        self._eval_timer = QTimer(self)
        self._eval_timer.timeout.connect(self._poll_eval)
        self._eval_timer.start(250)
        self.log("info", f"Запуск eval: {m.name} (pid={pid})")

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
            if d.get("type") == "eval_result":
                m = self._eval_model
                self.registry.save_eval(m.name, {
                    "days": d.get("days", 0),
                    "people": d.get("people", 0),
                    "bases": d.get("bases", 0),
                    "avg_return": d.get("avg_return", 0),
                })
                self.log("info", f"Eval {m.name}: days={d.get('days', 0):.1f} "
                                  f"people={d.get('people', 0):.1f} bases={d.get('bases', 0):.1f}")
                self._update_stats_panel()
                self._eval_done = True
            elif d.get("type") == "log":
                self.log(d.get("level", "info"), d.get("message", ""))
            elif d.get("type") == "error":
                self.log("error", d.get("message", ""))

    def _poll_eval(self):
        self._read_eval_file()
        if self._eval_pid is None:
            return
        if not hasattr(self, '_eval_start_time'):
            self._eval_start_time = time.time()
        if time.time() - self._eval_start_time < 2.0:
            return
        if not self._poll_pid_alive(self._eval_pid):
            self._read_eval_file()
            eval_succeeded = self._eval_done
            self._cleanup_eval()
            if not eval_succeeded:
                self.log("error", "Eval завершился с ошибкой")
            self.btn_eval.setEnabled(True)

    def _resume_training(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Дообучение", "Сначала выберите модель из списка")
            return
        if not m.model_file.exists():
            self.log("error", f"Файл модели не найден: {m.model_file}")
            return
        self.log("info", f"Дообучение модели: {m.name}")
        self._start_training(resume_model=m.model_file)

    def _cleanup_eval(self):
        if self._eval_timer is not None:
            self._eval_timer.stop()
            self._eval_timer = None
        if self._eval_msg is not None:
            try:
                os.unlink(self._eval_msg)
            except OSError:
                pass
            self._eval_msg = None
            self._eval_offset = 0
        self._eval_pid = None
        self._eval_model = None
        self._eval_done = False
        self._eval_start_time = 0

    # ---------- watch champion ----------

    def _watch_selected(self):
        m = self._selected_model()
        if m is None:
            QMessageBox.information(self, "Наблюдение", "Сначала выберите модель из списка")
            return
        if self._watch_pid is not None:
            self.log("warn", "Наблюдение уже запущено")
            return
        import sys as _sys
        import tempfile
        watch_msg = os.path.join(tempfile.gettempdir(), f"colony_watch_{int(time.time()*1000)}.log")
        model_dir = m.model_file.parent
        args = [
            "-u",
            str(Path(__file__).resolve().parent.parent / "watch_champion.py"),
            "--model-dir", str(model_dir),
            "--episodes", "1",
            "--max-steps", "500",
            "--speed", "5",
            "--device", "cpu",
            "--log-file", watch_msg,
        ]
        stage = self._get_watch_stage()
        if stage is not None:
            args.extend(["--curriculum-stage", str(stage)])
        if self.watch_visual_chk.isChecked():
            args.append("--visual")
        workdir = str(Path(__file__).resolve().parent.parent)
        ok, pid = QProcess.startDetached(_sys.executable, args, workdir)
        if not ok:
            self.log("error", "Не удалось запустить наблюдение")
            return
        self._watch_pid = pid
        self._watch_start_time = time.time()
        self._watch_log = watch_msg
        self._watch_log_offset = 0
        self._reset_watch_stats()
        self.btn_watch.setEnabled(False)
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._poll_watch)
        self._watch_timer.start(500)
        self.log("info", f"Наблюдение за {m.name} запущено (pid={pid}), 500 шагов @ 5 ш/с")

    def _poll_watch(self):
        self._read_watch_log()
        if self._watch_pid is None:
            return
        if time.time() - self._watch_start_time < 2.0:
            return
        if not self._watch_process_alive():
            self._read_watch_log()
            self._cleanup_watch()
            self.log("info", "Наблюдение завершено")
            self.btn_watch.setEnabled(True)

    def _read_watch_log(self):
        if getattr(self, "_watch_log", None) is None:
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
                        self._update_watch_stats(d)
                        continue
                    elif d.get("type") == "log":
                        self.log("info", d.get("message", ""))
                        continue
                    elif d.get("type") == "error":
                        self.log("error", d.get("message", ""))
                        continue
                    elif d.get("type") == "done":
                        self.log("info", d.get("message", "Наблюдение завершено"))
                        continue
                except json.JSONDecodeError:
                    pass
            self.log("info", line)

    def _update_watch_stats(self, d: Dict[str, Any]):
        for key in ("day", "action", "reward", "total_reward", "people", "bases", "money"):
            lbl = self._watch_labels.get(key)
            if lbl is None:
                continue
            if key == "action":
                lbl.setText(str(d.get("action", "—")))
            elif key == "reward":
                lbl.setText(f"{d.get('reward', 0.0):+.2f}")
            elif key == "total_reward":
                lbl.setText(f"{d.get('total_reward', 0.0):+.1f}")
            else:
                lbl.setText(str(d.get(key, "—")))

    # ---------- watch stage ----------

    WATCH_STAGE_INFO: Dict[int, str] = {
        0: "Все здания доступны (куррикулум отключён).",
        1: "Базовые здания: Дом, Ферма, Огород, Переработка, Рыбалка, Пастбище, Пчельник, Ловушка, Свалка, Промзона, Шахта, Порт, Аэропорт, Военный лагерь.",
        2: "+ Средние: Пила, Угольная шахта, Железная шахта, Электростанция, Каменная шахта, Деревообработка, Текстиль, Сталь, Химия, Электроника, Склад.",
        3: "+ Полные: Большая пила, Атомная станция, Супердом.",
    }

    def _watch_stage_mode_changed(self, _checked: bool = False):
        use_model = self.watch_use_model_stage_chk.isChecked()
        override = self.watch_override_stage_chk.isChecked()

        if use_model:
            self.watch_override_stage_chk.setEnabled(False)
            self.watch_stage_combo.setEnabled(False)
            self.watch_stage_info_label.setText("Stage будет прочитан из best_model.meta.json модели.")
        elif override:
            self.watch_stage_combo.setEnabled(True)
            stage = self._watch_stage_from_combo()
            self.watch_stage_info_label.setText(self.WATCH_STAGE_INFO.get(stage, ""))
        else:
            self.watch_stage_combo.setEnabled(False)
            self.watch_stage_info_label.setText("Выберите режим: stage модели или переопределение.")

    def _watch_stage_from_combo(self) -> int:
        idx = self.watch_stage_combo.currentIndex()
        return idx  # 0=0, 1=1, 2=2, 3=3

    def _get_watch_stage(self):
        """Return stage int or None. None = let watch_champion.py read from meta."""
        if self.watch_use_model_stage_chk.isChecked():
            return None
        if self.watch_override_stage_chk.isChecked():
            return self._watch_stage_from_combo()
        return None

    def _update_watch_stage_label(self):
        """Show which stage the selected model was trained at (from meta.json)."""
        m = self._selected_model()
        if m is None:
            return
        meta_path = m.path / "best_model.meta.json"
        if not meta_path.exists():
            self.watch_stage_info_label.setText("Meta-файл не найден — stage не определён.")
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            stage = meta.get("curriculum_stage_at_best")
            if stage is not None:
                self.watch_stage_info_label.setText(
                    f"Модель обучена на stage {stage}. "
                    + self.WATCH_STAGE_INFO.get(stage, "")
                )
            else:
                self.watch_stage_info_label.setText("Модель обучена без курикулума (все здания).")
        except (json.JSONDecodeError, OSError):
            self.watch_stage_info_label.setText("Не удалось прочитать meta.json.")

    def _reset_watch_stats(self):
        for lbl in self._watch_labels.values():
            lbl.setText("—")

    def _watch_process_alive(self) -> bool:
        if self._watch_pid is None:
            return False
        try:
            subprocess.run(["tasklist", "/FI", f"PID eq {self._watch_pid}"],
                           capture_output=True, timeout=5)
        except Exception:
            return False
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        h = kernel32.OpenProcess(SYNCHRONIZE, 0, self._watch_pid)
        if not h:
            return False
        result = kernel32.WaitForSingleObject(h, 0)
        kernel32.CloseHandle(h)
        return result != 0x0

    def _cleanup_watch(self):
        if self._watch_timer is not None:
            self._watch_timer.stop()
            self._watch_timer = None
        if getattr(self, "_watch_log", None) is not None:
            try:
                os.unlink(self._watch_log)
            except OSError:
                pass
            self._watch_log = None
            self._watch_log_offset = 0
        self._watch_pid = None
        self._watch_start_time = 0

    # ---------- training ----------

    def _on_param_changed(self):
        pass

    def _randomize_seed(self):
        spec = spec_for("seed")
        lo, hi = int(spec.min), int(spec.max)
        self.param_rows["seed"].set_value(random.randint(lo, hi))

    def _on_random_seed_toggled(self, checked: bool):
        if checked:
            self._randomize_seed()

    def _collect_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "name": self.name_edit.text().strip() or "run",
            "use_amp": self.chk_amp.isChecked(),
            "torch_compile": self.chk_compile.isChecked(),
        }
        for key, row in self.param_rows.items():
            cfg[key] = row.value()
        net = int(cfg.pop("net_arch", 256))
        cfg["net_arch"] = [net, net]
        curriculum = self._get_curriculum_data()
        if curriculum:
            ok, msg = self._validate_curriculum(curriculum)
            if not ok:
                QMessageBox.warning(self, "Курикулум", msg)
                curriculum = []
        cfg["curriculum_schedule"] = curriculum
        return cfg

    def _start_training(self, resume_model: Optional[Path] = None):
        if self._train_pid is not None:
            self.log("warn", "Обучение уже запущено")
            return
        if getattr(self, "chk_random_seed", None) is not None and self.chk_random_seed.isChecked():
            self._randomize_seed()
        cfg = self._collect_config()
        if resume_model is not None:
            cfg["name"] = cfg["name"] + "_ft"
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(cfg, tmp, ensure_ascii=False)
        tmp.close()
        msg_file = os.path.join(tempfile.gettempdir(), f"colony_ui_{int(time.time()*1000)}.jsonl")

        import sys as _sys
        args = [
            "-u",
            str(Path(__file__).resolve().parent / "worker.py"),
            "--config", tmp.name,
            "--name", cfg["name"],
            "--output", msg_file,
        ]
        if resume_model is not None:
            args.extend(["--resume-model", str(resume_model)])
        workdir = str(Path(__file__).resolve().parent.parent)
        ok, pid = QProcess.startDetached(_sys.executable, args, workdir)
        if not ok:
            self.log("error", "Не удалось запустить процесс обучения")
            return
        self._train_pid = pid
        self._msg_file = msg_file
        self._msg_offset = 0

        self._msg_timer = QTimer(self)
        self._msg_timer.timeout.connect(self._poll_training)
        self._msg_timer.start(250)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск…")
        self.log("info", f"Запуск обучения: {cfg['name']} steps={cfg['total_timesteps']:,} (pid={pid})")

    def _poll_training(self):
        self._poll_messages()
        if self._train_pid is not None and not self._poll_process_alive():
            self._poll_messages()
            self._on_train_finished(0)

    def _poll_messages(self):
        if self._msg_file is None:
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
            except ValueError as e:
                self.log("warn", f"Некорректная строка: {line[:80]} ({e})")
                continue
            if isinstance(msg, P.ReadyMsg):
                self.log("info", "Процесс обучения готов")
            elif isinstance(msg, P.LogMsg):
                self.log(msg.level, msg.message)
            elif isinstance(msg, P.ProgressMsg):
                self._update_progress(msg)
            elif isinstance(msg, P.SavedMsg):
                self.log("info", f"Сохранён чекпоинт: {msg.path}")
            elif isinstance(msg, P.DoneMsg):
                self.log("info", f"Обучение завершено: {msg.total:,} шагов за {msg.time_s:.1f}s, "
                                  f"best={msg.best_reward:.2f}, эпизодов={msg.episodes}")
            elif isinstance(msg, P.ErrorMsg):
                self.log("error", msg.message)

    def _update_progress(self, m: P.ProgressMsg):
        total = max(m.total, 1)
        pct = int(1000 * m.done / total)
        self.progress_bar.setValue(min(1000, pct))
        fps = m.fps if m.fps > 0 else 0.0
        eta = ""
        if fps > 0 and m.done < m.total:
            secs = (m.total - m.done) / fps
            eta = f" · осталось ~{int(secs // 60)} мин"
        self.status_label.setText(
            f"{m.done:,}/{m.total:,} · FPS {fps:,.0f} · best {m.best_reward:.1f}"
            f" · эпизоды {m.episodes}{eta}"
        )

    def _stop_training(self):
        if self._train_pid is None:
            return
        self.log("info", "Остановка процесса…")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(self._train_pid)],
                           capture_output=True, timeout=10)
            self.log("warn", "Процесс обучения принудительно завершён")
        except Exception as e:
            self.log("error", f"Не удалось остановить процесс: {e}")
        self._cleanup_training()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Остановлено")

    def _poll_pid_alive(self, pid: int) -> bool:
        if pid is None:
            return False
        try:
            subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                           capture_output=True, timeout=5)
        except Exception:
            return False
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        h = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
        if not h:
            return False
        result = kernel32.WaitForSingleObject(h, 0)
        kernel32.CloseHandle(h)
        return result != 0x0

    def _poll_process_alive(self) -> bool:
        return self._poll_pid_alive(self._train_pid)

    def _on_train_finished(self, code: int = 0):
        if self._pending_stop_timer is not None and self._pending_stop_timer.isActive():
            self._pending_stop_timer.stop()
        self._cleanup_training()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self.progress_bar.setValue(1000)
            self.status_label.setText("Завершено")
            self._set_progress_green(True)
        else:
            self.log("error", f"Процесс обучения завершился с кодом {code}")
        self.refresh_models()
        self.train_finished.emit(code)

    def _cleanup_training(self):
        if self._msg_timer is not None:
            self._msg_timer.stop()
            self._msg_timer = None
        if self._msg_file is not None:
            try:
                os.unlink(self._msg_file)
            except OSError:
                pass
            self._msg_file = None
            self._msg_offset = 0
        self._train_pid = None

    def _set_progress_green(self, on: bool):
        palette = self.progress_bar.palette()
        if on:
            palette.setColor(QPalette.ColorRole.Highlight, QColor("#2E8B57"))
        else:
            palette.setColor(QPalette.ColorRole.Highlight, QColor("#4A90D9"))
        self.progress_bar.setPalette(palette)
        self.progress_bar.update()

    # ---------- state ----------

    def _reset_params(self):
        self.name_edit.setText(str(DEFAULT_PARAMS["name"]))
        for key, row in self.param_rows.items():
            row.set_value(DEFAULT_PARAMS[key])
        self.chk_amp.setChecked(bool(DEFAULT_PARAMS["use_amp"]))
        self.chk_compile.setChecked(bool(DEFAULT_PARAMS["torch_compile"]))
        if getattr(self, "chk_random_seed", None) is not None:
            self.chk_random_seed.blockSignals(True)
            self.chk_random_seed.setChecked(False)
            self.chk_random_seed.blockSignals(False)
        self.curriculum_profile_combo.blockSignals(True)
        self.curriculum_profile_combo.setCurrentIndex(1)
        self.curriculum_profile_combo.blockSignals(False)
        self._cur_profile_changed(1)
        self.log("info", "Параметры сброшены к значениям по умолчанию")

    def _restore_state(self):
        cfg = self.config
        self.name_edit.setText(str(cfg.get("name", DEFAULT_PARAMS["name"])))
        for key, row in self.param_rows.items():
            v = cfg.get(key, DEFAULT_PARAMS[key])
            if isinstance(v, (list, tuple)):
                v = v[0] if v else DEFAULT_PARAMS[key]
            row.set_value(float(v))
        self.chk_amp.setChecked(bool(cfg.get("use_amp", DEFAULT_PARAMS["use_amp"])))
        self.chk_compile.setChecked(bool(cfg.get("torch_compile", DEFAULT_PARAMS["torch_compile"])))
        if getattr(self, "chk_random_seed", None) is not None:
            self.chk_random_seed.blockSignals(True)
            self.chk_random_seed.setChecked(bool(cfg.get("random_seed", False)))
            self.chk_random_seed.blockSignals(False)
        geom = cfg.get("geometry")
        if geom:
            try:
                self.resize(geom["w"], geom["h"])
            except Exception:
                pass
        curriculum = cfg.get("curriculum_schedule")
        if curriculum is not None:
            self._cur_clear()
            for item in curriculum:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    self._cur_add_row(int(item[0]), int(item[1]))
        else:
            self._cur_profile_changed(1)

    def save_state(self) -> Dict[str, Any]:
        state = self.config.copy()
        state["name"] = self.name_edit.text()
        state["use_amp"] = self.chk_amp.isChecked()
        state["torch_compile"] = self.chk_compile.isChecked()
        state["random_seed"] = bool(
            getattr(self, "chk_random_seed", None) is not None
            and self.chk_random_seed.isChecked()
        )
        for key, row in self.param_rows.items():
            state[key] = row.value()
        state["curriculum_schedule"] = self._get_curriculum_data()
        state["geometry"] = {"w": self.width(), "h": self.height()}
        m = self._selected_model()
        if m is not None:
            state["selected_model"] = m.name
        return state


def save_config(state: Dict[str, Any]):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Не удалось сохранить конфиг: {e}")


def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
