#!/usr/bin/env python3
"""Interactive Training Dashboard - real-time plots with pyqtgraph."""

from __future__ import annotations

import numpy as np
import time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QSplitter, QPushButton, QToolTip,
    QProgressBar, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint
from PySide6.QtGui import QCursor
from PySide6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class TrainingDashboardWidget(QWidget):
    """Interactive dashboard with real-time training plots."""

    plot_data_received = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._kl_buffer = []
        self._entropy_buffer = []
        self._action_buffer = []
        self._buffer_size = 1000

        self._kl_curve = None
        self._entropy_curve = None
        self._target_line = None
        self._kl_plot_widget = None
        self._entropy_plot_widget = None
        self._action_plot_widget = None
        self._action_bar = None

        self._scroll_timer = QTimer()
        self._scroll_timer.timeout.connect(self._scroll_view)
        self._scroll_timer.setInterval(5000)

        self._setup_ui()
        self._setup_plots()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(3)
        title_label = QLabel("Training Dashboard")
        title_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #4a90d9; padding: 2px; background: transparent;")
        title_row.addWidget(title_label)
        help_btn = QPushButton("?")
        help_btn.setStyleSheet(
            "QPushButton { font-size:8px; font-weight:bold; color:#888; "
            "background:#2a2a2a; border:1px solid #444; border-radius:6px; "
            "min-width:12px; max-width:12px; min-height:12px; max-height:12px; "
            "padding:0; }"
            "QPushButton:hover { color:#4a90d9; border-color:#4a90d9; }")
        help_btn.setCursor(QCursor(Qt.PointingHandCursor))
        help_btn.clicked.connect(lambda: QToolTip.showText(
            help_btn.mapToGlobal(QPoint(help_btn.width() + 4, 0)),
            "Интерактивные графики в реальном времени.\n"
            "• KL — график KL-дивергенции (красная линия = целевое значение)\n"
            "• Entropy — график энтропии (разнообразия действий)\n"
            "• Actions — распределение частоты выбора действий (%)\n"
            "Графики автоматически прокручиваются по времени.",
            help_btn))
        title_row.addWidget(help_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ожидание…")
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setFixedWidth(280)
        self.progress_bar.setStyleSheet("font-size:9px;")
        title_row.addWidget(self.progress_bar)

        title_row.addStretch(1)
        layout.addLayout(title_row)

        # Use QHBoxLayout instead of QSplitter to avoid stretching
        plots_layout = QHBoxLayout()
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(4)

        # KL plot
        kl_frame = QFrame()
        kl_frame.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        kl_layout = QVBoxLayout(kl_frame)
        kl_layout.setContentsMargins(0, 0, 0, 0)
        kl_layout.setSpacing(0)
        kl_title_row = QHBoxLayout()
        kl_title_row.setContentsMargins(0, 0, 0, 0)
        kl_title_row.setSpacing(1)
        kl_title = QLabel("KL")
        kl_title.setStyleSheet("font-size: 7pt; color: #888; padding: 0px; margin: 0px; background: transparent;")
        kl_title.setFixedHeight(12)
        kl_title_row.addWidget(kl_title)
        kl_help = QPushButton("?")
        kl_help.setStyleSheet(
            "QPushButton { font-size:7px; font-weight:bold; color:#666; "
            "background:#222; border:1px solid #333; border-radius:5px; "
            "min-width:10px; max-width:10px; min-height:10px; max-height:10px; "
            "padding:0; }"
            "QPushButton:hover { color:#4a90d9; border-color:#4a90d9; }")
        kl_help.setCursor(QCursor(Qt.PointingHandCursor))
        kl_help.clicked.connect(lambda: QToolTip.showText(
            kl_help.mapToGlobal(QPoint(kl_help.width() + 2, 0)),
            "KL-дивергенция — разница между текущей и исходной стратегией.\n"
            "Красная пунктирная линия = целевое значение (~0.05).", kl_help))
        kl_title_row.addWidget(kl_help)
        kl_title_row.addStretch(1)
        kl_layout.addLayout(kl_title_row)
        self._kl_plot_widget = pg.PlotWidget(background='#1e1e1e', showGrid=(True, True, '#333'))
        self._kl_plot_widget.setMouseEnabled(x=False, y=False)
        self._kl_plot_widget.hideAxis('bottom')
        self._kl_plot_widget.hideAxis('left')
        self._kl_plot_widget.setMinimumHeight(80)
        self._kl_plot_widget.setMaximumHeight(110)
        self._kl_plot_widget.setStyleSheet("QGraphicsView { padding: 0px; margin: 0px; }")
        kl_layout.addWidget(self._kl_plot_widget)
        plots_layout.addWidget(kl_frame, 1)

        # Entropy plot
        ent_frame = QFrame()
        ent_frame.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        ent_layout = QVBoxLayout(ent_frame)
        ent_layout.setContentsMargins(0, 0, 0, 0)
        ent_layout.setSpacing(0)
        ent_title_row = QHBoxLayout()
        ent_title_row.setContentsMargins(0, 0, 0, 0)
        ent_title_row.setSpacing(1)
        ent_title = QLabel("Entropy")
        ent_title.setStyleSheet("font-size: 7pt; color: #888; padding: 0px; margin: 0px; background: transparent;")
        ent_title.setFixedHeight(12)
        ent_title_row.addWidget(ent_title)
        ent_help = QPushButton("?")
        ent_help.setStyleSheet(
            "QPushButton { font-size:7px; font-weight:bold; color:#666; "
            "background:#222; border:1px solid #333; border-radius:5px; "
            "min-width:10px; max-width:10px; min-height:10px; max-height:10px; "
            "padding:0; }"
            "QPushButton:hover { color:#4a90d9; border-color:#4a90d9; }")
        ent_help.setCursor(QCursor(Qt.PointingHandCursor))
        ent_help.clicked.connect(lambda: QToolTip.showText(
            ent_help.mapToGlobal(QPoint(ent_help.width() + 2, 0)),
            "Энтропия — мера разнообразия действий агента.\n"
            "Высокая энтропия = агент исследует больше действий.\n"
            "Низкая энтропия = агент повторяет одни и те же действия.", ent_help))
        ent_title_row.addWidget(ent_help)
        ent_title_row.addStretch(1)
        ent_layout.addLayout(ent_title_row)
        self._entropy_plot_widget = pg.PlotWidget(background='#1e1e1e', showGrid=(True, True, '#333'))
        self._entropy_plot_widget.setMouseEnabled(x=False, y=False)
        self._entropy_plot_widget.hideAxis('bottom')
        self._entropy_plot_widget.hideAxis('left')
        self._entropy_plot_widget.setMinimumHeight(80)
        self._entropy_plot_widget.setMaximumHeight(110)
        self._entropy_plot_widget.setStyleSheet("QGraphicsView { padding: 0px; margin: 0px; }")
        ent_layout.addWidget(self._entropy_plot_widget)
        plots_layout.addWidget(ent_frame, 1)

        # Action distribution plot
        act_frame = QFrame()
        act_frame.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        act_layout = QVBoxLayout(act_frame)
        act_layout.setContentsMargins(0, 0, 0, 0)
        act_layout.setSpacing(0)
        act_title_row = QHBoxLayout()
        act_title_row.setContentsMargins(0, 0, 0, 0)
        act_title_row.setSpacing(1)
        act_title = QLabel("Actions")
        act_title.setStyleSheet("font-size: 7pt; color: #888; padding: 0px; margin: 0px; background: transparent;")
        act_title.setFixedHeight(12)
        act_title_row.addWidget(act_title)
        act_help = QPushButton("?")
        act_help.setStyleSheet(
            "QPushButton { font-size:7px; font-weight:bold; color:#666; "
            "background:#222; border:1px solid #333; border-radius:5px; "
            "min-width:10px; max-width:10px; min-height:10px; max-height:10px; "
            "padding:0; }"
            "QPushButton:hover { color:#4a90d9; border-color:#4a90d9; }")
        act_help.setCursor(QCursor(Qt.PointingHandCursor))
        act_help.clicked.connect(lambda: QToolTip.showText(
            act_help.mapToGlobal(QPoint(act_help.width() + 2, 0)),
            "Распределение частоты выбора действий агентом (%).\n"
            "Показывает, какие действия агент выбирает чаще всего.\n"
            "10 столбцов = 10 наиболее частых действий.", act_help))
        act_title_row.addWidget(act_help)
        act_title_row.addStretch(1)
        act_layout.addLayout(act_title_row)
        self._action_plot_widget = pg.PlotWidget(background='#1e1e1e', showGrid=(True, True, '#333'))
        self._action_plot_widget.setLabel('left', '%', units='')
        self._action_plot_widget.setLabel('bottom', '', units='')
        self._action_plot_widget.setMinimumHeight(80)
        self._action_plot_widget.setMaximumHeight(110)
        self._action_plot_widget.setStyleSheet("QGraphicsView { padding: 0px; margin: 0px; }")
        act_layout.addWidget(self._action_plot_widget)
        plots_layout.addWidget(act_frame, 1)

        layout.addLayout(plots_layout)

    def _setup_plots(self):
        if self._kl_plot_widget is None:
            return

        self._kl_curve = self._kl_plot_widget.plot(pen=pg.mkPen('cyan', width=2), symbol='o', symbolSize=3)
        self._target_line = pg.InfiniteLine(pos=0.05, angle=90, pen=pg.mkPen('r', width=1, style=Qt.DashLine), movable=False)
        self._kl_plot_widget.addItem(self._target_line)
        self._kl_plot_widget.showAxis('bottom')
        self._kl_plot_widget.showAxis('left')
        self._kl_plot_widget.setLabel('left', 'KL', units='')
        self._kl_plot_widget.setLabel('bottom', 'Time', units='s')

        self._entropy_curve = self._entropy_plot_widget.plot(pen=pg.mkPen('lime', width=2), symbol='o', symbolSize=3)
        self._entropy_plot_widget.showAxis('bottom')
        self._entropy_plot_widget.showAxis('left')
        self._entropy_plot_widget.setLabel('left', 'Entropy', units='')
        self._entropy_plot_widget.setLabel('bottom', 'Time', units='s')

        self._action_bar = pg.BarGraphItem(x=[0], height=[0], width=0.8, brush='cyan')
        self._action_plot_widget.addItem(self._action_bar)
        self._action_plot_widget.setYRange(0, 100)
        self._action_plot_widget.setXRange(-0.5, 9.5)
        self._action_plot_widget.showAxis('bottom')
        self._action_plot_widget.showAxis('left')
        self._action_plot_widget.setLabel('left', '%', units='')
        self._action_plot_widget.setLabel('bottom', 'Action', units='')

        self._scroll_timer.start()

    def update_data(self, kl: float = None, entropy: float = None,
                    top_actions: dict = None):
        self._update_kl_plot(kl)
        self._update_entropy_plot(entropy)
        self._update_action_plot(top_actions)

    def _update_kl_plot(self, kl: float):
        if kl is not None and not np.isnan(kl):
            self._kl_buffer.append((time.time(), kl))
            if len(self._kl_buffer) > self._buffer_size:
                self._kl_buffer.pop(0)
            if self._kl_curve is not None and len(self._kl_buffer) > 1:
                x = np.array([t for t, _ in self._kl_buffer])
                y = np.array([v for _, v in self._kl_buffer])
                self._kl_curve.setData(x, y)

    def _update_entropy_plot(self, entropy: float):
        if entropy is not None and not np.isnan(entropy):
            self._entropy_buffer.append((time.time(), entropy))
            if len(self._entropy_buffer) > self._buffer_size:
                self._entropy_buffer.pop(0)
            if self._entropy_curve is not None and len(self._entropy_buffer) > 1:
                x = np.array([t for t, _ in self._entropy_buffer])
                y = np.array([v for _, v in self._entropy_buffer])
                self._entropy_curve.setData(x, y)

    def _update_action_plot(self, top_actions: dict):
        if top_actions is None or not top_actions:
            return
        values = list(top_actions.values())
        if self._action_bar is not None:
            n = len(values[:10])
            self._action_bar.setOpts(x=np.arange(n), height=values[:10], width=0.8)

    def _scroll_view(self):
        now = time.time()
        if self._kl_plot_widget is not None and self._kl_buffer:
            self._kl_plot_widget.setXRange(now - 60, now, padding=0)
            if len(self._kl_buffer) >= 2:
                vals = [v for _, v in self._kl_buffer[-60:]]
                lo = min(vals)
                hi = max(vals)
                margin = max((hi - lo) * 0.2, 0.001)
                self._kl_plot_widget.setYRange(lo - margin, hi + margin, padding=0)
        if self._entropy_plot_widget is not None and self._entropy_buffer:
            self._entropy_plot_widget.setXRange(now - 60, now, padding=0)
            if len(self._entropy_buffer) >= 2:
                vals = [v for _, v in self._entropy_buffer[-60:]]
                lo = min(vals)
                hi = max(vals)
                margin = max((hi - lo) * 0.2, 0.001)
                self._entropy_plot_widget.setYRange(lo - margin, hi + margin, padding=0)
