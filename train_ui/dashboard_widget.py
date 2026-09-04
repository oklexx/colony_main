#!/usr/bin/env python3
"""Interactive Training Dashboard - real-time plots with pyqtgraph."""

from __future__ import annotations

import numpy as np
import time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer
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

        title_label = QLabel("Training Dashboard")
        title_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #4a90d9; padding: 2px; background: transparent;")
        layout.addWidget(title_label)

        splitter = QSplitter(Qt.Horizontal)

        # KL plot
        kl_frame = QFrame()
        kl_frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 4px; }")
        kl_layout = QVBoxLayout(kl_frame)
        kl_layout.setContentsMargins(2, 2, 2, 2)
        kl_title = QLabel("KL")
        kl_title.setStyleSheet("font-size: 8pt; color: #888; padding: 1px; background: transparent;")
        kl_layout.addWidget(kl_title)
        self._kl_plot_widget = pg.PlotWidget(background='#1e1e1e', showGrid=(True, True, '#333'))
        self._kl_plot_widget.setMouseEnabled(x=False, y=False)
        self._kl_plot_widget.hideAxis('bottom')
        self._kl_plot_widget.hideAxis('left')
        self._kl_plot_widget.setMinimumHeight(60)
        kl_layout.addWidget(self._kl_plot_widget)
        splitter.addWidget(kl_frame)

        # Entropy plot
        ent_frame = QFrame()
        ent_frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 4px; }")
        ent_layout = QVBoxLayout(ent_frame)
        ent_layout.setContentsMargins(2, 2, 2, 2)
        ent_title = QLabel("Entropy")
        ent_title.setStyleSheet("font-size: 8pt; color: #888; padding: 1px; background: transparent;")
        ent_layout.addWidget(ent_title)
        self._entropy_plot_widget = pg.PlotWidget(background='#1e1e1e', showGrid=(True, True, '#333'))
        self._entropy_plot_widget.setMouseEnabled(x=False, y=False)
        self._entropy_plot_widget.hideAxis('bottom')
        self._entropy_plot_widget.hideAxis('left')
        self._entropy_plot_widget.setMinimumHeight(60)
        ent_layout.addWidget(self._entropy_plot_widget)
        splitter.addWidget(ent_frame)

        # Action distribution plot
        act_frame = QFrame()
        act_frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 4px; }")
        act_layout = QVBoxLayout(act_frame)
        act_layout.setContentsMargins(2, 2, 2, 2)
        act_title = QLabel("Actions")
        act_title.setStyleSheet("font-size: 8pt; color: #888; padding: 1px; background: transparent;")
        act_layout.addWidget(act_title)
        self._action_plot_widget = pg.PlotWidget(background='#1e1e1e', showGrid=(True, True, '#333'))
        self._action_plot_widget.setLabel('left', '%', units='')
        self._action_plot_widget.setLabel('bottom', '', units='')
        self._action_plot_widget.setMinimumHeight(60)
        act_layout.addWidget(self._action_plot_widget)
        splitter.addWidget(act_frame)

        layout.addWidget(splitter)

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
