#!/usr/bin/env python3
"""Action Loop Widget - displays loop detection alerts and statistics."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView, QToolTip
)
from PySide6.QtCore import Qt, Property, QPoint
from PySide6.QtGui import QCursor
from typing import Dict, List, Optional


class ActionLoopWidget(QWidget):
    """Widget displaying action loop detection alerts and statistics."""
    
    def __init__(self, parent=None):
        # Initialize attributes BEFORE _setup_ui()
        self._consecutive_threshold = 3  # Default threshold
        self._loop_detected = False
        self._loop_action_name = None
        self._envs_with_loops = 0
        
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self._alert_frame = QFrame()
        self._alert_frame.setVisible(False)
        self._alert_frame.setStyleSheet("background-color: #3e1a1a; border: 1px solid #c62828; border-radius: 4px; padding: 4px;")
        alert_layout = QVBoxLayout(self._alert_frame)
        alert_layout.setContentsMargins(4, 2, 4, 2)
        alert_layout.setSpacing(1)

        self._alert_label = QLabel("LOOP DETECTED")
        self._alert_label.setStyleSheet("font-size: 9pt; font-weight: bold; color: #ef5350; background: transparent;")
        alert_layout.addWidget(self._alert_label)

        self._action_name_label = QLabel("Action: --")
        self._action_name_label.setStyleSheet("font-size: 8pt; color: #e57373; background: transparent;")
        alert_layout.addWidget(self._action_name_label)

        self._stats_label = QLabel("Consecutive: 0 / 3")
        self._stats_label.setStyleSheet("font-size: 8pt; color: #ef9a9a; background: transparent;")
        alert_layout.addWidget(self._stats_label)

        layout.addWidget(self._alert_frame)

        self._alert_spacer = QLabel("")
        self._alert_spacer.setVisible(True)
        layout.addWidget(self._alert_spacer)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(3)
        title = QLabel("Loop Detection")
        title.setStyleSheet("font-size: 10pt; font-weight: bold; color: #4a90d9; padding: 2px; background: transparent;")
        title_row.addWidget(title)
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
            "Обнаружение зацикливания действий агента.\n"
            "Если агент повторяет одно и то же действие несколько раз\n"
            "подряд (превышая порог), появляется предупреждение.\n"
            "• Envs in Loops — количество сред в состоянии зацикливания\n"
            "• Status — ACTIVE (есть зацикливание) или Normal\n"
            "• Action — название повторяющегося действия\n"
            "• Threshold — порог подряд идущих повторов",
            help_btn))
        title_row.addWidget(help_btn)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self._stats_grid = QTableWidget()
        self._stats_grid.setColumnCount(2)
        self._stats_grid.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_grid.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._stats_grid.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._stats_grid.setMaximumHeight(90)
        self._stats_grid.setStyleSheet("""
            QTableWidget { background-color: #2a2a2a; color: #d4d4d4; gridline-color: #3a3a3a; border: 1px solid #3a3a3a; border-radius: 3px; font-size: 8pt; }
            QTableWidget::item { padding: 2px; background: #2a2a2a; color: #d4d4d4; }
            QHeaderView::section { background-color: #2a2a2a; color: #888; border: 1px solid #3a3a3a; padding: 2px; font-size: 8pt; }
        """)
        layout.addWidget(self._stats_grid)
    
    def set_loop_status(
        self, 
        loop_detected: bool = False,
        action_name: str = None,
        consecutive_count: int = 0,
        threshold: int = 3,
        envs_with_loops: int = 0,
        total_envs: int = 8,
        history_stats: Dict = None
    ):
        """Set loop detection status and update display.
        
        Args:
            loop_detected: Whether any loops are currently active
            action_name: Name of the repeated action if detected
            consecutive_count: Number of consecutive actions
            threshold: Alert threshold value
            envs_with_loops: Number of environments in loops
            total_envs: Total number of environments
            history_stats: Optional dict with additional statistics
        """
        self._loop_detected = loop_detected
        self._loop_action_name = action_name
        self._envs_with_loops = envs_with_loops
        
        # Show/hide alert
        if loop_detected and action_name:
            self._alert_label.setVisible(True)
            self._action_name_label.setText(f"Repeated action: {action_name}")
            self._stats_label.setText(f"Consecutive: {consecutive_count} / {threshold}")
            
            # Color code based on severity
            pct = (envs_with_loops / max(total_envs, 1)) * 100
            if pct < 50:
                self._alert_frame.setStyleSheet("""
                    background-color: rgba(255, 165, 0, 0.1);
                    border: 2px solid orange;
                    border-radius: 8px;
                    padding: 15px;
                """)
                self._alert_label.setStyleSheet("color: orange;")
                self._action_name_label.setStyleSheet("color: #e65100;")
            else:
                self._alert_frame.setStyleSheet("""
                    background-color: rgba(255, 0, 0, 0.1);
                    border: 2px solid red;
                    border-radius: 8px;
                    padding: 15px;
                """)
                self._alert_label.setStyleSheet("color: red;")
                self._action_name_label.setStyleSheet("color: #d32f2f;")
            
            self._alert_spacer.setVisible(False)
        else:
            self._alert_frame.setVisible(False)
            self._alert_spacer.setVisible(True)
        
        # Update statistics grid
        self._update_stats_grid(history_stats)
    
    def _update_stats_grid(self, history_stats: Dict = None):
        self._stats_grid.setRowCount(4)
        rows_data = [
            ("Envs in Loops", str(self._envs_with_loops)),
            ("Status", "ACTIVE" if self._loop_detected else "Normal"),
            ("Action", self._loop_action_name or "--"),
            ("Threshold", str(self._consecutive_threshold)),
        ]
        self._stats_grid.clearContents()
        for i, (metric, value) in enumerate(rows_data):
            self._stats_grid.setItem(i, 0, QTableWidgetItem(metric))
            self._stats_grid.setItem(i, 1, QTableWidgetItem(str(value)))
