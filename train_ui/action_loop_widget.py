#!/usr/bin/env python3
"""Action Loop Widget - displays loop detection alerts and statistics."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtCore import Qt, Property
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
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Alert label (hidden by default)
        alert_frame = QFrame()
        alert_frame.setVisible(False)
        alert_frame.setStyleSheet("""
            background-color: rgba(255, 0, 0, 0.1);
            border: 2px solid red;
            border-radius: 8px;
            padding: 15px;
        """)
        
        alert_layout = QVBoxLayout(alert_frame)
        
        # Alert icon/text
        self._alert_label = QLabel("⚠️ ACTION LOOP DETECTED")
        self._alert_label.setStyleSheet("""
            font-size: 14pt;
            font-weight: bold;
            color: red;
        """)
        alert_layout.addWidget(self._alert_label)
        
        # Action name display
        self._action_name_label = QLabel("Repeated action: MOVE_RIGHT")
        self._action_name_label.setStyleSheet("""
            font-size: 12pt;
            font-weight: bold;
            color: #d32f2f;
            padding: 5px 0;
        """)
        alert_layout.addWidget(self._action_name_label)
        
        # Consecutive count and threshold
        self._stats_label = QLabel(f"Consecutive: 10 / {self._consecutive_threshold}")
        self._stats_label.setStyleSheet("""
            font-size: 10pt;
            color: #c62828;
        """)
        alert_layout.addWidget(self._stats_label)
        
        layout.addWidget(alert_frame)
        
        # Hidden spacer (shown when no alert)
        self._alert_spacer = QLabel(" ")
        self._alert_spacer.setAlignment(Qt.AlignCenter)
        self._alert_spacer.setVisible(True)
        self._alert_spacer.setStyleSheet("""
            background-color: transparent;
            border: none;
            padding: 15px;
        """)
        layout.addWidget(self._alert_spacer)
        
        # Statistics grid
        stats_frame = QFrame()
        stats_frame.setVisible(True)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        stats_title = QLabel("Loop Detection Statistics")
        stats_title.setStyleSheet("""
            font-weight: bold;
            font-size: 11pt;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        """)
        stats_layout.addWidget(stats_title)
        
        # Stats grid
        self._stats_grid = QTableWidget()
        self._stats_grid.setColumnCount(3)
        self._stats_grid.setHorizontalHeaderLabels(["Metric", "Value", "Threshold"])
        self._stats_grid.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._stats_grid.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._stats_grid.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._stats_grid.setStyleSheet("""
            QTableWidget {
                gridline-color: #eee;
                alternate-background-color: #f9f9f9;
            }
            QTableWidgetItem {
                padding: 5px;
                border: none;
            }
        """)
        stats_layout.addWidget(self._stats_grid)
        
        layout.addWidget(stats_frame)
    
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
                alert_frame.setStyleSheet("""
                    background-color: rgba(255, 165, 0, 0.1);
                    border: 2px solid orange;
                    border-radius: 8px;
                    padding: 15px;
                """)
                self._alert_label.setStyleSheet("color: orange;")
                self._action_name_label.setStyleSheet("color: #e65100;")
            else:
                alert_frame.setStyleSheet("""
                    background-color: rgba(255, 0, 0, 0.1);
                    border: 2px solid red;
                    border-radius: 8px;
                    padding: 15px;
                """)
                self._alert_label.setStyleSheet("color: red;")
                self._action_name_label.setStyleSheet("color: #d32f2f;")
            
            self._alert_spacer.setVisible(False)
        else:
            alert_frame.setVisible(False)
            self._alert_spacer.setVisible(True)
        
        # Update statistics grid
        self._update_stats_grid(history_stats)
    
    def _update_stats_grid(self, history_stats: Dict = None):
        """Update the statistics table with current data."""
        self._stats_grid.setRowCount(5)
        
        rows_data = [
            ("Envs in Loops", f"{self._envs_with_loops}", f"/ {self._stats_grid.columnCount() > 2 and True or ''}"),
            ("Loop Rate", f"{(self._envs_with_loops * 100 / max(8, self._envs_with_loops + (8-self._envs_with_loops))):.1f}%", "N/A"),
            ("Threshold", f"{self._consecutive_threshold}", ""),
            ("Status", "ACTIVE" if self._loop_detected else "NORMAL", ""),
            ("Action", self._loop_action_name or "N/A", "")
        ]
        
        # Clear existing rows
        for row in range(self._stats_grid.rowCount()):
            for col in range(self._stats_grid.columnCount()):
                item = self._stats_grid.item(row, col)
                if item:
                    self._stats_grid.removeItem(item)
        
        # Add new rows
        for i, (metric, value, threshold) in enumerate(rows_data[:self._stats_grid.rowCount()]):
            self._stats_grid.setItem(i, 0, QTableWidgetItem(metric))
            self._stats_grid.setItem(i, 1, QTableWidgetItem(str(value)))
            if threshold:
                self._stats_grid.setItem(i, 2, QTableWidgetItem(str(threshold)))
