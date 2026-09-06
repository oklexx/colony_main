#!/usr/bin/env python3
"""Return Statistics Widget - displays episode return statistics."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QTableWidget, QTableWidgetItem,
    QMessageBox, QToolTip
)
from PySide6.QtCore import Qt, Property, QPoint
from PySide6.QtGui import QCursor
from typing import Dict, List, Optional


class ReturnStatisticsWidget(QWidget):
    """Widget displaying episode return statistics with histogram."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Statistics storage
        self._avg_return = 0.0
        self._median_return = 0.0
        self._max_return = 0.0
        self._min_return = 0.0
        
        # History for trends
        self._return_history: List[float] = []
        self._history_size = 100
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(3)
        title = QLabel("Return Statistics")
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
            "Статистика наград (return) за эпизоды.\n"
            "• Avg — средняя награда за последние эпизоды\n"
            "• Median — медианная награда (менее чувствительна к выбросам)\n"
            "• Max — максимальная достигнутая награда\n"
            "• Min — минимальная награда\n"
            "Показывает стабильность обучения агента.",
            help_btn))
        title_row.addWidget(help_btn)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self._avg_label = self._create_stat_label("Avg:")
        self._med_label = self._create_stat_label("Median:")
        self._max_label = self._create_stat_label("Max:")
        self._min_label = self._create_stat_label("Min:")

        layout.addWidget(self._avg_label)
        layout.addWidget(self._med_label)
        layout.addWidget(self._max_label)
        layout.addWidget(self._min_label)

    def _create_stat_label(self, prefix: str) -> QLabel:
        label = QLabel(f"{prefix} --")
        label.setStyleSheet("font-size: 9pt; padding: 2px 4px; color: #d4d4d4; background: transparent;")
        return label
    
    def update_statistics(
        self,
        avg: float = None,
        median: float = None,
        max_val: float = None,
        min_val: float = None,
        returns: List[float] = None,
        episode_count: int = 0
    ):
        """Update return statistics.
        
        Args:
            avg: Average return (calculated if not provided)
            median: Median return (calculated if not provided)
            max_val: Maximum return (calculated if not provided)
            min_val: Minimum return (calculated if not provided)
            returns: List of all returns for statistics calculation
            episode_count: Current episode count
        """
        # Calculate statistics from returns if not provided
        if returns is not None and len(returns) > 0:
            avg = sum(returns) / len(returns)
            
            sorted_returns = sorted(returns)
            n = len(sorted_returns)
            median = sorted_returns[n // 2] if n % 2 == 1 else (sorted_returns[n//2 - 1] + sorted_returns[n//2]) / 2
            
            max_val = max(returns)
            min_val = min(returns)
        
        # Store values
        self._avg_return = avg or 0.0
        self._median_return = median or 0.0
        self._max_return = max_val or 0.0
        self._min_return = min_val or 0.0
        
        # Update display
        self._update_display()
    
    def _update_display(self):
        self._avg_label.setText(f"Avg: {self._avg_return:.2f}")
        self._avg_label.setStyleSheet(f"font-size: 9pt; padding: 2px 4px; color: #81c784; background: transparent;")

        self._med_label.setText(f"Median: {self._median_return:.2f}")
        self._med_label.setStyleSheet("font-size: 9pt; padding: 2px 4px; color: #d4d4d4; background: transparent;")

        self._max_label.setText(f"Max: {self._max_return:.2f}")
        self._max_label.setStyleSheet(f"font-size: 9pt; padding: 2px 4px; color: #64b5f6; background: transparent;")

        self._min_label.setText(f"Min: {self._min_return:.2f}")
        self._min_label.setStyleSheet("font-size: 9pt; padding: 2px 4px; color: #e57373; background: transparent;")
    
    def add_return(self, return_value: float):
        """Add a single return value to history."""
        self._return_history.append(return_value)
        
        # Keep only last N values
        if len(self._return_history) > self._history_size:
            self._return_history.pop(0)
        
        # Recalculate statistics periodically (every 10 new values)
        if len(self._return_history) % 10 == 0:
            self.update_statistics(returns=self._return_history.copy())
