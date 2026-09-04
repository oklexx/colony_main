#!/usr/bin/env python3
"""Return Statistics Widget - displays episode return statistics."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QTableWidget, QTableWidgetItem,
    QMessageBox
)
from PySide6.QtCore import Qt, Property
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
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Statistics panel
        stats_frame = QFrame()
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(10, 5, 10, 5)
        
        # Labels for each statistic
        self._avg_label = self._create_stat_label("Average Return:")
        self._med_label = self._create_stat_label("Median Return:")
        self._max_label = self._create_stat_label("Maximum Return:")
        self._min_label = self._create_stat_label("Minimum Return:")
        
        stats_layout.addWidget(self._avg_label)
        stats_layout.addWidget(self._med_label)
        stats_layout.addWidget(self._max_label)
        stats_layout.addWidget(self._min_label)
        
        layout.addWidget(stats_frame)
    
    def _create_stat_label(self, prefix: str) -> QLabel:
        """Create a label for displaying a statistic."""
        label = QLabel(f"{prefix} {0:.1f}")
        label.setStyleSheet("""
            font-size: 12pt;
            padding: 5px;
            border-radius: 4px;
            background-color: #e3f2fd;
        """)
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
        """Update the widget display with current statistics."""
        # Average label (highlighted)
        avg_text = f"{self._avg_return:.2f}"
        if self._avg_return >= 10:
            style = "background-color: #4caf50;"
        else:
            style = "background-color: #ffcdd2;"
        
        self._avg_label.setText(f"Average Return: {avg_text}")
        self._avg_label.setStyleSheet(f"""
            font-size: 12pt;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        """)
        
        # Median label
        med_text = f"{self._median_return:.2f}"
        self._med_label.setText(f"Median Return: {med_text}")
        self._med_label.setStyleSheet("font-size: 11pt; padding: 5px; background-color: #e3f2fd;")
        
        # Max label (highlighted in green if good)
        max_text = f"{self._max_return:.2f}"
        if self._max_return >= 15:
            style_max = "background-color: #4caf50;"
        elif self._max_return >= 8:
            style_max = "background-color: #8bc34a;"
        else:
            style_max = "background-color: #ffcdd2;"
        
        self._max_label.setText(f"Maximum Return: {max_text}")
        self._max_label.setStyleSheet("font-size: 11pt; padding: 5px;")
        self._max_label.setStyleSheet(f"""
            font-size: 11pt;
            padding: 5px;
            background-color: {style_max};
        """)
        
        # Min label
        min_text = f"{self._min_return:.2f}"
        self._min_label.setText(f"Minimum Return: {min_text}")
        self._min_label.setStyleSheet("font-size: 11pt; padding: 5px; background-color: #e3f2fd;")
    
    def add_return(self, return_value: float):
        """Add a single return value to history."""
        self._return_history.append(return_value)
        
        # Keep only last N values
        if len(self._return_history) > self._history_size:
            self._return_history.pop(0)
        
        # Recalculate statistics periodically (every 10 new values)
        if len(self._return_history) % 10 == 0:
            self.update_statistics(returns=self._return_history.copy())
