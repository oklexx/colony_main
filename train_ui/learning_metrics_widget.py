#!/usr/bin/env python3
"""Learning Metrics Widget - displays training progress and performance metrics."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QProgressBar,
    QSizePolicy
)
from PySide6.QtCore import Qt
from typing import Dict, Optional


class LearningMetricsWidget(QWidget):
    """Widget displaying learning progress and performance metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Metrics storage
        self._episode_count = 0
        self._total_reward = 0.0
        self._avg_reward = 0.0
        self._best_reward = 0.0
        self._current_step = 0
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Main metrics panel with title
        metrics_frame = QFrame()
        metrics_frame.setObjectName("learning_metrics_frame")
        metrics_frame.setStyleSheet("""
            QFrame#learning_metrics_frame {
                background-color: #f5f5f5;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        frame_layout = QVBoxLayout(metrics_frame)
        frame_layout.setContentsMargins(12, 12, 12, 12)
        frame_layout.setSpacing(8)
        
        # Title label (visible title)
        self._title_label = QLabel("📈 Learning Metrics")
        self._title_label.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            color: #1976D2;
            padding-bottom: 8px;
            border-bottom: 2px solid #BBDEFB;
        """)
        frame_layout.addWidget(self._title_label)
        
        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(0, 8, 0, 0)
        metrics_layout.setSpacing(8)
        
        # Metrics grid layout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        # Episode counter
        self._episodes_label = QLabel("Episodes: 0")
        self._episodes_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #424242;
            padding: 6px 10px;
            background-color: white;
            border-radius: 4px;
        """)
        grid_layout.addWidget(self._episodes_label, 0, 0)
        
        # Current step counter
        self._steps_label = QLabel("Step: 0")
        self._steps_label.setStyleSheet(self._episodes_label.styleSheet())
        grid_layout.addWidget(self._steps_label, 0, 1)
        
        # Total reward
        self._total_reward_label = QLabel("Total Reward: 0.0")
        self._total_reward_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #2E7D32;
            padding: 6px 10px;
            background-color: #E8F5E9;
            border-radius: 4px;
        """)
        grid_layout.addWidget(self._total_reward_label, 1, 0)
        
        # Average reward
        self._avg_reward_label = QLabel("Avg Reward: 0.0")
        self._avg_reward_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #1565C0;
            padding: 8px 12px;
            background-color: #E3F2FD;
            border-radius: 4px;
        """)
        grid_layout.addWidget(self._avg_reward_label, 1, 1)
        
        # Best reward (highlighted)
        self._best_reward_label = QLabel("Best Reward: 0.0")
        self._best_reward_label.setStyleSheet("""
            font-size: 13pt;
            font-weight: bold;
            color: white;
            padding: 10px 15px;
            background-color: #4CAF50;
            border-radius: 6px;
        """)
        grid_layout.addWidget(self._best_reward_label, 2, 0, 1, 2)
        
        frame_layout.addLayout(metrics_layout)
        
        layout.addWidget(metrics_frame)
    
    def update_metrics(
        self, 
        episodes: int = None,
        steps: int = None,
        total_reward: float = None,
        avg_reward: float = None,
        best_reward: float = None
    ):
        """Update learning metrics.
        
        Args:
            episodes: Current episode count
            steps: Current training step
            total_reward: Cumulative reward
            avg_reward: Average reward per episode
            best_reward: Best single-episode reward achieved
        """
        if episodes is not None:
            self._episode_count = episodes
        if steps is not None:
            self._current_step = steps
        if total_reward is not None:
            self._total_reward = total_reward
        if avg_reward is not None:
            self._avg_reward = avg_reward
        if best_reward is not None:
            self._best_reward = best_reward
        
        self._update_display()
    
    def _update_display(self):
        """Update all metric labels with current values."""
        # Episodes counter
        self._episodes_label.setText(f"Episodes: {self._episode_count:,}")
        
        # Steps counter
        self._steps_label.setText(f"Step: {self._current_step:,}")
        
        # Total reward (green if positive)
        if self._total_reward >= 0:
            color = "#2E7D32"
            bg_color = "#E8F5E9"
        else:
            color = "#C62828"
            bg_color = "#FFEBEE"
        
        self._total_reward_label.setText(f"Total Reward: {self._total_reward:.1f}")
        self._total_reward_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: bold;
            color: {color};
            padding: 6px 10px;
            background-color: {bg_color};
            border-radius: 4px;
        """)
        
        # Average reward
        if self._avg_reward >= 0:
            color = "#1565C0"
            bg_color = "#E3F2FD"
        else:
            color = "#C62828"
            bg_color = "#FFEBEE"
        
        self._avg_reward_label.setText(f"Avg Reward: {self._avg_reward:.1f}")
        self._avg_reward_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: bold;
            color: {color};
            padding: 8px 12px;
            background-color: {bg_color};
            border-radius: 4px;
        """)
        
        # Best reward (always highlighted in green)
        self._best_reward_label.setText(f"Best Reward: {self._best_reward:.1f}")
        self._best_reward_label.setStyleSheet("""
            font-size: 13pt;
            font-weight: bold;
            color: white;
            padding: 10px 15px;
            background-color: #4CAF50;
            border-radius: 6px;
        """)
