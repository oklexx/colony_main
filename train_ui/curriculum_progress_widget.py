#!/usr/bin/env python3
"""Curriculum Progress Widget - displays current stage and progress."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PySide6.QtCore import Qt, Property


class CurriculumProgressWidget(QWidget):
    """Widget displaying curriculum training progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Default values
        self._curriculum_stage = 0
        self._progress_percent = 0.0
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Stage badge
        self._stage_badge = QLabel("Stage: 0/3")
        self._stage_badge.setAlignment(Qt.AlignCenter)
        self._stage_badge.setStyleSheet("""
            background-color: #2196F3;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 20px;
            font-size: 11pt;
        """)
        layout.addWidget(self._stage_badge)
        
        # Progress bar container
        bar_container = QFrame()
        bar_container.setFixedHeight(25)
        bar_layout = QHBoxLayout(bar_container)
        bar_layout.addStretch()
        
        # Progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #ddd;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        bar_layout.addWidget(self._progress_bar)
        bar_layout.addStretch()
        
        layout.addWidget(bar_container)
        
        # Available actions info
        self._actions_label = QLabel("Actions: INITIALIZE | MOVE_RIGHT | BUILD_HOUSE | PRESERVE | WAIT")
        self._actions_label.setStyleSheet("""
            font-size: 9pt;
            color: #666;
            padding: 3px 0;
        """)
        layout.addWidget(self._actions_label)
        
        # Upcoming stages list
        self._upcoming_list = QListWidget()
        self._upcoming_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._upcoming_list.setMaximumHeight(80)
        self._upcoming_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                background-color: #2a2a2a;
                color: white;
                font-size: 9pt;
            }
            QListWidget::item {
                padding: 5px;
                background-color: #333;
                color: white;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background-color: #4a90d9;
                color: white;
            }
        """)
        layout.addWidget(self._upcoming_list)
    
    def update(self, progress_data: dict):
        """Update widget with curriculum progress data.
        
        Args:
            progress_data: Dict from EnvManager.get_curriculum_progress() containing:
                - stage: Current stage (0-3)
                - progress_percent: 0.0-1.0
                - available_actions: String of action names
                - next_stage_at_step: Optional step for next transition
                - upcoming_stages: List of dicts with 'stage' and 'at_step'
        """
        self._curriculum_stage = int(progress_data.get("stage", 0))
        
        # Update stage badge
        total_stages = getattr(self, "_total_stages", 3)
        self._stage_badge.setText(f"Stage: {self._curriculum_stage}/{total_stages}")
        
        # Update progress bar
        progress_percent = float(progress_data.get("progress_percent", 0.0))
        self._progress_bar.setValue(int(progress_percent * 100))
        
        # Color code based on progress
        if progress_percent < 0.3:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #ddd;
                    border: none;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background-color: orange;
                }
            """)
        elif progress_percent < 0.7:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #ddd;
                    border: none;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background-color: lime;
                }
            """)
        else:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #ddd;
                    border: none;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background-color: red;
                }
            """)
        
        # Update available actions
        actions_text = progress_data.get("available_actions", "")
        self._actions_label.setText(f"Actions: {actions_text[:60]}..." if len(actions_text) > 60 else f"Actions: {actions_text}")
        
        # Update upcoming stages list
        self._update_upcoming_stages(progress_data)
    
    def _update_upcoming_stages(self, progress_data: dict):
        """Update the upcoming stages list widget."""
        upcoming = progress_data.get("upcoming_stages", [])
        self._upcoming_list.clear()
        
        for stage_info in upcoming:
            stage_num = stage_info.get("stage", 0)
            step = stage_info.get("at_step", 0)
            
            item_text = f"Stage {stage_num}: {step:,} steps"
            if self._curriculum_stage == stage_num:
                item_text += " (current)"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, stage_num)
            self._upcoming_list.addItem(item)
