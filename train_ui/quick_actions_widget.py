#!/usr/bin/env python3
"""Quick Actions Widget - panel for manual control during training."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QMessageBox, QDialog, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Property
from typing import Dict, Optional, Callable


class QuickActionsWidget(QWidget):
    """Widget with quick control buttons for training."""
    
    # Signals for command execution
    cmd_boost_entropy = Signal(str, dict)  # ent_coef multiplier as string
    cmd_pause_training = Signal(bool)       # True to pause
    cmd_resume_training = Signal()          # Resume from paused
    cmd_stop_training = Signal(bool)        # final_save flag
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # State tracking
        self._training_paused = False
        self._boost_history: Dict[int, float] = {}  # step -> boost factor
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_label = QLabel("Quick Actions")
        title_label.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            padding-bottom: 10px;
            border-bottom: 2px solid #2196F3;
        """)
        layout.addWidget(title_label)
        
        # Pause/Resume section
        pause_frame = QFrame()
        pause_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 8px; padding: 10px;")
        pause_layout = QVBoxLayout(pause_frame)
        
        status_label = QLabel("Training: RUNNING")
        status_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #4caf50;
            padding: 8px;
            background-color: white;
            border-radius: 4px;
        """)
        pause_layout.addWidget(status_label)
        
        pause_resume_buttons = QHBoxLayout()
        
        self._pause_button = QPushButton("⏸️ Pause Training")
        self._pause_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        pause_resume_buttons.addWidget(self._pause_button)
        
        self._resume_button = QPushButton("▶️ Resume Training")
        self._resume_button.setEnabled(False)
        self._resume_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self._resume_button.clicked.connect(self._on_resume_clicked)
        pause_resume_buttons.addWidget(self._resume_button)
        
        pause_layout.addLayout(pause_resume_buttons)
        layout.addWidget(pause_frame)
        
        # Entropy Boost section
        boost_frame = QFrame()
        boost_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 8px; padding: 10px;")
        boost_layout = QVBoxLayout(boost_frame)
        
        boost_label = QLabel("⚡ Boost Entropy ×2")
        boost_label.setStyleSheet("""
            font-size: 12pt;
            font-weight: bold;
            color: #ff5722;
            padding-bottom: 8px;
        """)
        boost_layout.addWidget(boost_label)
        
        self._boost_button = QPushButton("Boost Entropy (×2)")
        self._boost_button.setStyleSheet("""
            QPushButton {
                background-color: #ff5722;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e64a19;
            }
        """)
        self._boost_button.clicked.connect(self._on_boost_clicked)
        boost_layout.addWidget(self._boost_button)
        
        layout.addWidget(boost_frame)
        
        # Reset Curriculum section
        reset_frame = QFrame()
        reset_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 8px; padding: 10px;")
        reset_layout = QVBoxLayout(reset_frame)
        
        reset_label = QLabel("🔄 Reset Curriculum")
        reset_label.setStyleSheet("""
            font-size: 12pt;
            font-weight: bold;
            color: #9c27b0;
            padding-bottom: 8px;
        """)
        reset_layout.addWidget(reset_label)
        
        self._reset_button = QPushButton("Reset to Stage 0")
        self._reset_button.setStyleSheet("""
            QPushButton {
                background-color: #9c27b0;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7b1fa2;
            }
        """)
        self._reset_button.clicked.connect(self._on_reset_clicked)
        reset_layout.addWidget(self._reset_button)
        
        layout.addWidget(reset_frame)
    
    def _on_pause_clicked(self):
        """Handle pause button click."""
        reply = QMessageBox.question(
            self,
            "Pause Training?",
            "Are you sure you want to pause training?\n\n"
            "The model weights will be preserved and can be resumed later.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.cmd_pause_training.emit(True)
            self._training_paused = True
            self._pause_button.setEnabled(False)
            self._resume_button.setEnabled(True)
            
            status_label = self.findChild(QLabel, "status_label")
            if status_label:
                status_label.setText("Training: PAUSED")
                status_label.setStyleSheet("""
                    font-size: 11pt;
                    font-weight: bold;
                    color: #ff9800;
                    padding: 8px;
                    background-color: white;
                    border-radius: 4px;
                """)
    
    def _on_resume_clicked(self):
        """Handle resume button click."""
        self.cmd_resume_training.emit()
        self._training_paused = False
        self._pause_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        
        status_label = self.findChild(QLabel, "status_label")
        if status_label:
            status_label.setText("Training: RUNNING")
            status_label.setStyleSheet("""
                font-size: 11pt;
                font-weight: bold;
                color: #4caf50;
                padding: 8px;
                background-color: white;
                border-radius: 4px;
            """)
    
    def _on_boost_clicked(self):
        """Handle entropy boost button click."""
        self._send_command("boost_entropy", {"multiplier": 2.0})
        
        # Visual feedback
        msg = QMessageBox()
        msg.setWindowTitle("Entropy Boosted")
        msg.setText("Entropy coefficient doubled for this rollout.")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    
    def _on_reset_clicked(self):
        """Handle curriculum reset button click."""
        self._send_command("reset_curriculum", {"stage": 0})
        
        # Visual feedback
        msg = QMessageBox()
        msg.setWindowTitle("Curriculum Reset")
        msg.setText("Training has been reset to Stage 0 (all buildings unlocked).")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
    
    def _send_command(self, cmd: str, payload: Optional[Dict] = None):
        """Send a command via the signal mechanism.
        
        Args:
            cmd: Command name (boost_entropy, pause, resume, reset_curriculum)
            payload: Optional payload dictionary
        """
        payload = payload or {}
        
        if cmd == "pause_training":
            self.cmd_pause_training.emit(payload.get("paused", True))
        elif cmd == "resume_training":
            self.cmd_resume_training.emit()
        elif cmd == "stop_training":
            final_save = payload.get("final_save", True)
            self.cmd_stop_training.emit(final_save)
        elif cmd == "boost_entropy":
            multiplier = payload.get("multiplier", 2.0)
            self.cmd_boost_entropy.emit(str(multiplier), payload)
        elif cmd == "reset_curriculum":
            stage = payload.get("stage", 0)
            self._send_command("reset_curriculum", {"stage": stage})

    def set_training_state(self, is_running: bool):
        """Update UI based on training state.
        
        Args:
            is_running: True if training is actively running
        """
        self._training_paused = not is_running
        
        # Update button states
        self._pause_button.setEnabled(is_running)
        self._resume_button.setEnabled(not is_running and self._training_paused)
        self._boost_button.setEnabled(is_running)
        self._reset_button.setEnabled(is_running)
