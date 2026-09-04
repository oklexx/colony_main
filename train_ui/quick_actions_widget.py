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
    cmd_boost_entropy = Signal(str, dict)
    cmd_pause_training = Signal(bool)
    cmd_resume_training = Signal()
    cmd_stop_training = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_paused = False
        self._boost_history: Dict[int, float] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title_label = QLabel("Quick Actions")
        title_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #4a90d9;
            padding: 2px;
            border-bottom: 1px solid #3a3a3a;
        """)
        layout.addWidget(title_label)

        # Pause/Resume
        self._status_label = QLabel("Status: IDLE")
        self._status_label.setStyleSheet("font-size: 9pt; color: #888; padding: 2px; background: transparent;")
        layout.addWidget(self._status_label)

        btn_style = """
            QPushButton {
                background-color: #3a3a3a;
                color: #d4d4d4;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 9pt;
                min-height: 20px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { color: #666; background-color: #2a2a2a; }
        """

        pr_layout = QHBoxLayout()
        self._pause_button = QPushButton("Pause")
        self._pause_button.setStyleSheet(btn_style + "QPushButton { background-color: #e65100; color: white; } QPushButton:hover { background-color: #bf360c; }")
        self._pause_button.clicked.connect(self._on_pause_clicked)
        pr_layout.addWidget(self._pause_button)

        self._resume_button = QPushButton("Resume")
        self._resume_button.setEnabled(False)
        self._resume_button.setStyleSheet(btn_style + "QPushButton { background-color: #1565c0; color: white; } QPushButton:hover { background-color: #0d47a1; }")
        self._resume_button.clicked.connect(self._on_resume_clicked)
        pr_layout.addWidget(self._resume_button)
        layout.addLayout(pr_layout)

        # Boost + Reset
        rr_layout = QHBoxLayout()
        self._boost_button = QPushButton("Boost Entropy x2")
        self._boost_button.setStyleSheet(btn_style + "QPushButton { background-color: #bf360c; color: white; } QPushButton:hover { background-color: #8b2500; }")
        self._boost_button.clicked.connect(self._on_boost_clicked)
        rr_layout.addWidget(self._boost_button)

        self._reset_button = QPushButton("Reset Stage 0")
        self._reset_button.setStyleSheet(btn_style + "QPushButton { background-color: #6a1b9a; color: white; } QPushButton:hover { background-color: #4a148c; }")
        self._reset_button.clicked.connect(self._on_reset_clicked)
        rr_layout.addWidget(self._reset_button)
        layout.addLayout(rr_layout)

    def _log(self, msg: str):
        print(f"[QuickActions] {msg}")

    def _on_pause_clicked(self):
        reply = QMessageBox.question(
            self,
            "Pause Training?",
            "Are you sure you want to pause training?\n\n"
            "The model weights will be preserved and can be resumed later.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._log("Pause training requested")
            self.cmd_pause_training.emit(True)
            self._training_paused = True
            self._pause_button.setEnabled(False)
            self._resume_button.setEnabled(True)
            self._status_label.setText("Training: PAUSED")
            self._status_label.setStyleSheet("""
                font-size: 11pt;
                font-weight: bold;
                color: #ff9800;
                padding: 8px;
                background-color: white;
                border-radius: 4px;
            """)

    def _on_resume_clicked(self):
        self._log("Resume training requested")
        self.cmd_resume_training.emit()
        self._training_paused = False
        self._pause_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self._status_label.setText("Training: RUNNING")
        self._status_label.setStyleSheet("""
            font-size: 11pt;
            font-weight: bold;
            color: #4caf50;
            padding: 8px;
            background-color: white;
            border-radius: 4px;
        """)

    def _on_boost_clicked(self):
        self._log("Boost entropy x2 sent to worker")
        self._send_command("boost_entropy", {"multiplier": 2.0})

    def _on_reset_clicked(self):
        self._log("Reset curriculum to stage 0 sent to worker")
        self._send_command("reset_curriculum", {"stage": 0})

    def _send_command(self, cmd: str, payload: Optional[Dict] = None):
        payload = payload or {}

        if cmd == "boost_entropy":
            multiplier = payload.get("multiplier", 2.0)
            self.cmd_boost_entropy.emit(str(multiplier), payload)
        elif cmd == "reset_curriculum":
            stage = payload.get("stage", 0)
            self.cmd_boost_entropy.emit("reset_curriculum", {"stage": stage})
        elif cmd == "pause_training":
            self.cmd_pause_training.emit(payload.get("paused", True))
        elif cmd == "resume_training":
            self.cmd_resume_training.emit()
        elif cmd == "stop_training":
            final_save = payload.get("final_save", True)
            self.cmd_stop_training.emit(final_save)

    def set_training_state(self, is_running: bool):
        self._training_paused = not is_running
        self._pause_button.setEnabled(is_running)
        self._resume_button.setEnabled(not is_running and self._training_paused)
        self._boost_button.setEnabled(is_running)
        self._reset_button.setEnabled(is_running)
        self._log(f"Training state updated: running={is_running}")
