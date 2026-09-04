#!/usr/bin/env python3
"""KL Status Widget - displays KL divergence status with color coding."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Property


class KLStatusWidget(QWidget):
    """Widget displaying KL divergence status for PPO training."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Default ranges
        self.kl_low = 0.01      # LOW status
        self.kl_optimal = 0.25  # OPTIMAL status  
        self.kl_high = 0.08     # HIGH status (threshold above optimal)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Status indicator label
        self._status_label = QLabel("OPTIMAL")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("""
            background-color: lime;
            color: black;
            font-weight: bold;
            border-radius: 4px;
            padding: 5px 15px;
            font-size: 12pt;
        """)
        layout.addWidget(self._status_label)
        
        # Progress bar container
        bar_container = QFrame()
        bar_container.setFixedHeight(20)
        bar_layout = QHBoxLayout(bar_container)
        bar_layout.addStretch()
        
        # Progress bar with custom colors
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setValue(50)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setRange(0, 100)
        bar_layout.addWidget(self._progress_bar)
        bar_layout.addStretch()
        
        # Info label with exact values
        self._info_label = QLabel("KL: 0.025 | ent_coef: 0.005")
        self._info_label.setStyleSheet("""
            font-size: 10pt;
            color: #888;
            padding: 3px 0;
        """)
        bar_layout.addWidget(self._info_label)
        
        layout.addWidget(bar_container)
    
    def update(self, kl: float, ent_coef: float = None):
        """Update widget with new KL value.
        
        Args:
            kl: Current KL divergence value
            ent_coef: Entropy coefficient (optional)
        """
        self._update_status(kl)
        self._update_progress_bar(kl)
        self._update_info_label(kl, ent_coef)
    
    def _update_status(self, kl: float):
        """Update status label with color coding."""
        if kl < self.kl_low:
            self._status_label.setText("LOW")
            self._status_label.setStyleSheet("""
                background-color: orange;
                color: black;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 12pt;
            """)
        elif kl < self.kl_optimal:
            self._status_label.setText("OK")
            self._status_label.setStyleSheet("""
                background-color: lime;
                color: black;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 12pt;
            """)
        elif kl < self.kl_high:
            self._status_label.setText("OPTIMAL")
            self._status_label.setStyleSheet("""
                background-color: turquoise;
                color: black;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 12pt;
            """)
        else:
            self._status_label.setText("HIGH")
            self._status_label.setStyleSheet("""
                background-color: red;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 12pt;
            """)
    
    def _update_progress_bar(self, kl: float):
        """Update progress bar (0.01-0.8 normalized to 0-100%)."""
        # Normalize KL value to progress range
        min_kl = self.kl_low
        max_kl = self.kl_high
        
        if max_kl > min_kl:
            progress = (kl - min_kl) / (max_kl - min_kl) * 100
            # Cap at 0-100%
            progress = max(0, min(100, progress))
        else:
            progress = 50
        
        self._progress_bar.setValue(int(progress))
        
        # Color code the bar based on status
        if kl < self.kl_low:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: orange;
                    border: none;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: orange;
                }
            """)
        elif kl < self.kl_optimal:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: lime;
                    border: none;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: lime;
                }
            """)
        elif kl < self.kl_high:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: turquoise;
                    border: none;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: turquoise;
                }
            """)
        else:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: red;
                    border: none;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: red;
                }
            """)
    
    def _update_info_label(self, kl: float, ent_coef: float = None):
        """Update info label with exact values."""
        if ent_coef is None:
            ent_coef_str = "N/A"
        else:
            ent_coef_str = f"{ent_coef:.4f}"
        
        self._info_label.setText(f"KL: {kl:.5f} | ent_coef: {ent_coef_str}")
    
    def set_thresholds(self, low: float, optimal: float, high: float):
        """Set custom threshold values.
        
        Args:
            low: KL value for LOW status (< this)
            optimal: KL value for OPTIMAL start (>= this and < high)
            high: KL value for HIGH status (>= this)
        """
        self.kl_low = max(0, low)
        self.kl_optimal = max(self.kl_low + 1e-6, optimal)
        self.kl_high = max(self.kl_optimal + 1e-6, high)
