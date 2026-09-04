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
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        title = QLabel("KL Divergence")
        title.setStyleSheet("font-size: 10pt; font-weight: bold; color: #4a90d9; padding: 2px; background: transparent;")
        layout.addWidget(title)

        self._status_label = QLabel("OPTIMAL")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("""
            background-color: #2e7d32;
            color: white;
            font-weight: bold;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 10pt;
        """)
        layout.addWidget(self._status_label)

        bar_container = QFrame()
        bar_container.setFixedHeight(16)
        bar_container.setStyleSheet("background: transparent; border: none;")
        bar_layout = QHBoxLayout(bar_container)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addStretch()

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setValue(50)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedHeight(8)
        bar_layout.addWidget(self._progress_bar)
        bar_layout.addStretch()

        layout.addWidget(bar_container)

        self._info_label = QLabel("KL: 0.025 | ent_coef: 0.005")
        self._info_label.setStyleSheet("font-size: 9pt; color: #888; padding: 1px; background: transparent;")
        layout.addWidget(self._info_label)
    
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
        if kl < self.kl_low:
            self._status_label.setText("LOW")
            self._status_label.setStyleSheet("background-color: #e65100; color: white; font-weight: bold; border-radius: 4px; padding: 3px 10px; font-size: 10pt;")
        elif kl < self.kl_optimal:
            self._status_label.setText("OK")
            self._status_label.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; padding: 3px 10px; font-size: 10pt;")
        elif kl < self.kl_high:
            self._status_label.setText("OPTIMAL")
            self._status_label.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; border-radius: 4px; padding: 3px 10px; font-size: 10pt;")
        else:
            self._status_label.setText("HIGH")
            self._status_label.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; border-radius: 4px; padding: 3px 10px; font-size: 10pt;")
    
    def _update_progress_bar(self, kl: float):
        min_kl = self.kl_low
        max_kl = self.kl_high
        if max_kl > min_kl:
            progress = (kl - min_kl) / (max_kl - min_kl) * 100
            progress = max(0, min(100, progress))
        else:
            progress = 50
        self._progress_bar.setValue(int(progress))
        if kl < self.kl_low:
            color = "#e65100"
        elif kl < self.kl_optimal:
            color = "#2e7d32"
        elif kl < self.kl_high:
            color = "#1565c0"
        else:
            color = "#c62828"
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #2a2a2a; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}
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
