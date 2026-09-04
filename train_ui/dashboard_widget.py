#!/usr/bin/env python3
"""Interactive Training Dashboard - real-time plots with pyqtgraph."""

from __future__ import annotations

import numpy as np
import time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class TrainingDashboardWidget(QWidget):
    """Interactive dashboard with real-time training plots."""
    
    # Signals for data updates
    plot_data_received = Signal(dict)  # Received data point
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Data buffers (last N points)
        self._kl_buffer = []
        self._entropy_buffer = []
        self._action_buffer = []
        self._buffer_size = 1000
        
        # Plots configuration
        self._setup_plots()
        
        # Auto-scroll timer
        self._scroll_timer = QTimer()
        self._scroll_timer.timeout.connect(self._scroll_view)
        self._scroll_timer.setInterval(100)  # Update every 100ms
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        title_label = QLabel("📊 Training Dashboard")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #2196F3;
            padding: 10px;
            background-color: white;
        """)
        layout.addWidget(title_label)
        
        # Splitter for plots
        splitter = QSplitter(Qt.Horizontal)
        self._add_plot_to_splitter(splitter, "KL Divergence", "kl_plot")
        self._add_plot_to_splitter(splitter, "Entropy", "entropy_plot")
        self._add_plot_to_splitter(splitter, "Action Distribution", "action_plot")
        
        layout.addWidget(splitter)
    
    def _add_plot_to_splitter(self, splitter, title: str, plot_name: str):
        """Add a plot to the splitter."""
        # Create plot area
        plot_frame = QFrame()
        plot_frame.setStyleSheet("background-color: #fafafa; border-radius: 8px;")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 12pt;
            font-weight: bold;
            padding: 8px;
            background-color: #e3f2fd;
            border-radius: 4px;
        """)
        plot_layout.addWidget(title_label)
        
        # Plot widget
        self._plot_widget = pg.PlotWidget(
            title=title,
            background='w',
            showGrid=(True, False),
        )
        self._plot_widget.setMouseEnabled(x=True, y=True)  # Enable zoom/pan
        
        plot_layout.addWidget(self._plot_widget)
        
        splitter.addWidget(plot_frame)
    
    def _setup_plots(self):
        """Configure all plots."""
        self._kl_curve = self._plot_widget.addGraph(
            pen='b',
            symbol='o',
            symbolSize=3,
            symbolicPen={'width': 1}
        )
        self._kl_curve.setPen('b')
        
        # KL target line at 0.05
        self._target_line = pg.InfiniteLine(pos=0.05, angle=90, pen='r', movable=False)
        self._plot_widget.addItem(self._target_line)
        
        self._entropy_curve = self._plot_widget.addGraph(
            pen='g',
            symbol='o',
            symbolSize=3,
            symbolicPen={'width': 1}
        )
        self._entropy_curve.setPen('g')
        
        # Action distribution plot (separate widget)
        action_plot = pg.PlotWidget(
            title="Action Distribution",
            background='w',
            showGrid=(True, False),
        )
        
        # X-axis for 10 actions
        action_plot.setLabel('left', 'Frequency (%)', units='')
        action_plot.setLabel('bottom', 'Action Index', '')
        
        self._action_bar = action_plot.addBarGraph(
            y='[0]',
            pen='r',
            width=0.8,
            brush=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'teal']
        )
        
        action_plot.setYRange(0, 100)
        action_plot.setXRange(-0.5, 9.5)
        
        # Add to layout directly
        self._plot_widget.layout().addWidget(action_plot)
    
    def update_data(self, kl: float = None, entropy: float = None, 
                    top_actions: dict = None):
        """Update dashboard with new data.
        
        Args:
            kl: Current KL divergence value
            entropy: Current entropy value
            top_actions: Dict of action name -> percentage (top 5)
        """
        self._update_kl_plot(kl)
        self._update_entropy_plot(entropy)
        self._update_action_plot(top_actions)
    
    def _update_kl_plot(self, kl: float):
        """Update KL divergence plot."""
        if kl is not None and not np.isnan(kl):
            # Add to buffer
            self._kl_buffer.append((time.time(), kl))
            
            # Keep only last N points
            if len(self._kl_buffer) > self._buffer_size:
                self._kl_buffer.pop(0)
            
            # Update curve
            x = np.array([t for t, v in self._kl_buffer])
            y = np.array([v for t, v in self._kl_buffer])
            
            if len(x) > 1:
                self._kl_curve.setData(x=x, y=y, pen='b', symbol='o')
    
    def _update_entropy_plot(self, entropy: float):
        """Update entropy plot."""
        if entropy is not None and not np.isnan(entropy):
            # Add to buffer
            self._entropy_buffer.append((time.time(), entropy))
            
            # Keep only last N points
            if len(self._entropy_buffer) > self._buffer_size:
                self._entropy_buffer.pop(0)
            
            # Update curve
            x = np.array([t for t, v in self._entropy_buffer])
            y = np.array([v for t, v in self._entropy_buffer])
            
            if len(x) > 1:
                self._entropy_curve.setData(x=x, y=y, pen='g', symbol='o')
    
    def _update_action_plot(self, top_actions: dict):
        """Update action distribution bar chart."""
        if top_actions is None or not top_actions:
            return
        
        # Convert dict to list of values (already percentages from AsyncTrainer)
        values = list(top_actions.values())
        
        # Update bar graph
        if hasattr(self, '_action_bar') and self._action_bar:
            self._action_bar.setValues(y=values[:10])  # Support up to 10 actions
    
    def _scroll_view(self):
        """Auto-scroll plots to show recent data."""
        for plot in [self._kl_curve, self._entropy_curve]:
            if plot and hasattr(plot, 'viewBox'):
                plot.setXRange(
                    time.time() - 60,  # Show last 60 seconds
                    time.time(),
                    padding=0,
                    asymmetric=False
                )
