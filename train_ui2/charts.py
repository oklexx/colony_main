"""Lightweight QPainter charts for UI 2.0 (no matplotlib dependency).

Chart keeps rolling history per series, auto-scales, draws grid + legend +
last values. Designed for ~120px tall panels stacked in the Monitoring tab.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from train_ui2.theme import DIM, FIELD, LINE, color_for


class Chart(QWidget):
    """Multi-series rolling line chart."""

    def __init__(self, title: str = "", maxlen: int = 400, height: int = 130,
                 y_zero_line: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.maxlen = maxlen
        self.series: Dict[str, deque] = {}
        self.y_zero_line = y_zero_line
        self.setMinimumHeight(height)
        self.setMaximumHeight(height + 40)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())

    def add_series(self, name: str, color: Optional[QColor] = None):
        if name not in self.series:
            self.series[name] = deque(maxlen=self.maxlen)
        return color or color_for(name)

    def push(self, values: Dict[str, float]):
        """Append one sample for each named series (missing keys -> gap)."""
        if not values:
            return
        for name, dq in self.series.items():
            v = values.get(name)
            try:
                v = float(v)
                if v != v or v in (float("inf"), float("-inf")):
                    v = None
            except (TypeError, ValueError):
                v = None
            dq.append(v)
        self.update()

    def clear(self):
        for dq in self.series.values():
            dq.clear()
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(FIELD))

        pad_l, pad_r, pad_t, pad_b = 6, 6, 16 if self.title else 6, 14
        plot = QRectF(pad_l, pad_t, max(10, w - pad_l - pad_r),
                      max(10, h - pad_t - pad_b))

        # collect range
        vmin, vmax = float("inf"), float("-inf")
        n = 0
        for dq in self.series.values():
            n = max(n, len(dq))
            for v in dq:
                if v is None:
                    continue
                vmin = min(vmin, v)
                vmax = max(vmax, v)
        if n == 0 or vmin == float("inf"):
            p.setPen(QColor(DIM))
            p.drawText(plot, Qt.AlignCenter, "нет данных")
            if self.title:
                p.drawText(QRectF(4, 1, w - 8, 14), Qt.AlignLeft | Qt.AlignVCenter,
                           self.title)
            p.end()
            return
        if vmax - vmin < 1e-9:
            vmax += 1.0
            vmin -= 1.0
        span = vmax - vmin
        vmin -= span * 0.08
        vmax += span * 0.08
        if self.y_zero_line and vmin > 0 > vmin - span:
            vmin = min(vmin, 0.0)
            vmax = max(vmax, 0.0)

        # grid + y labels
        p.setPen(QPen(QColor(LINE), 1, Qt.DotLine))
        for i in range(5):
            y = plot.top() + plot.height() * i / 4
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            val = vmax - (vmax - vmin) * i / 4
            p.setPen(QColor(DIM))
            p.drawText(QRectF(2, y - 7, 44, 14), Qt.AlignLeft | Qt.AlignVCenter,
                       _fmt(val))
            p.setPen(QPen(QColor(LINE), 1, Qt.DotLine))
        if self.y_zero_line and vmin < 0 < vmax:
            y0 = plot.top() + plot.height() * (vmax - 0) / (vmax - vmin)
            p.setPen(QPen(QColor(90, 90, 90), 1, Qt.DashLine))
            p.drawLine(QPointF(plot.left(), y0), QPointF(plot.right(), y0))

        # series
        legend_x = plot.left() + 4
        for name, dq in self.series.items():
            if not dq:
                continue
            col = color_for(name)
            pen = QPen(col, 1.6)
            p.setPen(pen)
            path = QPainterPath()
            started = False
            m = len(dq)
            for i, v in enumerate(dq):
                if v is None:
                    started = False
                    continue
                x = plot.left() + plot.width() * (i / max(1, m - 1))
                y = plot.top() + plot.height() * (vmax - v) / (vmax - vmin)
                if not started:
                    path.moveTo(x, y)
                    started = True
                else:
                    path.lineTo(x, y)
            p.drawPath(path)
            # legend chip + last value
            last = next((v for v in reversed(dq) if v is not None), None)
            txt = f"{name} {_fmt(last) if last is not None else '—'}"
            p.setPen(col)
            p.drawText(QRectF(legend_x, 1, 160, 14), Qt.AlignLeft | Qt.AlignVCenter, txt)
            legend_x += 12 + p.fontMetrics().horizontalAdvance(txt)
        if self.title:
            p.setPen(QColor(DIM))
            p.drawText(QRectF(plot.right() - 220, 1, 220, 14),
                       Qt.AlignRight | Qt.AlignVCenter, self.title)
        p.end()


def _fmt(v: float) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if a >= 10_000:
        return f"{v/1000:.1f}k"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


class Bars(QWidget):
    """Horizontal percentage bars (top actions)."""

    def __init__(self, height: int = 96, parent=None):
        super().__init__(parent)
        self.items: List[tuple] = []  # (name, pct)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)

    def set_items(self, items: Dict[str, float]):
        self.items = sorted(items.items(), key=lambda kv: -kv[1])[:5]
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(FIELD))
        if not self.items:
            p.setPen(QColor(DIM))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "нет данных")
            p.end()
            return
        row_h = min(18, h // max(1, len(self.items)))
        max_pct = max((v for _, v in self.items), default=1.0) or 1.0
        for i, (name, pct) in enumerate(self.items):
            y = i * row_h + 2
            p.setPen(QColor(DIM))
            p.drawText(QRectF(4, y, 130, row_h - 3), Qt.AlignLeft | Qt.AlignVCenter,
                       name[:18])
            bar_w = (w - 200) * (pct / max_pct)
            p.fillRect(QRectF(140, y + 2, max(2, bar_w), row_h - 8), color_for("return"))
            p.setPen(QColor(DIM))
            p.drawText(QRectF(w - 56, y, 52, row_h - 3),
                       Qt.AlignRight | Qt.AlignVCenter, f"{pct:.1f}%")
        p.end()
