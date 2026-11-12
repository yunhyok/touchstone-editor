"""Interactive line plot drawn with QPainter: wheel zoom, drag pan and log axes."""

import math
from dataclasses import dataclass

import numpy as np
import shiboken6
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

COLORS = ("#5aa9ff", "#ff9f43", "#5fd38d", "#ff6b6b", "#c792ea", "#ffd166",
          "#4dd0e1", "#f78fb3", "#a3d977", "#b0bec5", "#ff8a65", "#9fa8da")
_PREFIXES = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
_MARKER_LIMIT = 20_000
_LEGEND_LIMIT = 8
_HOVER_RADIUS = 14


@dataclass
class Trace:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str
    markers: bool = False
    note: str = ""
    dashed: bool = False


@dataclass
class Region:
    start: float
    stop: float
    label: str


def engineering(value, digits=4):
    """Format a number with an SI prefix, for example 1.5k or 20m."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    exponent = min(12, max(-15, 3 * math.floor(math.log10(abs(value)) / 3)))
    return f"{value / 10 ** exponent:.{digits}g}{_PREFIXES[exponent]}"


def _linear_ticks(low, high, target):
    raw = (high - low) / max(1, target)
    if not math.isfinite(raw) or raw <= 0:
        return [], 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(factor for factor in (1, 2, 5, 10) if factor * magnitude >= raw) * magnitude
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    if last - first > 200:
        return [], step
    return [index * step for index in range(first, last + 1)], step


def _linear_labels(ticks, step):
    largest = max((abs(tick) for tick in ticks), default=0)
    exponent = 0
    if largest and not 1e-2 <= largest < 1e4:
        exponent = min(12, max(-15, 3 * math.floor(math.log10(largest) / 3)))
    scale = 10.0 ** exponent
    decimals = min(12, max(0, math.ceil(-math.log10(step / scale) - 1e-9)))
    return [f"{tick / scale:.{decimals}f}{_PREFIXES[exponent]}" for tick in ticks]


def _axis_ticks(low, high, log, target):
    """Return (positions, labels, minor positions) in axis coordinates."""
    if not log:
        ticks, step = _linear_ticks(low, high, target)
        return ticks, _linear_labels(ticks, step), []
    span = high - low
    if span < 0.9:
        # Inside one decade, round values read better than fractional powers of ten.
        values, step = _linear_ticks(10.0 ** low, 10.0 ** high, target)
        values = [value for value in values if value > 0]
        return [math.log10(value) for value in values], _linear_labels(values, step), []
    decades = range(math.floor(low), math.ceil(high) + 1)
    stride = max(1, math.ceil(span / max(2, target)))
    # A short span also labels 2 and 5 so at least a few ticks stay readable.
    labelled = (1, 2, 5) if span < 3 else (1,)
    major, minor = [], []
    for decade in decades:
        for factor in range(1, 10):
            position = decade + math.log10(factor)
            if not low - 1e-9 <= position <= high + 1e-9:
                continue
            if factor in labelled and (factor > 1 or decade % stride == 0):
                major.append((position, engineering(factor * 10.0 ** decade)))
            elif stride == 1 and span <= 8:
                minor.append(position)
    return [position for position, _ in major], [label for _, label in major], minor


def _polygon(x, y):
    polygon = QPolygonF()
    polygon.resize(len(x))
    if len(x):
        buffer = np.frombuffer(shiboken6.VoidPtr(polygon.data(), 16 * len(x), True), dtype=np.float64)
        buffer[0::2] = x
        buffer[1::2] = y
    return polygon


def _clip_polyline(x, y, left, top, right, bottom):
    """Clip a polyline to a rectangle and return the visible runs as (x, y) arrays."""
    if len(x) < 2:
        return []
    x1, y1, dx, dy = x[:-1], y[:-1], np.diff(x), np.diff(y)
    visible = np.isfinite(dx) & np.isfinite(dy)
    enter, leave = np.zeros(len(dx)), np.ones(len(dx))
    with np.errstate(divide="ignore", invalid="ignore"):
        for direction, distance in ((-dx, x1 - left), (dx, right - x1), (-dy, y1 - top), (dy, bottom - y1)):
            ratio = distance / direction
            visible &= ~((direction == 0) & (distance < 0))
            enter = np.where(direction < 0, np.maximum(enter, ratio), enter)
            leave = np.where(direction > 0, np.minimum(leave, ratio), leave)
        visible &= enter <= leave
    index = np.flatnonzero(visible)
    if not len(index):
        return []
    enter, leave = enter[index], leave[index]
    start_x, start_y = x1[index] + enter * dx[index], y1[index] + enter * dy[index]
    end_x, end_y = x1[index] + leave * dx[index], y1[index] + leave * dy[index]
    # A run continues while each segment starts where the previous one ended.
    breaks = np.flatnonzero((np.diff(index) != 1) | (leave[:-1] != 1) | (enter[1:] != 0)) + 1
    runs = []
    for first, last in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [len(index)]))):
        runs.append((np.concatenate((start_x[first:first + 1], end_x[first:last])),
                     np.concatenate((start_y[first:first + 1], end_y[first:last]))))
    return runs


class PlotWidget(QWidget):
    """Plot traces over shaded frequency regions.

    Wheel zooms around the cursor (Ctrl: x only, Shift: y only), dragging pans and
    double-click or reset_view() restores the default view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.traces = []
        self.regions = []
        self.x_log = self.y_log = True
        self.x_title = self.y_title = ""
        self.x_unit = self.y_unit = ""
        self.message = ""
        self.busy = ""
        self.view = None
        # True once the user zooms or pans; the caller can then keep that view across updates.
        self.modified = False
        self._transformed = []
        self._screen = []
        self._pixmap = None
        self._drag = None
        self._hover = None
        self.setMinimumSize(280, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        self.setAccessibleName("보간 결과 그래프")

    # Data and view -------------------------------------------------------------------

    def set_data(self, traces, regions=(), keep_view=False):
        self.traces = list(traces)
        self.regions = list(regions)
        self._transform()
        if not keep_view or self.view is None:
            self.view = self.default_view()
            self.modified = False
        self._changed()

    def set_scales(self, x_log, y_log):
        if (x_log, y_log) != (self.x_log, self.y_log):
            self.x_log, self.y_log = x_log, y_log
            self._transform()
            self.view = self.default_view()
            self.modified = False
            self._changed()

    def set_busy(self, text):
        if text != self.busy:
            self.busy = text
            self.update()

    def reset_view(self):
        self.view = self.default_view()
        self.modified = False
        self._changed()

    def set_view(self, x0, x1, y0, y1):
        """Apply a view unless it is degenerate or beyond what the axes can label."""
        for low, high, log in ((x0, x1, self.x_log), (y0, y1, self.y_log)):
            limit = 300.0 if log else 1e300
            if not (math.isfinite(low) and math.isfinite(high) and -limit <= low < high <= limit):
                return
            if high - low < 1e-9 * max(1.0, abs(low), abs(high)):
                return
        self.view = (x0, x1, y0, y1)
        self.modified = True
        self._pixmap = None
        self._hover = None
        self.update()

    def _changed(self):
        self._pixmap = None
        self._hover = None
        self.update()

    @staticmethod
    def _axis(values, log):
        values = np.asarray(values, dtype=np.float64)
        if not log:
            return np.where(np.isfinite(values), values, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)

    def _transform(self):
        self._transformed = []
        for trace in self.traces:
            u, v = self._axis(trace.x, self.x_log), self._axis(trace.y, self.y_log)
            finite = np.isfinite(u)
            # A log axis hides a leading zero frequency; the rest can be searched when sorted.
            lead = int(np.argmax(finite)) if finite.any() else len(u)
            ordered = bool(finite[lead:].all() and (np.diff(u[lead:]) >= 0).all())
            self._transformed.append((u, v, lead, ordered))

    def default_view(self):
        bounds = []
        for axis in range(2):
            low, high = math.inf, -math.inf
            for pair in self._transformed:
                valid = np.isfinite(pair[0]) & np.isfinite(pair[1])
                if valid.any():
                    low = min(low, float(pair[axis][valid].min()))
                    high = max(high, float(pair[axis][valid].max()))
            if axis == 0 and low > high:
                # Without traces the regions still define a sensible frequency span.
                edges = self._axis([edge for region in self.regions for edge in (region.start, region.stop)],
                                   self.x_log)
                edges = edges[np.isfinite(edges)]
                if len(edges):
                    low, high = float(edges.min()), float(edges.max())
            if low > high:
                low, high = 0.0, 1.0
            if high - low <= 1e-12 * max(1.0, abs(low), abs(high)):
                # A single value still needs a span the view limits accept.
                pad = 0.5 if (self.x_log, self.y_log)[axis] else max(abs(low) * 0.1, 0.5)
            else:
                pad = (high - low) * (0.02 if axis == 0 else 0.06)
            bounds += [low - pad, high + pad]
        return tuple(bounds)

    # Geometry --------------------------------------------------------------------------

    def plot_rect(self):
        metrics = QFontMetricsF(self.font())
        left = metrics.horizontalAdvance("-000.00m") + metrics.height() + 14
        top = metrics.height() + 12
        bottom = 2 * metrics.height() + 16
        return QRectF(left, top, max(10.0, self.width() - left - 16), max(10.0, self.height() - top - bottom))

    def _scale(self, rect):
        x0, x1, y0, y1 = self.view
        return (rect.width() / (x1 - x0), rect.left() - x0 * rect.width() / (x1 - x0),
                -rect.height() / (y1 - y0), rect.bottom() + y0 * rect.height() / (y1 - y0))

    # Painting --------------------------------------------------------------------------

    def paintEvent(self, event):
        ratio = self.devicePixelRatioF()
        if self._pixmap is None or self._pixmap.size() != (self.size() * ratio):
            self._pixmap = QPixmap(self.size() * ratio)
            self._pixmap.setDevicePixelRatio(ratio)
            self._render(self._pixmap)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        self._paint_overlay(painter)
        painter.end()

    def _render(self, target):
        painter = QPainter(target)
        painter.setFont(self.font())
        painter.fillRect(self.rect(), QColor("#202226"))
        rect = self.plot_rect()
        painter.fillRect(rect, QColor("#16181c"))
        metrics = QFontMetricsF(self.font())
        ax, bx, ay, by = self._scale(rect)
        x0, x1, y0, y1 = self.view
        self._paint_regions(painter, rect, metrics, ax, bx)

        ticks_x = _axis_ticks(x0, x1, self.x_log, max(2, int(rect.width() // 90)))
        ticks_y = _axis_ticks(y0, y1, self.y_log, max(2, int(rect.height() // 48)))
        painter.setPen(QPen(QColor("#262b32"), 1))
        for tick in ticks_x[2]:
            painter.drawLine(QPointF(ax * tick + bx, rect.top()), QPointF(ax * tick + bx, rect.bottom()))
        for tick in ticks_y[2]:
            painter.drawLine(QPointF(rect.left(), ay * tick + by), QPointF(rect.right(), ay * tick + by))
        painter.setPen(QPen(QColor("#3a414b"), 1))
        for tick in ticks_x[0]:
            painter.drawLine(QPointF(ax * tick + bx, rect.top()), QPointF(ax * tick + bx, rect.bottom()))
        for tick in ticks_y[0]:
            painter.drawLine(QPointF(rect.left(), ay * tick + by), QPointF(rect.right(), ay * tick + by))

        painter.setPen(QColor("#b9c1cc"))
        for tick, label in zip(*ticks_x[:2]):
            width = metrics.horizontalAdvance(label)
            painter.drawText(QPointF(ax * tick + bx - width / 2, rect.bottom() + metrics.ascent() + 5), label)
        for tick, label in zip(*ticks_y[:2]):
            width = metrics.horizontalAdvance(label)
            painter.drawText(QPointF(rect.left() - width - 7, ay * tick + by + metrics.ascent() / 2 - 1), label)
        painter.drawText(QRectF(rect.left(), rect.bottom() + metrics.height() + 6, rect.width(), metrics.height() + 4),
                         Qt.AlignCenter, self.x_title)
        painter.save()
        painter.translate(metrics.height(), rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-rect.height() / 2, -metrics.height(), rect.height(), metrics.height() + 4),
                         Qt.AlignCenter, self.y_title)
        painter.restore()

        self._paint_traces(painter, rect, ax, bx, ay, by)
        painter.setPen(QPen(QColor("#626c7a"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        if self.message:
            painter.setPen(QColor("#dce3ed"))
            painter.drawText(rect.adjusted(16, 16, -16, -16), Qt.AlignCenter | Qt.TextWordWrap, self.message)
        painter.end()

    def _paint_regions(self, painter, rect, metrics, ax, bx):
        x0, x1 = self.view[:2]
        for number, region in enumerate(self.regions):
            start, stop = self._axis([region.start, region.stop], self.x_log)
            # A region reaching zero on a log axis extends to the left edge.
            start = x0 if not np.isfinite(start) else start
            if not np.isfinite(stop) or stop <= x0 or start >= x1:
                continue
            left, right = max(rect.left(), ax * start + bx), min(rect.right(), ax * stop + bx)
            painter.fillRect(QRectF(left, rect.top(), right - left, rect.height()),
                             QColor(143, 193, 255, 30 if number % 2 == 0 else 12))
            painter.setPen(QPen(QColor("#6f8fb8"), 1, Qt.DashLine))
            for edge in (start, stop):
                if x0 <= edge <= x1:
                    painter.drawLine(QPointF(ax * edge + bx, rect.top()), QPointF(ax * edge + bx, rect.bottom()))
            painter.setPen(QColor("#9fc3f0"))
            label = metrics.elidedText(region.label, Qt.ElideRight, right - left - 6)
            painter.drawText(QRectF(left, rect.top() - metrics.height() - 4, right - left, metrics.height() + 2),
                             Qt.AlignCenter, label)

    def _paint_traces(self, painter, rect, ax, bx, ay, by):
        painter.save()
        painter.setClipRect(rect)
        x0, x1 = self.view[:2]
        self._screen = []
        total = 0
        drawn = []
        for trace, (u, v, lead, ordered) in zip(self.traces, self._transformed):
            first, last = 0, len(u)
            if ordered:
                # Keep one point beyond each edge so the line reaches the border.
                first = max(lead, lead + int(np.searchsorted(u[lead:], x0, side="left")) - 1)
                last = min(len(u), lead + int(np.searchsorted(u[lead:], x1, side="right")) + 1)
            px, py = ax * u[first:last] + bx, ay * v[first:last] + by
            self._screen.append((px, py, first))
            drawn.append((trace, px, py))
            total += len(px)
        painter.setRenderHint(QPainter.Antialiasing, total <= 150_000)
        margin = 4.0
        for trace, px, py in drawn:
            color = QColor(trace.color)
            if trace.markers:
                inside = (np.isfinite(px) & np.isfinite(py) & (px >= rect.left() - margin)
                          & (px <= rect.right() + margin) & (py >= rect.top() - margin)
                          & (py <= rect.bottom() + margin))
                if total > _MARKER_LIMIT:
                    color.setAlpha(150)
                pen = QPen(color, 5.0 if total <= _MARKER_LIMIT else 3.0)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawPoints(_polygon(px[inside], py[inside]))
                continue
            pen = QPen(color, 1.0, Qt.DashLine if trace.dashed else Qt.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for run_x, run_y in _clip_polyline(px, py, rect.left() - margin, rect.top() - margin,
                                               rect.right() + margin, rect.bottom() + margin):
                painter.drawPolyline(_polygon(run_x, run_y))
        painter.restore()

    def _paint_overlay(self, painter):
        rect = self.plot_rect()
        metrics = QFontMetricsF(self.font())
        painter.setFont(self.font())
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.busy:
            width = metrics.horizontalAdvance(self.busy) + 16
            box = QRectF(rect.left() + 8, rect.top() + 8, width, metrics.height() + 8)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(56, 108, 176, 230))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(box, Qt.AlignCenter, self.busy)
        self._paint_legend(painter, rect, metrics)
        if self._hover is None:
            return
        index, point = self._hover
        trace = self.traces[index]
        px, py, first = self._screen[index]
        painter.save()
        painter.setClipRect(rect)
        if not trace.markers:
            pen = QPen(QColor(trace.color), 2.5)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            for run_x, run_y in _clip_polyline(px, py, rect.left() - 4, rect.top() - 4,
                                               rect.right() + 4, rect.bottom() + 4):
                painter.drawPolyline(_polygon(run_x, run_y))
        position = QPointF(px[point], py[point])
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QColor(trace.color))
        painter.drawEllipse(position, 4.5, 4.5)
        painter.restore()
        lines = [trace.label,
                 f"{engineering(float(trace.x[first + point]), 6)}{self.x_unit}",
                 f"{float(trace.y[first + point]):.6g}{self.y_unit}" + (f"  ·  {trace.note}" if trace.note else "")]
        width = max(metrics.horizontalAdvance(line) for line in lines) + 16
        height = len(lines) * metrics.height() + 10
        left = position.x() + 14 if position.x() + 14 + width <= self.width() else position.x() - 14 - width
        top = min(max(rect.top(), position.y() - height - 8), self.height() - height - 2)
        box = QRectF(max(2.0, left), top, width, height)
        painter.setPen(QPen(QColor("#717c8d"), 1))
        painter.setBrush(QColor(32, 34, 38, 240))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor("#eceef1"))
        for number, line in enumerate(lines):
            painter.drawText(QPointF(box.left() + 8, box.top() + 5 + metrics.ascent() + number * metrics.height()),
                             line)

    def _paint_legend(self, painter, rect, metrics):
        labels = {}
        for trace in self.traces:
            labels.setdefault(trace.label, (trace.color, trace.dashed))
        if not 1 <= len(labels) <= _LEGEND_LIMIT:
            return
        width = min(max(metrics.horizontalAdvance(label) for label in labels) + 42, rect.width() * 0.5)
        height = len(labels) * metrics.height() + 10
        if width < 80 or height > rect.height() - 16:
            return
        # Use the corner that covers the fewest plotted points.
        corners = [QRectF(left, top, width, height)
                   for top in (rect.top() + 8, rect.bottom() - height - 8)
                   for left in (rect.right() - width - 8, rect.left() + 8)]
        if self.busy:
            corners.append(corners.pop(1))

        def covered(box):
            return sum(int(((screen[0] >= box.left()) & (screen[0] <= box.right())
                            & (screen[1] >= box.top()) & (screen[1] <= box.bottom())).sum())
                       for screen in self._screen)

        box = min(corners, key=covered)
        painter.setPen(QPen(QColor("#505761"), 1))
        painter.setBrush(QColor(32, 34, 38, 215))
        painter.drawRoundedRect(box, 4, 4)
        for number, (label, (color, dashed)) in enumerate(labels.items()):
            top = box.top() + 5 + number * metrics.height()
            painter.setPen(QPen(QColor(color), 2.0, Qt.DashLine if dashed else Qt.SolidLine))
            painter.drawLine(QPointF(box.left() + 8, top + metrics.height() / 2),
                             QPointF(box.left() + 24, top + metrics.height() / 2))
            painter.setPen(QColor("#eceef1"))
            painter.drawText(QPointF(box.left() + 30, top + metrics.ascent()),
                             metrics.elidedText(label, Qt.ElideRight, width - 36))

    # Interaction -----------------------------------------------------------------------

    def zoom(self, factor, position, horizontal=True, vertical=True):
        """Scale the view around a widget position; a factor below one zooms in."""
        rect = self.plot_rect()
        x0, x1, y0, y1 = self.view
        anchor_x = x0 + (position.x() - rect.left()) / rect.width() * (x1 - x0)
        anchor_y = y0 + (rect.bottom() - position.y()) / rect.height() * (y1 - y0)

        def scaled(low, high, anchor):
            share = (anchor - low) / (high - low)
            span = (high - low) * factor
            return anchor - share * span, anchor + (1 - share) * span

        if horizontal:
            x0, x1 = scaled(x0, x1, anchor_x)
        if vertical:
            y0, y1 = scaled(y0, y1, anchor_y)
        self.set_view(x0, x1, y0, y1)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            return
        rect = self.plot_rect()
        position = event.position()
        modifiers = event.modifiers()
        # Over an axis, or with a modifier, only that axis zooms.
        horizontal = not (modifiers & Qt.ShiftModifier) and position.x() >= rect.left()
        vertical = not (modifiers & Qt.ControlModifier) and position.y() <= rect.bottom()
        if not horizontal and not vertical:
            horizontal = vertical = True
        self.zoom(0.85 ** (delta / 120), position, horizontal, vertical)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            self._drag = (event.position(), self.view)
            self._hover = None
            self.setCursor(Qt.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            origin, (x0, x1, y0, y1) = self._drag
            rect = self.plot_rect()
            shift_x = (event.position().x() - origin.x()) / rect.width() * (x1 - x0)
            shift_y = (event.position().y() - origin.y()) / rect.height() * (y1 - y0)
            self.set_view(x0 - shift_x, x1 - shift_x, y0 + shift_y, y1 + shift_y)
            return
        hover = self._nearest(event.position())
        if hover != self._hover:
            self._hover = hover
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag is not None:
            self._drag = None
            self.setCursor(Qt.CrossCursor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = None
            self.setCursor(Qt.CrossCursor)
            self.reset_view()

    def leaveEvent(self, event):
        if self._hover is not None:
            self._hover = None
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D and event.modifiers() & Qt.ControlModifier:
            self.reset_view()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        self._pixmap = None
        super().resizeEvent(event)

    def _nearest(self, position):
        rect = self.plot_rect()
        if not rect.adjusted(-4, -4, 4, 4).contains(position) or self._pixmap is None:
            return None
        best, limit = None, _HOVER_RADIUS ** 2
        # Later traces are drawn on top, so they win ties.
        for index, screen in enumerate(self._screen):
            if screen is None or not len(screen[0]):
                continue
            distance = (screen[0] - position.x()) ** 2 + (screen[1] - position.y()) ** 2
            distance = np.where(np.isfinite(distance), distance, np.inf)
            point = int(np.argmin(distance))
            if distance[point] <= limit:
                best, limit = (index, point), float(distance[point])
        return best
