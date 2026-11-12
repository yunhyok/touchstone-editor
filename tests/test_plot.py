import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from touchstone_editor.plot import PlotWidget, Region, Trace, _axis_ticks, _clip_polyline, engineering


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def plot(application):
    widget = PlotWidget()
    widget.resize(640, 420)
    widget.show()
    yield widget
    widget.close()


def test_engineering_notation():
    assert [engineering(value) for value in (0, 1, 1500, 2e9, 0.02, 4.7e-6, -3e3)] == [
        "0", "1", "1.5k", "2G", "20m", "4.7µ", "-3k"]


def test_log_ticks_label_decades_then_125_then_plain_values():
    wide = _axis_ticks(1.5, 9.3, True, 9)
    assert wide[1] == ["100", "1k", "10k", "100k", "1M", "10M", "100M", "1G"] and len(wide[2]) > 40
    assert _axis_ticks(5.2, 6.8, True, 9)[1] == ["200k", "500k", "1M", "2M", "5M"]
    narrow = _axis_ticks(5.9, 6.3, True, 9)
    assert narrow[1][0] == "0.8M" and narrow[2] == []
    # Too many decades for the width skip labels instead of overlapping.
    assert _axis_ticks(-12, 12, True, 6)[1] == ["1p", "10n", "100µ", "1", "10k", "100M", "1T"]


def test_linear_ticks_use_round_steps_and_a_shared_prefix():
    positions, labels, minor = _axis_ticks(0, 2e9, False, 9)
    assert positions == [0, 5e8, 1e9, 1.5e9, 2e9] and labels == ["0.0G", "0.5G", "1.0G", "1.5G", "2.0G"]
    assert _axis_ticks(0.012, 0.094, False, 6)[1] == ["0.02", "0.04", "0.06", "0.08"]
    assert _axis_ticks(-40, 5, False, 5)[1] == ["-40", "-30", "-20", "-10", "0"]
    assert _axis_ticks(1.0, 1.0, False, 5)[0] == []


def test_polyline_clipping_keeps_exact_border_crossings_and_splits_gaps():
    x = np.array([-10.0, 5.0, 20.0, 30.0, 40.0, 50.0])
    y = np.array([5.0, 5.0, 5.0, np.nan, 5.0, 500.0])
    runs = _clip_polyline(x, y, 0, 0, 100, 100)
    assert len(runs) == 2 and runs[0][0].tolist() == [0.0, 5.0, 20.0]
    # The last segment leaves through the bottom border at y = 100.
    assert runs[1][0].tolist() == pytest.approx([40.0, 40.0 + 10 * 95 / 495])
    assert runs[1][1].tolist() == pytest.approx([5.0, 100.0])
    assert _clip_polyline(np.array([1.0]), np.array([1.0]), 0, 0, 9, 9) == []
    assert _clip_polyline(np.array([-5.0, -1.0]), np.array([1.0, 1.0]), 0, 0, 9, 9) == []
    # Coordinates far outside the view, as in a deep zoom, are still cut at the border.
    far = _clip_polyline(np.array([-1e15, 1e15]), np.array([-1e15, 1e15]), 0, 0, 100, 100)
    assert far[0][0].tolist() == pytest.approx([0, 100], abs=0.5)
    assert far[0][1].tolist() == pytest.approx([0, 100], abs=0.5)


def test_log_axes_skip_non_positive_values_and_regions_reach_the_edge(plot, application):
    frequency = np.array([0.0, 10.0, 100.0, 1000.0])
    plot.set_data([Trace("a", frequency, np.array([1.0, 0.0, 10.0, 100.0]), "#ffffff")],
                  [Region(0.0, 100.0, "first")])
    # 0 Hz and the zero value have no logarithm, so the view starts at the first drawable point.
    assert plot.view[0] == pytest.approx(2 - 0.02) and plot.view[1] == pytest.approx(3 + 0.02)
    assert plot.grab().width() == 640
    plot.set_scales(False, False)
    assert plot.view[0] < 0 < plot.view[2] + 10 and plot.view[1] > 1000


def test_empty_plot_uses_regions_for_its_span_and_shows_a_message(plot):
    plot.message = "no data"
    plot.set_data([], [Region(1e3, 1e6, "band")])
    assert plot.view[0] == pytest.approx(3 - 0.06) and plot.view[1] == pytest.approx(6 + 0.06)
    assert not plot.grab().isNull()


def test_zoom_keeps_the_anchor_and_rejects_degenerate_views(plot):
    x = np.logspace(1, 6, 50)
    plot.set_data([Trace("a", x, x, "#ffffff")])
    rect = plot.plot_rect()
    anchor = QPointF(rect.left() + rect.width() / 4, rect.top() + rect.height() / 4)
    before = plot.view
    plot.zoom(0.5, anchor)
    after = plot.view
    assert after[0] + (after[1] - after[0]) / 4 == pytest.approx(before[0] + (before[1] - before[0]) / 4)
    assert after[3] - (after[3] - after[2]) / 4 == pytest.approx(before[3] - (before[3] - before[2]) / 4)
    plot.zoom(0.5, anchor, horizontal=False)
    assert plot.view[:2] == after[:2] and plot.view[3] - plot.view[2] == pytest.approx((after[3] - after[2]) / 2)
    for _ in range(200):
        plot.zoom(0.1, anchor)
    assert plot.view[1] - plot.view[0] >= 1e-9
    for _ in range(400):
        plot.zoom(10, anchor)
    # A log axis stops before powers of ten that cannot be labelled.
    assert -300 <= plot.view[0] < plot.view[1] <= 300
    assert not plot.grab().isNull()
    plot.reset_view()
    assert plot.view == pytest.approx(before) and not plot.modified


def test_hover_finds_the_nearest_point_and_prefers_markers(plot, application):
    x = np.array([10.0, 100.0, 1000.0])
    line = Trace("line", x, np.array([1.0, 10.0, 100.0]), "#ffffff", note="보간")
    dots = Trace("line", x, np.array([1.0, 10.0, 100.0]), "#ffffff", markers=True, note="원본")
    plot.set_data([line, dots])
    plot.grab()
    scale = plot._scale(plot.plot_rect())
    middle = QPointF(scale[0] * 2 + scale[1], scale[2] * 1 + scale[3])
    assert plot._nearest(middle + QPointF(3, -4)) == (1, 1)
    assert plot._nearest(middle + QPointF(60, 60)) is None
    plot._hover = (1, 1)
    assert not plot.grab().isNull()
    plot.set_data([line])
    assert plot._hover is None
