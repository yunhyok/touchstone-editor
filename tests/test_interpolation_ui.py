import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit

import touchstone_editor.app as app_module
import touchstone_editor.interpolation_tab as tab_module
from touchstone_editor.app import MainWindow
from touchstone_editor.core import read_touchstone
from touchstone_editor.interpolation import (DECADE_COUNT, LINEAR_AXIS, LINEAR_COUNT, LINEAR_STEP, LOG_AXIS, Band,
                                             load_network, s_to_z)
from touchstone_editor.theme import apply_theme

SOURCE = ("! Port[1] = CH10\n! Port[2] = CH2\n! Port[3] = CH1\n# Hz S RI R 50\n"
          "100 -0.9 0 0.01 0 0.02 0\n 0.01 0 -0.8 0 0.03 0\n 0.02 0 0.03 0 -0.7 0\n"
          "1000 -0.8 0.1 0.02 0 0.03 0\n 0.02 0 -0.7 0.1 0.04 0\n 0.03 0 0.04 0 -0.6 0.1\n"
          "3000 -0.5 0.3 0.04 0 0.05 0\n 0.04 0 -0.4 0.3 0.06 0\n 0.05 0 0.06 0 -0.3 0.3\n")


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def settle(window, application):
    for _ in range(3):
        application.processEvents()
        while window.job is not None:
            window.job.wait()
            application.processEvents()
        window.interpolation.wait()


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.QMessageBox, "information", lambda *args: None)
    source = tmp_path / "example.s3p"
    source.write_text(SOURCE)
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    window.load_document(read_touchstone(source))
    window.pages.setCurrentWidget(window.interpolation)
    settle(window, application)
    yield window
    settle(window, application)
    window.undo_stack.setClean()
    window.close()


def select(tab, *rows):
    selection = tab.table.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(tab.proxy.index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)


def linear_bands(tab, step=500):
    tab.set_bands([Band(100, 3000, LINEAR_STEP, step)])
    tab.axis.setCurrentIndex(tab.axis.findData(LINEAR_AXIS))
    tab.wait()


def test_tab_loads_data_and_clips_default_bands_to_the_source(window):
    tab = window.interpolation
    assert tab.network.frequencies.tolist() == [100.0, 1000.0, 3000.0]
    assert tab.stack.currentIndex() == 1
    # Only 100 Hz-1 kHz and 1-3 kHz of the default bands overlap this file.
    assert tab.plan.ranges[0] == (100.0, 1000.0, 101, 99)
    assert tab.plan.ranges[1][:2] == (1000.0, 3000.0)
    assert tab.plan.ranges[2] is None
    assert np.isin([100.0, 1000.0, 3000.0], tab.plan.frequencies).all()
    assert tab.bands.item(0, 4).text() == "101점 (+99) · 100 Hz ~ 1 kHz"
    assert tab.bands.item(2, 4).text() == "원본 범위 밖"
    added = int(tab.plan.added.sum())
    assert f"원본 3점 + 추가 {added:,}점" in tab.report.text()
    assert f"<b>{added + 3:,}점</b> · 약 " in tab.report.text()
    assert tab.axis.currentData() == LOG_AXIS
    assert "3 points" in window.meta_label.text()
    assert window.save_button.text() == "보간 결과 저장…" and window.save_button.isEnabled()
    window.pages.setCurrentIndex(0)
    assert window.save_button.text() == "Ansys용으로 저장…"


def test_invalid_band_reports_its_row_and_blocks_saving(window):
    tab = window.interpolation
    tab.bands.item(0, 1).setText("abc")
    tab.wait()
    assert "구간 1:" in tab.report.text()
    assert not tab.can_export() and not window.save_button.isEnabled()
    tab.bands.item(0, 1).setText("2k")
    tab.wait()
    assert tab.bands.item(0, 1).text() == "2 kHz"
    assert tab.can_export() and window.save_button.isEnabled()
    tab.bands.item(0, 3).setText("1.5")
    tab.wait()
    assert "정수" in tab.report.text() and not tab.can_export()
    tab.bands.item(0, 3).setText("1e999999k")
    tab.wait()
    assert not tab.can_export()
    tab.reset_bands()
    tab.wait()
    assert tab.can_export() and tab.bands.rowCount() == 3


def test_band_mode_add_and_remove(window, application):
    tab = window.interpolation
    base = len(tab.plan.frequencies)
    mode = tab.bands.cellWidget(0, 2)
    # Scrolling over the mode cell must not change the band.
    centre = QPointF(mode.rect().center())
    QApplication.sendEvent(mode, QWheelEvent(centre, mode.mapToGlobal(centre), QPoint(0, 0), QPoint(0, -120),
                                             Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert mode.currentData() == LINEAR_COUNT and tab.bands.item(0, 3).text() == "101"
    mode.setCurrentIndex(mode.findData(LINEAR_STEP))
    tab.wait()
    # A step replaces the count with one hundredth of the band width.
    assert tab.bands.item(0, 3).text() == "10 Hz"
    assert len(tab.plan.frequencies) == base - 10
    tab.remove_band()
    tab.remove_band()
    tab.wait()
    assert tab.bands.rowCount() == 1 and tab.plan.frequencies[-2:].tolist() == [1000.0, 3000.0]
    tab.add_band()
    tab.wait()
    assert tab.bands.item(1, 0).text() == "1 kHz" and tab.bands.item(1, 1).text() == "3 kHz"
    assert tab.bands.cellWidget(1, 2).currentData() == DECADE_COUNT and tab.bands.item(1, 3).text() == "100"
    tab.bands.cellWidget(1, 2).setCurrentIndex(tab.bands.cellWidget(1, 2).findData(LINEAR_COUNT))
    tab.bands.item(1, 3).setText("3")
    tab.wait()
    assert tab.plan.frequencies[-3:].tolist() == [1000.0, 2000.0, 3000.0]
    # Removing a band while a preview is still pending must not break the plot regions.
    tab.remove_band()
    tab.job_done(tab.wanted, tab.cache[tab.wanted])
    assert [region.label for region in tab.plot.regions] == ["1 · Step 10 Hz", "2 · Linear 3"]
    tab.wait()
    assert [region.label for region in tab.plot.regions] == ["1 · Step 10 Hz"]


def test_interpolation_axis_changes_preview_and_export(window, application, monkeypatch):
    tab = window.interpolation
    tab.parameter.setCurrentIndex(tab.parameter.findData("S"))
    tab.quantity.setCurrentIndex(tab.quantity.findData("real"))
    tab.set_bands([Band(1000, 3000, LINEAR_COUNT, 3)])
    select(tab, 0)
    tab.wait()
    index = tab.plan.frequencies.tolist().index(2000.0)
    # 2 kHz lies at log(2)/log(3) of the way from 1 kHz to 3 kHz on the log axis, and halfway on the linear one.
    position = np.log(2) / np.log(3)
    assert tab.plot.traces[0].y[index] == pytest.approx(-0.8 + 0.3 * position)
    tab.axis.setCurrentIndex(tab.axis.findData(LINEAR_AXIS))
    tab.wait()
    assert tab.plot.traces[0].y[index] == pytest.approx(-0.65)
    target = window.document.source.with_name("linear.s3p")
    monkeypatch.setattr(window, "choose_target", lambda tag, title: target)
    window.save_current()
    settle(window, application)
    assert b"on a linear frequency axis for all ports" in target.read_bytes()
    assert load_network(read_touchstone(target)).matrices[2][0, 0] == pytest.approx(-0.65 + 0.2j)


def test_channel_list_sorts_naturally_filters_and_checks(window, application):
    tab = window.interpolation
    names = lambda: [tab.proxy.index(row, 2).data() for row in range(tab.proxy.rowCount())]
    assert names() == ["CH10", "CH2", "CH1"]
    tab.table.sortByColumn(2, Qt.AscendingOrder)
    assert names() == ["CH1", "CH2", "CH10"]
    tab.table.sortByColumn(2, Qt.DescendingOrder)
    assert names() == ["CH10", "CH2", "CH1"]
    assert tab.ri_ports().tolist() == [True, True, True]
    tab.proxy.setData(tab.proxy.index(1, 0), Qt.Unchecked, Qt.CheckStateRole)
    assert tab.ri_ports().tolist() == [True, False, True]
    tab.table.sortByColumn(0, Qt.AscendingOrder)
    assert names() == ["CH10", "CH1", "CH2"]
    tab.table.sortByColumn(1, Qt.AscendingOrder)
    # A click in the check column toggles that channel without changing the plotted selection.
    select(tab, 0)
    viewport = tab.table.viewport()
    cell = tab.table.visualRect(tab.proxy.index(2, 0)).center()
    QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, cell)
    assert tab.ri_ports().tolist() == [True, False, False] and tab.selected_sources() == [1]
    # A press that drifts before the release still only toggles.
    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, cell)
    QTest.mouseMove(viewport, cell + QPoint(1, 1))
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, cell + QPoint(1, 1))
    assert tab.ri_ports().tolist() == [True, False, True] and tab.selected_sources() == [1]
    # A right click selects the row as in any table, and must not leave the cell deaf to the next left click.
    QTest.mouseClick(viewport, Qt.RightButton, Qt.NoModifier, cell)
    assert tab.ri_ports().tolist() == [True, False, True] and tab.selected_sources() == [3]
    QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, cell)
    assert tab.ri_ports().tolist() == [True, False, False] and tab.selected_sources() == [3]
    select(tab, 2)
    tab.check_only_selected()
    assert tab.ri_ports().tolist() == [False, False, True]
    select(tab, 0, 1)
    QTest.keyClick(tab.table, Qt.Key_Space)
    assert tab.ri_ports().tolist() == [True, True, True]
    QTest.keyClick(tab.table, Qt.Key_Space)
    assert tab.ri_ports().tolist() == [False, False, True]
    tab.check(None, True)
    tab.table.clearSelection()
    tab.search.setText("ch1")
    assert names() == ["CH10", "CH1"]
    tab.wait()
    # Without a selection the plot follows the filter.
    assert {trace.label.split()[-1] for trace in tab.plot.traces} == {"CH10", "CH1"}
    tab.search.setText("none")
    tab.wait()
    assert tab.plot.traces == [] and "채널이 없습니다" in tab.plot.message


def test_plot_follows_selection_parameter_and_scales(window):
    tab = window.interpolation
    plot = tab.plot
    linear_bands(tab)
    targets = tab.plan.frequencies
    assert targets.tolist() == [100, 600, 1000, 1100, 1600, 2100, 2600, 3000]
    # Every channel: one interpolated line and one set of source markers each.
    assert len(plot.traces) == 6 and [trace.markers for trace in plot.traces] == [False] * 3 + [True] * 3
    assert plot.traces[0].label == "Z(1,1)  CH10" and plot.y_title == "|Z| (Ω)"
    assert [region.label for region in plot.regions] == ["1 · Step 500 Hz"]
    network = tab.network
    assert plot.traces[3].y[1] == pytest.approx(abs(s_to_z(network.matrices[1], network.references)[0, 0]))
    assert len(plot.traces[0].x) == len(targets)
    blend = (4 * network.matrices[0] + 5 * network.matrices[1]) / 9
    assert plot.traces[0].y[1] == pytest.approx(abs(s_to_z(blend, network.references)[0, 0]))

    select(tab, 0, 2)
    tab.wait()
    assert [trace.label for trace in plot.traces] == ["Z(1,1)  CH10", "Z(3,3)  CH1"] * 2
    tab.transfer.setChecked(True)
    tab.wait()
    assert [trace.label for trace in plot.traces[:4]] == ["Z(1,1)  CH10", "Z(3,3)  CH1",
                                                         "Z(1,3)  CH10  ←  CH1", "Z(3,1)  CH1  ←  CH10"]
    tab.parameter.setCurrentIndex(tab.parameter.findData("S"))
    tab.quantity.setCurrentIndex(tab.quantity.findData("db"))
    tab.wait()
    assert not tab.y_log.isChecked() and not plot.y_log and plot.x_log
    assert plot.y_title == "|S| (dB)"
    assert plot.traces[4].y[0] == pytest.approx(20 * np.log10(0.9))
    tab.x_log.setChecked(False)
    tab.show_source.setChecked(False)
    tab.wait()
    assert not plot.x_log and all(not trace.markers for trace in plot.traces)
    tab.show_result.setChecked(False)
    tab.wait()
    assert plot.traces == [] and plot.message


def test_unchecked_channels_and_the_comparison_use_magnitude_and_angle(window):
    tab = window.interpolation
    tab.parameter.setCurrentIndex(tab.parameter.findData("S"))
    tab.quantity.setCurrentIndex(tab.quantity.findData("real"))
    linear_bands(tab)
    select(tab, 0)
    tab.wait()
    index = tab.plan.frequencies.tolist().index(2100.0)
    position = 0.55
    start, stop = complex(-0.8, 0.1), complex(-0.5, 0.3)
    ri = tab.plot.traces[0].y[index]
    assert ri == pytest.approx(-0.8 + 0.3 * position)
    polar = ((abs(start) * (1 - position) + abs(stop) * position)
             * np.cos(np.angle(start) + (np.angle(stop) - np.angle(start)) * position))
    # The comparison overlays the all-magnitude/angle result as a dashed line.
    tab.compare.setChecked(True)
    tab.wait()
    assert [(trace.dashed, trace.markers) for trace in tab.plot.traces] == [(True, False), (False, False),
                                                                           (False, True)]
    assert tab.plot.traces[0].label.endswith("· MA")
    assert tab.plot.traces[0].y[index] == pytest.approx(polar) and tab.plot.traces[1].y[index] == pytest.approx(ri)
    tab.compare.setChecked(False)
    tab.check(None, False)
    tab.wait()
    assert tab.plot.traces[0].y[index] == pytest.approx(polar)
    assert polar != pytest.approx(ri)
    assert "RI 보간 0 / 3개" in tab.channel_label.text() and "RI 보간 채널 없음" in tab.report.text()


def test_report_warns_about_sparse_bands_and_steep_phase(window, application, monkeypatch):
    tab = window.interpolation
    report = lambda: tab.report.text()
    # This file turns up to 35.5 degrees between 1 kHz and 3 kHz, beyond what real/imaginary handles well.
    assert "⚠ 위상 변화: RI 채널 3개가 20° 초과 (최대 35.5°)" in report() and report().count("⚠") == 1
    assert "(1 kHz ~ 3 kHz)" in report()
    # The checks finished in the background: gain and symmetry of the added points.
    assert "수동성: 최대 특이값" in report() and "대칭성:" in report()
    assert tab.checks.added_gain <= tab.checks.source_gain + 1e-12
    assert "MA–RI 차이: 원본 격자 최대" in report() and "출력 격자 최대" in report()
    assert "기준 임피던스 곱" in report()
    tab.set_bands([Band(100, 3000, DECADE_COUNT, 20)])
    tab.wait()
    assert "⚠ 구간 1: decade당 20점" in report()

    # A channel that turns 90 degrees between source points is flagged and can be unchecked in one step.
    turning = window.document.source.with_name("turning.s2p")
    turning.write_text("! Port[1] = LINE\n! Port[2] = POWER\n# Hz S MA R 50\n"
                       "100 0.9 0 0.001 0 0.001 0 0.9 179\n1000 0.9 90 0.001 0 0.001 0 0.9 178\n")
    window.open_file(str(turning))
    settle(window, application)
    assert [tab.model.index(row, 3).data() for row in range(2)] == ["90.0°", "1.0°"]
    assert "⚠ 위상 변화: RI 채널 1개가 20° 초과 (최대 90.0°)" in report() and "29.3%" in report()
    assert tab.model.index(0, 3).data(Qt.ForegroundRole) is not None
    assert tab.model.index(1, 3).data(Qt.ForegroundRole) is None
    tab.table.sortByColumn(3, Qt.DescendingOrder)
    assert tab.proxy.index(0, 2).data() == "LINE"
    tab.uncheck_steep()
    tab.wait()
    assert tab.ri_ports().tolist() == [False, True]
    # The band settings stay across files, so only the sparse-band warning is left.
    assert "✔ 위상 변화 최대 1.0° (RI 채널, 지점 추가 구간)" in report()
    assert "⚠ 위상 변화" not in report() and report().count("⚠") == 1
    assert tab.model.index(0, 3).data(Qt.ForegroundRole) is None

    # Large problems wait for the button instead of checking on every change.
    monkeypatch.setattr(tab_module, "CHECK_BUDGET", 0)
    tab.check(None, True)
    tab.wait()
    assert tab.checks is None and not tab.check_button.isHidden() and "버튼으로 실행" in report()
    tab.check_button.click()
    tab.wait()
    assert tab.checks is not None and tab.check_button.isHidden() and "수동성: 최대 특이값" in report()


def test_wheel_zoom_drag_pan_and_ctrl_d_reset(window, application):
    tab = window.interpolation
    plot = tab.plot
    home = plot.view
    rect = plot.plot_rect()
    centre = rect.center()
    wheel = QWheelEvent(centre, plot.mapToGlobal(centre), QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                        Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(plot, wheel)
    zoomed = plot.view
    assert zoomed[1] - zoomed[0] == pytest.approx(0.85 * (home[1] - home[0]))
    assert zoomed[3] - zoomed[2] == pytest.approx(0.85 * (home[3] - home[2]))
    assert (zoomed[0] + zoomed[1]) / 2 == pytest.approx((home[0] + home[1]) / 2)
    # Ctrl limits the zoom to the frequency axis.
    wheel = QWheelEvent(centre, plot.mapToGlobal(centre), QPoint(0, 0), QPoint(0, -120), Qt.NoButton,
                        Qt.ControlModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(plot, wheel)
    assert plot.view[1] - plot.view[0] == pytest.approx(home[1] - home[0])
    assert plot.view[2:] == zoomed[2:]

    before = plot.view
    start = QPoint(int(centre.x()), int(centre.y()))
    QTest.mousePress(plot, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(plot, start + QPoint(50, 20))
    QTest.mouseRelease(plot, Qt.LeftButton, Qt.NoModifier, start + QPoint(50, 20))
    shift = 50 / rect.width() * (before[1] - before[0])
    assert plot.view[0] == pytest.approx(before[0] - shift)
    assert plot.view[2] > before[2] and plot.modified

    # A changed setting keeps the view the user chose.
    tab.axis.setCurrentIndex(tab.axis.findData(LINEAR_AXIS))
    tab.wait()
    assert plot.view[0] == pytest.approx(before[0] - shift)
    QTest.keyClick(plot, Qt.Key_D, Qt.ControlModifier)
    assert plot.view == pytest.approx(plot.default_view()) and not plot.modified
    plot.zoom(0.5, QPointF(centre))
    tab.plot_reset()
    assert plot.view == pytest.approx(plot.default_view())


def test_saving_writes_interpolated_data_with_current_names_and_order(window, application, monkeypatch):
    tab = window.interpolation
    window.pages.setCurrentIndex(0)
    window.edit_name(0, "RENAMED")
    window.sort_ports(False)
    assert [port.name for port in window.model.ports] == ["CH1", "CH2", "RENAMED"]
    window.pages.setCurrentWidget(tab)
    tab.wait()
    assert [tab.proxy.index(row, 2).data() for row in range(3)] == ["CH1", "CH2", "RENAMED"]
    linear_bands(tab)
    target = window.document.source.with_name("result.s3p")
    monkeypatch.setattr(window, "choose_target", lambda tag, title: target)
    QTest.keyClick(window, Qt.Key_S, Qt.ControlModifier)
    if not target.exists():
        window.save_current()
    settle(window, application)
    assert window.undo_stack.isClean()
    result = read_touchstone(target)
    assert [port.name for port in result.ports] == ["CH1", "CH2", "RENAMED"]
    text = target.read_text()
    assert "! Source file: example.s3p" in text and "3 from the source, 5 added, 8 in total." in text
    assert "! Band 1: 100 Hz to 3 kHz, step 500 Hz, 5 new points." in text
    assert "Added points are interpolated values, not solver results." in text
    saved = load_network(result)
    assert saved.frequencies.tolist() == [100, 600, 1000, 1100, 1600, 2100, 2600, 3000]
    original = tab.network.matrices
    order = [2, 1, 0]
    assert np.allclose(saved.matrices[1], ((4 * original[0] + 5 * original[1]) / 9)[np.ix_(order, order)])
    assert np.allclose(saved.matrices[4], (0.7 * original[1] + 0.3 * original[2])[np.ix_(order, order)])
    assert np.array_equal(saved.matrices[2], original[1][np.ix_(order, order)])


def test_saving_uses_an_edit_that_is_still_open_or_waiting(window, application, monkeypatch):
    tab = window.interpolation
    linear_bands(tab)
    target = window.document.source.with_name("pending.s3p")
    monkeypatch.setattr(window, "choose_target", lambda tag, title: target)
    # An edit followed at once by a save: the debounce timer has not fired yet.
    tab.bands.item(0, 3).setText("1k")
    assert tab.timer.isActive()
    window.save_current()
    settle(window, application)
    assert load_network(read_touchstone(target)).frequencies.tolist() == [100, 1000, 1100, 2100, 3000]
    # A cell editor that is still open when the save starts.
    item = tab.bands.item(0, 3)
    tab.bands.setCurrentItem(item)
    tab.bands.editItem(item)
    editor = tab.bands.viewport().findChild(QLineEdit)
    editor.setText("250")
    window.save_current()
    settle(window, application)
    assert tab.bands.item(0, 3).text() == "250 Hz"
    assert len(load_network(read_touchstone(target)).frequencies) == 14


def test_port_shortcuts_stay_on_the_port_page(window, application):
    tab = window.interpolation
    window.pages.setCurrentIndex(0)
    window.table.selectRow(0)
    window.pages.setCurrentWidget(tab)
    tab.quantity.setFocus()
    before = [port.name for port in window.model.ports]
    QTest.keyClick(tab.quantity, Qt.Key_Down, Qt.AltModifier)
    application.processEvents()
    assert [port.name for port in window.model.ports] == before and window.undo_stack.isClean()
    popup = application.activePopupWidget()
    if popup is not None:
        popup.close()


def test_large_files_wait_for_the_load_button_and_new_files_reset_the_tab(window, application, monkeypatch):
    tab = window.interpolation
    monkeypatch.setattr(tab_module, "AUTO_LOAD_BYTES", 0)
    other = window.document.source.with_name("other.s1p")
    other.write_text("# Hz S RI R 50\n0 0.6 0\n10 0.5 0\n20 0.25 0\n")
    window.open_file(str(other))
    settle(window, application)
    assert window.document.source == other and "other.s1p" in window.windowTitle()
    assert tab.network is None and tab.stack.currentIndex() == 0
    assert not tab.load_button.isHidden() and "파일 크기" in tab.prompt.text()
    assert not window.save_button.isEnabled()
    tab.load_button.click()
    settle(window, application)
    assert tab.network.frequencies.tolist() == [0.0, 10.0, 20.0] and tab.stack.currentIndex() == 1
    assert tab.model.rowCount() == 1 and tab.ri_ports().tolist() == [True]
    # The 0-1 kHz linear count is 0, 0.2, ... 20 Hz here, but nothing may land between 0 Hz and 10 Hz.
    assert np.allclose(tab.plan.frequencies, np.concatenate(([0.0], np.linspace(10, 20, 51))))
    assert window.save_button.isEnabled()


def test_failed_load_returns_to_the_prompt_without_retrying(window, application, monkeypatch):
    tab = window.interpolation
    errors = []
    monkeypatch.setattr(window, "show_error", errors.append)
    broken = window.document.source.with_name("broken.s1p")
    broken.write_text("# Hz S RI R 50\n10 0.5 0\n5 0.25 0\n")
    window.open_file(str(broken))
    settle(window, application)
    assert len(errors) == 1 and "increasing" in errors[0]
    assert tab.network is None and tab.stack.currentIndex() == 0 and not tab.load_button.isHidden()


def test_memory_limited_network_still_plans_and_exports(window, application):
    tab = window.interpolation
    network = load_network(window.document, memory_limit=1)
    assert network.matrices is None
    tab.set_network(network)
    tab.wait()
    assert tab.plot.traces == [] and "메모리" in tab.plot.message
    assert [region.label for region in tab.plot.regions] == ["1 · Linear 101", "2 · Decade 100"]
    assert "위상 변화:" in tab.report.text() and "점검 생략" in tab.report.text()
    assert tab.can_export() and window.save_button.isEnabled()
