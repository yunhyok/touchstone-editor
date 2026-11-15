import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

from touchstone_editor.app import MainWindow
from touchstone_editor.core import read_touchstone
from touchstone_editor.theme import apply_theme


@pytest.fixture(scope="session")
def application():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


@pytest.fixture
def window(application, tmp_path):
    source = tmp_path / "example.s3p"
    source.write_text("! Port[1] = CH10\n! Port[2] = CH2\n! Port[3] = CH1\n"
                      "# Hz S RI R 50\n"
                      "1 11 0 12 0 13 0\n21 0 22 0 23 0\n31 0 32 0 33 0\n")
    window = MainWindow()
    window.load_document(read_touchstone(source))
    yield window
    window.undo_stack.setClean()
    window.close()


def select(window, *rows):
    selection = window.table.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(window.proxy.index(row, 0),
                         QItemSelectionModel.Select | QItemSelectionModel.Rows)


def test_sort_manual_move_and_undo_keep_source_mapping(window):
    window.sort_ports(False)
    assert [p.source_index for p in window.model.ports] == [3, 2, 1]
    select(window, 0, 1)
    window.move_selected(1)
    assert [p.source_index for p in window.model.ports] == [1, 3, 2]
    window.undo_stack.undo()
    assert [p.source_index for p in window.model.ports] == [3, 2, 1]
    window.undo_stack.redo()
    assert [p.source_index for p in window.model.ports] == [1, 3, 2]
    window.position.setValue(1)
    select(window, 2)
    window.move_to_position()
    assert [p.source_index for p in window.model.ports] == [2, 1, 3]


def test_preview_selected_only_apply_and_undo(window):
    select(window, 1)
    window.scope.setCurrentIndex(1)
    window.prefix.setText("RX_")
    assert window.preview.rowCount() == 1
    assert window.preview.item(0, 1).text() == "RX_CH2"
    window.apply_names()
    assert [p.name for p in window.model.ports] == ["CH10", "RX_CH2", "CH1"]
    assert window.prefix.text() == ""
    window.undo_stack.undo()
    assert [p.name for p in window.model.ports] == ["CH10", "CH2", "CH1"]


def test_restore_selection_emits_once_and_preserves_mapping(window):
    from PySide6.QtTest import QSignalSpy

    window.sort_ports(False)
    spy = QSignalSpy(window.table.selectionModel().selectionChanged)
    window.restore_selection({1, 3})
    assert spy.count() == 1
    assert {window.model.ports[row].source_index for row in window.selected_rows()} == {1, 3}


def test_regex_preview_errors_and_selected_undo(window):
    select(window, 0)
    window.scope.setCurrentIndex(1)
    window.operation.setCurrentIndex(window.operation.findData("regex"))
    window.find_text.setText("[")
    assert not window.apply_button.isEnabled()
    assert "정규식" in window.preview_label.text()
    window.find_text.setText(r"^CH([0-9]+)$")
    window.replacement.setText("SIGNAL/$1")
    assert window.preview.item(0, 1).text() == "SIGNAL/10"
    assert window.apply_button.isEnabled()
    window.apply_names()
    assert [p.name for p in window.model.ports] == ["SIGNAL/10", "CH2", "CH1"]
    window.undo_stack.undo()
    assert [p.name for p in window.model.ports] == ["CH10", "CH2", "CH1"]


def test_filter_edit_updates_correct_source_port(window):
    window.search.setText("CH2")
    assert window.proxy.rowCount() == 1
    window.proxy.setData(window.proxy.index(0, 3), "NEW")
    assert window.model.ports[1].name == "NEW"
    assert window.model.ports[0].name == "CH10"
    assert not window.move_button.isEnabled()


def test_character_range_preview_selected_apply_and_undo(window):
    select(window, 0, 1)
    window.scope.setCurrentIndex(1)
    window.operation.setCurrentIndex(window.operation.findData("range"))
    assert window.find_text.isHidden()
    assert not window.range_start.isHidden()
    window.range_end.setValue(2)
    assert window.preview.item(0, 1).text() == "10"
    window.replacement.setText("SIGNAL_")
    assert window.preview.item(0, 1).text() == "SIGNAL_10"
    assert window.preview.rowCount() == 2
    window.apply_names()
    assert [p.name for p in window.model.ports] == ["SIGNAL_10", "SIGNAL_2", "CH1"]
    assert window.range_start.value() == window.range_end.value() == 1
    assert window.range_start.isHidden()
    assert not window.find_text.isHidden()
    window.undo_stack.undo()
    assert [p.name for p in window.model.ports] == ["CH10", "CH2", "CH1"]
    window.undo_stack.redo()
    assert [p.name for p in window.model.ports] == ["SIGNAL_10", "SIGNAL_2", "CH1"]


def test_character_range_invalid_or_empty_result_blocks_apply(window):
    window.operation.setCurrentIndex(window.operation.findData("range"))
    window.range_start.setValue(3)
    assert not window.apply_button.isEnabled()
    assert window.preview.rowCount() == 0
    window.range_end.setValue(3)
    window.range_start.setValue(1)
    assert not window.apply_button.isEnabled()
    assert "비어" in window.preview_label.text()
    assert [p.name for p in window.model.ports] == ["CH10", "CH2", "CH1"]


@pytest.mark.parametrize("size", [(683, 352), (600, 352)])
def test_resize_round_trip_keeps_both_editors_visible(window, application, size):
    window.show()
    window.resize(*size)
    application.processEvents()
    assert (window.width(), window.height()) == size
    assert window.table.viewport().height() >= 34
    assert window.tabs.count() == 2
    assert window.table.isVisible()
    window.resize(1200, 800)
    application.processEvents()
    assert window.port_panel.isVisible()
    assert window.edit_panel.isVisible()
    assert min(window.splitter.sizes()) > 0


@pytest.mark.parametrize("count", [2, 3])
def test_extensionless_existing_target_still_prompts(window, application, monkeypatch, count):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import touchstone_editor.app as app_module

    if count == 2:
        window.include_ports([1], False)
    existing = window.document.source.with_name(f"existing.s{count}p")
    existing.write_bytes(b"keep this output")
    dialogs, prompts, exports = [], [], []

    class SaveDialog(QFileDialog):
        def __init__(self, *args):
            super().__init__(*args)
            self.setOption(QFileDialog.DontUseNativeDialog)
            dialogs.append(self)

    monkeypatch.setattr(app_module, "QFileDialog", SaveDialog)
    monkeypatch.setattr(window, "start_job", lambda *args: exports.append(args))

    def drive_dialog():
        dialog = dialogs[0]
        dialog.selectFile(str(existing.with_suffix("")))

        def decline():
            box = application.activeModalWidget()
            if isinstance(box, QMessageBox):
                prompts.append(box.text())
                box.done(QMessageBox.No)
            QTimer.singleShot(0, dialog.reject)

        QTimer.singleShot(0, decline)
        dialog.accept()

    QTimer.singleShot(0, drive_dialog)
    window.save_file()
    application.processEvents()
    assert prompts
    assert not exports
    assert existing.read_bytes() == b"keep this output"


def test_output_checks_follow_filter_sort_rename_and_undo(window):
    assert all(port.included for port in window.model.ports)
    window.search.setText("CH2")
    assert window.proxy.setData(window.proxy.index(0, 0), Qt.Unchecked, Qt.CheckStateRole)
    assert [port.source_index for port in window.output_ports()] == [1, 3]
    window.search.clear()
    window.sort_ports(False)
    window.edit_name(0, "RENAMED")
    assert [port.source_index for port in window.output_ports()] == [3, 1]
    assert [window.model.data(window.model.index(row, 0)) for row in range(3)] == [1, "—", 2]
    window.undo_stack.undo()
    window.undo_stack.undo()
    assert [port.source_index for port in window.output_ports()] == [1, 3]
    window.undo_stack.undo()
    assert len(window.output_ports()) == 3
    window.undo_stack.redo()
    assert [port.source_index for port in window.output_ports()] == [1, 3]


def test_output_selection_only_and_empty_selection_blocks_save(window, monkeypatch):
    select(window, 2)
    window.include_only_selected()
    assert [port.source_index for port in window.output_ports()] == [3]
    window.include_ports(range(3), False)
    assert not window.save_button.isEnabled()
    errors = []
    monkeypatch.setattr(window, "show_error", errors.append)
    monkeypatch.setattr(window, "choose_target", lambda *args: pytest.fail("Empty output opened save dialog"))
    window.save_file()
    assert errors and "하나 이상" in errors[0]
    window.undo_stack.undo()
    assert window.save_button.isEnabled()


def test_output_check_can_toggle_repeatedly_with_space(window, application):
    from PySide6.QtTest import QTest

    window.show()
    window.table.setFocus()
    window.table.setCurrentIndex(window.proxy.index(1, 0))
    QTest.keyClick(window.table, Qt.Key_Space)
    application.processEvents()
    assert not window.model.ports[1].included
    QTest.keyClick(window.table, Qt.Key_Space)
    application.processEvents()
    assert window.model.ports[1].included


def test_regular_save_uses_checked_ports_and_current_names(window, monkeypatch, tmp_path):
    from touchstone_editor.interpolation import load_network
    import touchstone_editor.app as app_module

    select(window, 0, 2)
    window.include_only_selected()
    window.sort_ports(False)
    window.edit_name(0, "RENAMED")
    target = tmp_path / "subset.s2p"
    monkeypatch.setattr(window, "choose_target", lambda *args: target)
    monkeypatch.setattr(app_module.QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(window, "start_job", lambda action, saved, message: saved(action(None, None)))
    window.save_file()
    result = read_touchstone(target)
    assert [port.name for port in result.ports] == ["RENAMED", "CH10"]
    assert load_network(result).matrices[0].real.tolist() == [[33, 31], [13, 11]]


@pytest.mark.parametrize("edit", ["name", "order"])
def test_subset_save_keeps_excluded_edits_unsaved(window, monkeypatch, tmp_path, edit):
    import touchstone_editor.app as app_module

    if edit == "name":
        window.edit_name(1, "UNSAVED")
    else:
        select(window, 1)
        window.move_selected(1)
    window.include_ports([1, 2], False)
    target = tmp_path / "subset.s1p"
    monkeypatch.setattr(window, "choose_target", lambda *args: target)
    monkeypatch.setattr(app_module.QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(window, "start_job", lambda action, saved, message: saved(action(None, None)))
    window.save_file()
    assert not window.undo_stack.isClean()
    assert "UNSAVED" not in target.read_text()
    prompts = []
    def decline(*args):
        prompts.append(args)
        return app_module.QMessageBox.No
    monkeypatch.setattr(app_module.QMessageBox, "question", decline)
    assert not window.confirm_discard() and prompts
