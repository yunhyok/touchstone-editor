import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QItemSelectionModel
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


def test_resize_round_trip_keeps_both_editors_visible(window, application):
    window.show()
    window.resize(683, 352)
    application.processEvents()
    assert window.width() == 683
    assert window.height() == 352
    assert window.tabs.count() == 2
    assert window.table.isVisible()
    window.resize(1200, 800)
    application.processEvents()
    assert window.port_panel.isVisible()
    assert window.edit_panel.isVisible()
    assert min(window.splitter.sizes()) > 0


def test_extensionless_existing_target_still_prompts(window, application, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import touchstone_editor.app as app_module

    existing = window.document.source.with_name("existing.s3p")
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
