"""Desktop port editor; numeric data stays in the streaming core."""

import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
                           Qt, QThread, QTimer, Signal)
from PySide6.QtGui import QAction, QIcon, QKeySequence, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                              QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                              QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu,
                              QMessageBox, QProgressBar, QPushButton, QScrollArea,
                              QSizePolicy, QSpinBox, QSplitter, QStackedWidget,
                              QTabWidget, QTableView, QTableWidget,
                              QTableWidgetItem, QVBoxLayout, QWidget)

from . import __version__
from .core import CancelledError, export_touchstone, read_touchstone
from .names import NameRule, natural_key, validate_names
from .theme import apply_theme


class PortModel(QAbstractTableModel):
    edit_requested = Signal(int, str)
    headers = ["새 포트", "원본 포트", "원본 이름", "저장할 채널명 · 더블클릭 편집"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ports = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.ports)

    def columnCount(self, parent=QModelIndex()):
        return 4

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        port = self.ports[index.row()]
        if role == Qt.ToolTipRole and index.column() == 3:
            return f"원본 포트 {port.source_index}: {port.original_name}\n저장 이름: {port.name}"
        if role in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return [index.row() + 1, port.source_index, port.original_name, port.name][index.column()]
        if role == Qt.TextAlignmentRole and index.column() < 2:
            return Qt.AlignCenter

    def flags(self, index):
        flags = super().flags(index)
        return flags | Qt.ItemIsEditable if index.column() == 3 else flags

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid() and index.column() == 3 and role == Qt.EditRole:
            self.edit_requested.emit(index.row(), str(value))
            return True
        return False

    def set_ports(self, ports):
        self.beginResetModel()
        self.ports = [replace(port) for port in ports]
        self.endResetModel()


class SnapshotCommand(QUndoCommand):
    def __init__(self, model, after, title):
        super().__init__(title)
        self.model = model
        self.before = [replace(port) for port in model.ports]
        self.after = [replace(port) for port in after]

    def redo(self):
        self.model.set_ports(self.after)

    def undo(self):
        self.model.set_ports(self.before)


class FileJob(QThread):
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()
    progress = Signal(int)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action
        self.stop = Event()

    def run(self):
        try:
            self.result.emit(self.action(self.progress.emit, self.stop.is_set))
        except CancelledError:
            self.cancelled.emit()
        except Exception as error:
            self.error.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.document = None
        self.job = None
        self.model = PortModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(100)
        self.setWindowTitle(f"Touchstone Editor {__version__}")
        self.setWindowIcon(QIcon(str(Path(__file__).resolve().parent.parent / "assets" / "icon.ico")))
        self.setMinimumSize(600, 300)
        self.build_ui()
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, available.width() - 24), min(820, available.height() - 32))
        self.model.edit_requested.connect(self.edit_name)
        self.model.modelReset.connect(self.refresh)
        self.undo_stack.cleanChanged.connect(self.update_title)
        self.refresh()

    @staticmethod
    def button(text, callback, primary=False):
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.clicked.connect(callback)
        return button

    def build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 16, 20, 12)
        outer.setSpacing(10)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("Touchstone Editor")
        title.setObjectName("title")
        self.subtitle = QLabel("PowerSI → Ansys  /  포트 순서와 채널명 편집")
        self.subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(self.subtitle)
        heading.addLayout(titles, 1)
        self.open_button = self.button("파일 열기", self.open_file)
        self.save_button = self.button("Ansys용으로 저장…", self.save_file, True)
        heading.addWidget(self.open_button)
        heading.addWidget(self.save_button)
        outer.addLayout(heading)
        self.file_label = QLabel(".sNp 파일을 열어 시작하세요. 원본과 다른 파일로 저장합니다.")
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.file_label)
        self.meta_label = QLabel("Touchstone 1.x  ·  S-parameter  ·  RI / MA / DB")
        self.meta_label.setObjectName("hint")
        outer.addWidget(self.meta_label)

        self.splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        self.port_panel = left
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("포트 번호·원본 이름·채널명 검색")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter_changed)
        row.addWidget(self.search, 1)
        sort_button = QPushButton("정렬")
        sort_menu = QMenu(sort_button)
        sort_menu.addAction("채널명 A → Z", lambda: self.sort_ports(False))
        sort_menu.addAction("채널명 Z → A", lambda: self.sort_ports(True))
        sort_menu.addAction("원본 순서 복원", self.restore_order)
        sort_button.setMenu(sort_menu)
        row.addWidget(sort_button)
        self.undo_action = self.undo_stack.createUndoAction(self, "실행 취소")
        self.redo_action = self.undo_stack.createRedoAction(self, "다시 실행")
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.addActions([self.undo_action, self.redo_action])
        self.undo_button = self.button("실행 취소", self.undo_stack.undo)
        self.redo_button = self.button("다시 실행", self.undo_stack.redo)
        self.undo_stack.canUndoChanged.connect(self.undo_button.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.redo_button.setEnabled)
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        left_layout.addLayout(row)
        move_row = QHBoxLayout()
        self.up_button = self.button("↑ 위로", lambda: self.move_selected(-1))
        self.down_button = self.button("↓ 아래로", lambda: self.move_selected(1))
        self.position = QSpinBox()
        self.position.setRange(1, 1)
        self.position.setPrefix("위치 ")
        self.position.setAccessibleName("이동할 포트 위치")
        self.move_button = self.button("선택 포트 이동", self.move_to_position)
        for widget in (self.up_button, self.down_button, self.position, self.move_button):
            move_row.addWidget(widget)
        move_row.addStretch()
        left_layout.addLayout(move_row)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 260)
        self.table.setColumnWidth(3, 320)
        header.setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self.refresh_preview)
        left_layout.addWidget(self.table, 1)
        self.selection_label = QLabel()
        self.selection_label.setObjectName("hint")
        left_layout.addWidget(self.selection_label)

        self.edit_panel = QScrollArea()
        self.edit_panel.setWidgetResizable(True)
        self.edit_panel.setMinimumWidth(300)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        group = QGroupBox("채널명 일괄 편집")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.scope = QComboBox()
        self.scope.addItems(["전체 포트", "선택한 포트"])
        self.operation = QComboBox()
        for label, value in [("찾아 바꾸기", "replace"), ("지정 문자 삭제", "delete"),
                             ("구분자 앞부분만 유지", "before"), ("구분자 뒷부분만 유지", "after"),
                             ("부품 정보 제거 · 채널만", "channel")]:
            self.operation.addItem(label, value)
        self.find_text = QLineEdit()
        self.find_text.setPlaceholderText("문자열 또는 구분자")
        self.find_text.setToolTip("일반 문자열 기준 · 정규식 미지원")
        self.replacement = QLineEdit()
        self.replacement.setPlaceholderText("비워 두면 삭제")
        self.prefix = QLineEdit()
        self.suffix = QLineEdit()
        self.whitespace = QCheckBox("모든 공백 삭제")
        self.special = QCheckBox("특수문자 삭제 (_ 유지)")
        self.special.setToolTip("문자·숫자·밑줄을 유지하고 나머지 특수문자를 삭제합니다.")
        form.addRow("적용 범위", self.scope)
        form.addRow("작업", self.operation)
        form.addRow("찾기 / 구분자", self.find_text)
        form.addRow("바꿀 내용", self.replacement)
        form.addRow("앞에 붙이기", self.prefix)
        form.addRow("뒤에 붙이기", self.suffix)
        form.addRow(self.whitespace)
        form.addRow(self.special)
        for field in (self.find_text, self.replacement, self.prefix, self.suffix):
            field.textChanged.connect(self.refresh_preview)
        for combo in (self.scope, self.operation):
            combo.currentIndexChanged.connect(self.refresh_preview)
        for check in (self.whitespace, self.special):
            check.toggled.connect(self.refresh_preview)
        right_layout.addWidget(group)
        preview_group = QGroupBox("적용 전 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        self.preview = QTableWidget(0, 2)
        self.preview.setHorizontalHeaderLabels(["현재", "변경 후"])
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview.setAlternatingRowColors(True)
        self.preview.verticalHeader().hide()
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview.setMinimumHeight(180)
        preview_layout.addWidget(self.preview)
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        self.apply_button = self.button("이름 변경 적용", self.apply_names, True)
        preview_layout.addWidget(self.apply_button)
        right_layout.addWidget(preview_group, 1)
        hint = QLabel("포트 이동: 행 선택 후 위/아래 또는 위치 지정\n"
                      "여러 행 선택: Ctrl / Shift\n"
                      "저장 시 포트 이름과 S 행렬의 행·열을 함께 변경")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        self.edit_panel.setWidget(right)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.edit_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([850, 350])
        self.tabs = QTabWidget()
        self.content = QStackedWidget()
        self.content.addWidget(self.splitter)
        self.content.addWidget(self.tabs)
        self.compact = False
        outer.addWidget(self.content, 1)
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.cancel_button = self.button("취소", self.cancel_job)
        self.cancel_button.hide()
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.cancel_button)
        outer.addLayout(progress_row)
        self.setCentralWidget(central)
        self.statusBar().showMessage("준비")
        for key, callback in [("Ctrl+O", self.open_file), ("Ctrl+Shift+S", self.save_file),
                              ("Alt+Up", lambda: self.move_selected(-1)),
                              ("Alt+Down", lambda: self.move_selected(1))]:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    def selected_rows(self):
        return sorted({self.proxy.mapToSource(index).row()
                       for index in self.table.selectionModel().selectedRows()})

    def restore_selection(self, source_indices):
        from PySide6.QtCore import QItemSelectionModel
        selection = self.table.selectionModel()
        selection.clearSelection()
        for row, port in enumerate(self.model.ports):
            if port.source_index in source_indices:
                index = self.proxy.mapFromSource(self.model.index(row, 0))
                if index.isValid():
                    selection.select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        selected = selection.selectedRows()
        if selected:
            self.table.scrollTo(selected[0])

    def commit(self, ports, title):
        if self.job or ports == self.model.ports:
            return
        selected = {self.model.ports[row].source_index for row in self.selected_rows()}
        self.undo_stack.push(SnapshotCommand(self.model, ports, title))
        self.restore_selection(selected)

    def edit_name(self, row, name):
        name = name.strip()
        try:
            validate_names([name])
        except ValueError as error:
            self.show_error(str(error))
            return
        ports = [replace(port) for port in self.model.ports]
        ports[row].name = name
        self.commit(ports, "채널명 편집")

    def sort_ports(self, reverse):
        self.commit(sorted(self.model.ports, key=lambda port: natural_key(port.name),
                           reverse=reverse), "채널명 정렬")

    def restore_order(self):
        self.commit(sorted(self.model.ports, key=lambda port: port.source_index), "원본 순서 복원")

    def move_selected(self, direction):
        if self.search.text() or self.job:
            return
        rows = set(self.selected_rows())
        ports = list(self.model.ports)
        for row in sorted(rows, reverse=direction > 0):
            target = row + direction
            if 0 <= target < len(ports) and target not in rows:
                ports[row], ports[target] = ports[target], ports[row]
                rows.remove(row)
                rows.add(target)
        self.commit(ports, "포트 이동")

    def move_to_position(self):
        if self.search.text() or self.job:
            return
        selected = set(self.selected_rows())
        if not selected:
            return
        moving = [port for row, port in enumerate(self.model.ports) if row in selected]
        other = [port for row, port in enumerate(self.model.ports) if row not in selected]
        index = min(self.position.value() - 1, len(other))
        self.commit(other[:index] + moving + other[index:], "위치 지정 이동")

    def filter_changed(self, text):
        self.proxy.setFilterFixedString(text)
        self.refresh()

    def name_changes(self):
        rule = NameRule(self.operation.currentData(), self.find_text.text(),
                        self.replacement.text(), self.prefix.text(), self.suffix.text(),
                        self.whitespace.isChecked(), self.special.isChecked())
        rows = self.selected_rows() if self.scope.currentIndex() == 1 else range(len(self.model.ports))
        ports = [replace(port) for port in self.model.ports]
        changed = []
        for row in rows:
            name = rule.apply(ports[row].name)
            if name != ports[row].name:
                changed.append((ports[row].name, name))
                ports[row].name = name
        return ports, changed

    def refresh_preview(self, *args):
        if not hasattr(self, "preview"):
            return
        ports, changed = self.name_changes()
        self.preview.setRowCount(min(len(changed), 80))
        for row, values in enumerate(changed[:80]):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.preview.setItem(row, column, item)
        valid = True
        message = f"{len(changed)}개 변경 예정"
        if len(changed) > 80:
            message += " · 처음 80개 표시"
        try:
            validate_names([port.name for port in ports])
        except ValueError as error:
            valid, message = False, str(error)
        names = [port.name for port in ports]
        if valid and len(set(names)) < len(names):
            message += "\n같은 이름이 있습니다. 포트 번호로 구분됩니다."
        self.preview_label.setText(message)
        self.apply_button.setEnabled(bool(changed) and valid and self.job is None)
        self.selection_label.setText(f"{self.proxy.rowCount()} / {len(ports)}개 표시  ·  "
                                     f"{len(self.selected_rows())}개 선택" +
                                     ("  ·  수동 이동은 검색 해제 후 가능" if self.search.text() else ""))
        operation = self.operation.currentData()
        self.find_text.setEnabled(operation != "channel")
        self.replacement.setEnabled(operation == "replace")

    def apply_names(self):
        ports, changed = self.name_changes()
        try:
            validate_names([port.name for port in ports])
        except ValueError as error:
            self.show_error(str(error))
            return
        self.commit(ports, f"채널명 {len(changed)}개 변경")
        # Clearing the recipe prevents accidental repeated prefix/suffix application.
        for field in (self.find_text, self.replacement, self.prefix, self.suffix):
            field.clear()
        self.whitespace.setChecked(False)
        self.special.setChecked(False)
        self.operation.setCurrentIndex(0)
        self.refresh_preview()

    def refresh(self, *args):
        ready = self.document is not None and self.job is None
        self.save_button.setEnabled(ready)
        self.content.setEnabled(ready)
        self.open_button.setEnabled(self.job is None)
        self.undo_button.setEnabled(ready and self.undo_stack.canUndo())
        self.redo_button.setEnabled(ready and self.undo_stack.canRedo())
        self.undo_action.setEnabled(ready and self.undo_stack.canUndo())
        self.redo_action.setEnabled(ready and self.undo_stack.canRedo())
        for widget in (self.up_button, self.down_button, self.position, self.move_button):
            widget.setEnabled(ready and not self.search.text())
        self.position.setMaximum(max(1, len(self.model.ports)))
        self.refresh_preview()

    def update_title(self, *args):
        suffix = " *" if not self.undo_stack.isClean() else ""
        self.setWindowTitle(f"Touchstone Editor {__version__}{suffix}")

    def confirm_discard(self):
        if self.undo_stack.isClean():
            return True
        return QMessageBox.question(self, "저장하지 않은 변경", "저장하지 않은 변경을 버리시겠습니까?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def open_file(self, path=None):
        if self.job or not self.confirm_discard():
            return
        if not isinstance(path, (str, Path)):
            path, _ = QFileDialog.getOpenFileName(self, "Touchstone 파일 열기", "",
                                                  "Touchstone (*.s*p *.S*P);;모든 파일 (*)")
        if path:
            self.start_job(lambda progress, cancel: read_touchstone(path, progress, cancel),
                           self.load_document, "파일 검사 중…")

    def load_document(self, document):
        self.document = document
        self.search.clear()
        self.undo_stack.clear()
        self.model.set_ports(document.ports)
        self.file_label.setText(str(document.source))
        self.file_label.setToolTip(str(document.source))
        self.meta_label.setText(f"{document.port_count} ports  ·  {document.point_count:,} 주파수 지점  ·  "
                               f"{document.option_line.strip()}")
        self.statusBar().showMessage("파일 검사 완료 · 채널명을 확인하고 저장하세요.")
        self.refresh()

    def save_file(self):
        if self.job or not self.document:
            return
        try:
            validate_names([port.name for port in self.model.ports])
        except ValueError as error:
            self.show_error(str(error))
            return
        suffix = f".s{self.document.port_count}p"
        suggested = self.document.source.with_name(self.document.source.stem + "_ansys" + suffix)
        dialog = QFileDialog(self, "Ansys용 Touchstone 저장", str(suggested.parent),
                             f"Touchstone (*{suffix})")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix(suffix.lstrip("."))
        dialog.selectFile(suggested.name)
        if dialog.exec() != QFileDialog.Accepted:
            return
        target = Path(dialog.selectedFiles()[0])
        ports = [replace(port) for port in self.model.ports]

        def save(progress, cancel):
            export_touchstone(self.document, ports, target, progress, cancel)
            return target

        def saved(result):
            self.undo_stack.setClean()
            self.statusBar().showMessage(f"저장 완료: {result}")
            QMessageBox.information(self, "저장 완료", f"Ansys용 파일을 저장했습니다.\n\n{result}")

        self.start_job(save, saved, "포트와 데이터 저장 중…")

    def start_job(self, action, completed, message):
        self.job = FileJob(action, self)
        self.job.progress.connect(self.progress_bar.setValue)
        self.job.result.connect(completed)
        self.job.error.connect(self.show_error)
        self.job.cancelled.connect(lambda: self.statusBar().showMessage("작업을 취소했습니다."))
        self.job.finished.connect(self.job_finished)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.statusBar().showMessage(message)
        self.refresh()
        self.job.start()

    def cancel_job(self):
        if self.job:
            self.job.stop.set()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("안전하게 취소하는 중…")

    def job_finished(self):
        self.job.deleteLater()
        self.job = None
        self.progress_bar.hide()
        self.cancel_button.hide()
        self.refresh()

    def show_error(self, message):
        self.statusBar().showMessage("작업 실패 · 원본 파일은 변경되지 않았습니다.")
        QMessageBox.warning(self, "작업을 완료하지 못했습니다", message)

    def closeEvent(self, event):
        if self.job:
            self.cancel_job()
            event.ignore()
            return
        if self.confirm_discard():
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "content"):
            return
        compact = self.width() < 1000
        if compact != self.compact:
            self.compact = compact
            if compact:
                self.tabs.addTab(self.port_panel, "포트 순서·이름")
                self.tabs.addTab(self.edit_panel, "일괄 이름 편집")
                self.content.setCurrentWidget(self.tabs)
            else:
                while self.tabs.count():
                    self.tabs.removeTab(0)
                self.splitter.addWidget(self.port_panel)
                self.splitter.addWidget(self.edit_panel)
                self.content.setCurrentWidget(self.splitter)
                self.port_panel.show()
                self.edit_panel.show()
                self.splitter.setSizes([self.width() - 400, 350])
        short = self.height() < 550
        self.subtitle.setVisible(not short)
        self.file_label.setVisible(not short)
        self.meta_label.setVisible(not short)
        self.table.setColumnHidden(2, compact)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Touchstone Editor")
    app.setApplicationVersion(__version__)
    apply_theme(app)
    window = MainWindow()
    window.show()
    if sys.argv[1:] == ["--smoke-test"]:
        QTimer.singleShot(0, app.quit)
    elif len(sys.argv) > 1:
        QTimer.singleShot(0, lambda: window.open_file(sys.argv[1]))
    sys.exit(app.exec())
