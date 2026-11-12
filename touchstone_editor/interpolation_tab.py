"""Interpolation tab: channel selection, frequency bands and the preview plot."""

import hashlib
from threading import Event

import numpy as np
from PySide6.QtCore import (QAbstractTableModel, QItemSelection, QItemSelectionModel, QModelIndex,
                           QSortFilterProxyModel, Qt, QThread, QTimer, Signal)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                              QLabel, QLineEdit, QMenu, QPushButton, QScrollArea, QSplitter, QStackedWidget, QTableView,
                              QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .core import CancelledError
from .interpolation import (DECADE_COUNT, DEFAULT_BANDS, LINEAR_COUNT, LINEAR_STEP, Band, build_plan,
                            compute_traces, estimate_size, format_frequency, parse_frequency)
from .names import natural_key
from .plot import COLORS, PlotWidget, Region, Trace

AUTO_LOAD_BYTES = 256 * 1024 * 1024
TRANSFER_LIMIT = 8
TRACE_LIMIT = 600
_MODES = ((LINEAR_COUNT, "Linear Count", "지점 수"), (DECADE_COUNT, "Decade Count", "decade당 지점 수"),
          (LINEAR_STEP, "Linear Step", "간격"))
_QUANTITIES = (("magnitude", "크기"), ("db", "dB"), ("phase", "위상 (°)"), ("real", "실수"), ("imaginary", "허수"))


def format_bytes(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class ChannelModel(QAbstractTableModel):
    """Ports in output order with a check mark for real/imaginary interpolation."""

    checks_changed = Signal()
    headers = ["RI 보간", "포트", "채널명"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ports = []
        self.checked = set()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.ports)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.DisplayRole:
            return self.headers[section]
        if role == Qt.ToolTipRole:
            return ["체크한 채널이 포함된 항은 실수·허수(RI) 선형 보간, 나머지는 크기·위상 보간",
                    "저장될 포트 번호", "저장될 채널명"][section]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        port = self.ports[index.row()]
        if index.column() == 0:
            if role == Qt.CheckStateRole:
                return Qt.Checked if port.source_index in self.checked else Qt.Unchecked
            return None
        if role == Qt.DisplayRole:
            return index.row() + 1 if index.column() == 1 else port.name
        if role == Qt.ToolTipRole:
            return f"원본 포트 {port.source_index}: {port.original_name}"
        if role == Qt.TextAlignmentRole and index.column() == 1:
            return Qt.AlignCenter

    def flags(self, index):
        flags = super().flags(index)
        return flags | Qt.ItemIsUserCheckable if index.column() == 0 else flags

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid() and index.column() == 0 and role == Qt.CheckStateRole:
            self.set_checked([self.ports[index.row()].source_index], Qt.CheckState(value) == Qt.Checked)
            return True
        return False

    def set_ports(self, ports):
        self.beginResetModel()
        self.ports = list(ports)
        self.endResetModel()

    def set_checked(self, sources, state):
        before = set(self.checked)
        self.checked = self.checked | set(sources) if state else self.checked - set(sources)
        if self.checked != before:
            if self.ports:
                self.dataChanged.emit(self.index(0, 0), self.index(len(self.ports) - 1, 0), [Qt.CheckStateRole])
            self.checks_changed.emit()


class ChannelProxy(QSortFilterProxyModel):
    text = ""

    def lessThan(self, left, right):
        model = self.sourceModel()
        first, second = model.ports[left.row()], model.ports[right.row()]
        if left.column() == 0:
            keys = [(port.source_index not in model.checked, row)
                    for port, row in ((first, left.row()), (second, right.row()))]
        elif left.column() == 2:
            keys = [(natural_key(port.name), row) for port, row in ((first, left.row()), (second, right.row()))]
        else:
            keys = [left.row(), right.row()]
        return keys[0] < keys[1]

    def filterAcceptsRow(self, row, parent):
        return (not self.text or self.text in self.sourceModel().ports[row].name.casefold()
                or self.text == str(row + 1))


class ChannelTable(QTableView):
    """Clicks in the check column toggle one channel; Space toggles every selected channel."""

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and index.column() == 0 and event.button() == Qt.LeftButton:
            # Checking a channel must not change which channels the plot shows.
            state = Qt.Unchecked if index.data(Qt.CheckStateRole) == Qt.Checked else Qt.Checked
            self.model().setData(index, state, Qt.CheckStateRole)
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Qt reports every second quick click as a double click.
        if self.indexAt(event.position().toPoint()).column() == 0:
            self.mousePressEvent(event)
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        rows = self.selectionModel().selectedRows()
        if event.key() == Qt.Key_Space and rows:
            proxy = self.model()
            model = proxy.sourceModel()
            sources = [model.ports[proxy.mapToSource(index).row()].source_index for index in rows]
            model.set_checked(sources, not all(source in model.checked for source in sources))
            event.accept()
            return
        super().keyPressEvent(event)


class TraceJob(QThread):
    """Evaluate the plotted entries without blocking the window."""

    done = Signal(object, object)
    failed = Signal(object, str)
    progress = Signal(object, int)

    def __init__(self, key, arguments, parent=None):
        super().__init__(parent)
        self.key = key
        self.arguments = arguments
        self.stop = Event()

    def run(self):
        try:
            values = compute_traces(*self.arguments, lambda percent: self.progress.emit(self.key, percent),
                                    self.stop.is_set)
        except CancelledError:
            return
        except Exception as error:
            self.failed.emit(self.key, str(error))
            return
        self.done.emit(self.key, values)


class InterpolationTab(QWidget):
    """Resample selected channels on banded frequency grids and preview the result."""

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.document = None
        self.network = None
        self.plan = None
        self.load_tried = False
        # Counts loaded data sets so results of an earlier file can never be shown for the next one.
        self.generation = 0
        self.jobs = []
        self.cache = {}
        self.wanted = None
        self.shown = None
        self.entries = []
        self.entry_total = 0
        self.hidden = False
        self.model = ChannelModel(self)
        self.proxy = ChannelProxy(self)
        self.proxy.setSourceModel(self.model)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.refresh)
        self.build_ui()
        self.model.checks_changed.connect(self.schedule)
        self.set_document(None)

    # Layout ----------------------------------------------------------------------------

    def build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        placeholder = QWidget()
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_layout.addStretch()
        self.prompt = QLabel()
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setWordWrap(True)
        placeholder_layout.addWidget(self.prompt)
        self.load_button = QPushButton("수치 데이터 불러오기")
        self.load_button.setProperty("primary", True)
        self.load_button.clicked.connect(self.load)
        placeholder_layout.addWidget(self.load_button, 0, Qt.AlignCenter)
        placeholder_layout.addStretch()
        self.stack.addWidget(placeholder)

        # Channels fill the left edge; the bands sit under the plot whose regions they define.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.build_channels())
        self.right = QSplitter(Qt.Vertical)
        self.right.addWidget(self.build_plot())
        self.right.addWidget(self.build_bands())
        self.right.setStretchFactor(0, 1)
        self.right.setStretchFactor(1, 0)
        self.right.setSizes([540, 190])
        self.splitter.addWidget(self.right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([330, 900])
        # A small window scrolls instead of squeezing the plot and tables into each other.
        self.splitter.setMinimumSize(940, 440)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.splitter)
        self.stack.addWidget(scroll)
        reset = QShortcut(QKeySequence("Ctrl+D"), self)
        reset.setContext(Qt.WidgetWithChildrenShortcut)
        reset.activated.connect(self.plot.reset_view)

    def build_channels(self):
        group = QGroupBox("채널")
        group.setMinimumWidth(240)
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("채널명·포트 번호 검색")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter_changed)
        row.addWidget(self.search, 1)
        check_button = QPushButton("RI 보간 대상")
        menu = QMenu(check_button)
        menu.addAction("전체 체크", lambda: self.check(None, True))
        menu.addAction("전체 해제", lambda: self.check(None, False))
        menu.addSeparator()
        menu.addAction("선택한 채널만 체크", self.check_only_selected)
        menu.addAction("선택한 채널 체크", lambda: self.check(self.selected_sources(), True))
        menu.addAction("선택한 채널 해제", lambda: self.check(self.selected_sources(), False))
        check_button.setMenu(menu)
        row.addWidget(check_button)
        layout.addLayout(row)
        self.table = ChannelTable()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(1, Qt.AscendingOrder)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(30)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(0, 84)
        self.table.setColumnWidth(1, 64)
        self.table.setAccessibleName("보간 및 그래프 채널 목록")
        self.table.selectionModel().selectionChanged.connect(self.schedule)
        layout.addWidget(self.table, 1)
        self.channel_label = QLabel()
        self.channel_label.setObjectName("hint")
        self.channel_label.setWordWrap(True)
        layout.addWidget(self.channel_label)
        return group

    def build_bands(self):
        group = QGroupBox("주파수 구간")
        columns = QHBoxLayout(group)
        self.bands = QTableWidget(0, 5)
        self.bands.setHorizontalHeaderLabels(["시작", "끝", "방식", "값", "적용 결과"])
        self.bands.setAlternatingRowColors(True)
        self.bands.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bands.setSelectionMode(QAbstractItemView.SingleSelection)
        self.bands.verticalHeader().setDefaultSectionSize(30)
        self.bands.verticalHeader().setMinimumWidth(24)
        header = self.bands.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        for column, width in enumerate((84, 84, 128, 84)):
            self.bands.setColumnWidth(column, width)
        self.bands.horizontalHeaderItem(0).setToolTip("예: 0, 1k, 10MHz, 2G")
        self.bands.horizontalHeaderItem(1).setToolTip("예: 0, 1k, 10MHz, 2G")
        self.bands.horizontalHeaderItem(3).setToolTip("Linear Count: 구간 전체 지점 수\n"
                                                      "Decade Count: decade당 지점 수\nLinear Step: 주파수 간격")
        self.bands.horizontalHeaderItem(4).setToolTip("원본 주파수 범위 안으로 줄인 실제 구간과 지점 수")
        self.bands.setAccessibleName("주파수 구간 설정")
        self.bands.setMinimumHeight(96)
        self.bands.itemChanged.connect(self.band_edited)
        columns.addWidget(self.bands, 7)
        layout = QVBoxLayout()
        columns.addLayout(layout, 3)
        row = QHBoxLayout()
        for text, callback in (("구간 추가", self.add_band), ("구간 삭제", self.remove_band),
                               ("기본값", self.reset_bands)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.keep_source = QCheckBox("원본 주파수 지점도 출력에 포함")
        self.keep_source.setToolTip("구간 지점에 더해 원본의 모든 주파수 지점을 원래 값 그대로 남깁니다.")
        self.keep_source.toggled.connect(self.schedule)
        layout.addWidget(self.keep_source)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addStretch()
        self.set_bands(DEFAULT_BANDS)
        return group

    def build_plot(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.parameter = QComboBox()
        self.parameter.addItem("Z-parameter", "Z")
        self.parameter.addItem("S-parameter", "S")
        self.parameter.setAccessibleName("표시할 파라미터")
        self.quantity = QComboBox()
        for value, label in _QUANTITIES:
            self.quantity.addItem(label, value)
        self.quantity.setAccessibleName("표시할 값")
        self.x_log = QCheckBox("X 로그")
        self.y_log = QCheckBox("Y 로그")
        self.x_log.setChecked(True)
        self.y_log.setChecked(True)
        self.show_source = QCheckBox("원본 지점")
        self.show_result = QCheckBox("보간 결과")
        self.show_source.setChecked(True)
        self.show_result.setChecked(True)
        self.transfer = QCheckBox("전달 항")
        self.transfer.setToolTip(f"선택한 채널 사이의 전달 항도 표시합니다. 채널 2~{TRANSFER_LIMIT}개 선택 시 적용됩니다.")
        reset = QPushButton("보기 초기화")
        reset.setToolTip("기본 위치·크기로 되돌리기 (Ctrl+D)")
        reset.clicked.connect(self.plot_reset)
        for widget in (self.parameter, self.quantity, self.x_log, self.y_log, self.show_source,
                       self.show_result, self.transfer):
            row.addWidget(widget)
        row.addStretch()
        row.addWidget(reset)
        layout.addLayout(row)
        self.plot = PlotWidget()
        self.plot.x_title, self.plot.x_unit = "주파수 (Hz)", "Hz"
        layout.addWidget(self.plot, 1)
        self.plot_label = QLabel()
        self.plot_label.setObjectName("hint")
        self.plot_label.setWordWrap(True)
        layout.addWidget(self.plot_label)
        self.parameter.currentIndexChanged.connect(self.schedule)
        self.quantity.currentIndexChanged.connect(self.quantity_changed)
        for check in (self.x_log, self.y_log):
            check.toggled.connect(self.scales_changed)
        for check in (self.show_source, self.show_result, self.transfer):
            check.toggled.connect(self.schedule)
        return panel

    # Bands -----------------------------------------------------------------------------

    def set_bands(self, bands):
        self.bands.blockSignals(True)
        self.bands.setRowCount(0)
        for band in bands:
            self.append_band(band)
        self.bands.blockSignals(False)
        self.schedule()

    def append_band(self, band):
        row = self.bands.rowCount()
        self.bands.insertRow(row)
        value = format_frequency(band.value) if band.mode == LINEAR_STEP else str(int(band.value))
        for column, text in ((0, format_frequency(band.start)), (1, format_frequency(band.stop)), (3, value)):
            self.bands.setItem(row, column, QTableWidgetItem(text))
        combo = QComboBox()
        for mode, label, hint in _MODES:
            combo.addItem(label, mode)
            combo.setItemData(combo.count() - 1, hint, Qt.ToolTipRole)
        combo.setCurrentIndex(combo.findData(band.mode))
        combo.currentIndexChanged.connect(lambda _, widget=combo: self.mode_changed(widget))
        self.bands.setCellWidget(row, 2, combo)
        result = QTableWidgetItem("")
        result.setFlags(result.flags() & ~Qt.ItemIsEditable)
        self.bands.setItem(row, 4, result)

    def add_band(self):
        rows = self.bands.rowCount()
        start = 0.0
        if rows:
            try:
                start = parse_frequency(self.bands.item(rows - 1, 1).text())
            except ValueError:
                pass
        stop = max(start * 10, 1e3)
        if self.network is not None and start < self.network.frequencies[-1]:
            stop = float(self.network.frequencies[-1])
        self.bands.blockSignals(True)
        self.append_band(Band(start, stop, LINEAR_COUNT, 101))
        self.bands.blockSignals(False)
        self.bands.selectRow(rows)
        self.schedule()

    def remove_band(self):
        row = self.bands.currentRow()
        if row < 0:
            row = self.bands.rowCount() - 1
        if row >= 0:
            self.bands.removeRow(row)
            self.schedule()

    def reset_bands(self):
        self.set_bands(DEFAULT_BANDS)

    def mode_changed(self, combo):
        for row in range(self.bands.rowCount()):
            if self.bands.cellWidget(row, 2) is combo:
                # A count and a step are not interchangeable, so each mode starts from its own default.
                value = {LINEAR_COUNT: "101", DECADE_COUNT: "100"}.get(combo.currentData())
                if value is None:
                    try:
                        span = (parse_frequency(self.bands.item(row, 1).text())
                                - parse_frequency(self.bands.item(row, 0).text()))
                        value = format_frequency(span / 100) if span > 0 else "1 MHz"
                    except ValueError:
                        value = "1 MHz"
                self.bands.blockSignals(True)
                self.bands.item(row, 3).setText(value)
                self.bands.blockSignals(False)
        self.schedule()

    def band_edited(self, item):
        mode = self.bands.cellWidget(item.row(), 2)
        if item.column() in (0, 1) or (item.column() == 3 and mode is not None and mode.currentData() == LINEAR_STEP):
            try:
                text = format_frequency(parse_frequency(item.text()))
            except ValueError:
                text = item.text()
            if text != item.text():
                self.bands.blockSignals(True)
                item.setText(text)
                self.bands.blockSignals(False)
        self.schedule()

    def read_bands(self):
        bands = []
        for row in range(self.bands.rowCount()):
            mode = self.bands.cellWidget(row, 2).currentData()
            try:
                start = parse_frequency(self.bands.item(row, 0).text())
                stop = parse_frequency(self.bands.item(row, 1).text())
                text = self.bands.item(row, 3).text()
                if mode == LINEAR_STEP:
                    value = parse_frequency(text)
                else:
                    try:
                        value = int(text.replace(",", "").strip())
                    except ValueError:
                        raise ValueError(f"개수는 정수로 입력하세요: {text.strip() or '빈 값'}") from None
            except ValueError as error:
                raise ValueError(f"구간 {row + 1}: {error}") from error
            bands.append(Band(start, stop, mode, value))
        return bands

    # Channels --------------------------------------------------------------------------

    def selected_sources(self):
        rows = sorted(self.proxy.mapToSource(index).row() for index in self.table.selectionModel().selectedRows())
        return [self.model.ports[row].source_index for row in rows]

    def select_sources(self, sources):
        sources = set(sources)
        batch = QItemSelection()
        for row, port in enumerate(self.model.ports):
            if port.source_index in sources:
                index = self.proxy.mapFromSource(self.model.index(row, 0))
                if index.isValid():
                    batch.select(index, index)
        self.table.selectionModel().select(batch, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def check(self, sources, state):
        if sources is None:
            sources = [port.source_index for port in self.model.ports]
        self.model.set_checked(sources, state)

    def check_only_selected(self):
        selected = self.selected_sources()
        if selected:
            self.model.set_checked([port.source_index for port in self.model.ports], False)
            self.model.set_checked(selected, True)

    def filter_changed(self, text):
        self.proxy.text = text.strip().casefold()
        self.proxy.invalidate()
        self.schedule()

    def visible_sources(self):
        return [self.model.ports[self.proxy.mapToSource(self.proxy.index(row, 0)).row()].source_index
                for row in range(self.proxy.rowCount())]

    def set_ports(self, ports):
        """Follow renames and reordering from the port tab while keeping checks and selection."""
        selected = self.selected_sources()
        self.model.set_ports(ports)
        self.select_sources(selected)
        self.schedule()

    def ri_ports(self):
        count = len(self.model.ports)
        return np.array([index in self.model.checked for index in range(1, count + 1)], dtype=bool)

    # Document and data -------------------------------------------------------------------

    def set_document(self, document):
        self.cancel_jobs()
        self.document = document
        self.network = None
        self.plan = None
        self.load_tried = False
        self.generation += 1
        self.cache.clear()
        self.wanted = self.shown = None
        self.search.clear()
        ports = document.ports if document is not None else []
        self.model.checked = {port.source_index for port in ports}
        self.model.set_ports(ports)
        self.plot.message = ""
        self.plot.set_data([], [])
        self.show_prompt()

    def show_prompt(self, loading=False):
        self.stack.setCurrentIndex(0)
        self.load_button.setVisible(self.document is not None and not loading)
        if self.document is None:
            self.prompt.setText("파일을 열면 주파수 보간을 사용할 수 있습니다.")
        elif loading:
            self.prompt.setText("수치 데이터를 불러오는 중…")
        else:
            try:
                size = format_bytes(self.document.source.stat().st_size)
            except OSError:
                size = "알 수 없음"
            self.prompt.setText("보간 미리보기에는 파일의 수치 데이터 전체가 필요합니다.\n"
                                f"파일 크기 {size} · 큰 파일은 시간이 걸릴 수 있습니다.")

    def activate(self):
        """Load small files as soon as the tab is shown; large ones wait for the button."""
        if self.document is None or self.network is not None or self.load_tried or self.host.job is not None:
            return
        try:
            small = self.document.source.stat().st_size <= AUTO_LOAD_BYTES
        except OSError:
            small = False
        if small:
            self.load()

    def load(self):
        if self.document is None or self.host.job is not None:
            return
        self.load_tried = True
        self.host.read_network()
        if self.host.job is not None:
            self.show_prompt(loading=True)

    def load_finished(self):
        """Restore the prompt when loading failed or was cancelled."""
        if self.network is None:
            self.show_prompt()

    def set_network(self, network):
        self.network = network
        self.generation += 1
        self.cache.clear()
        self.stack.setCurrentIndex(1)
        self.refresh()

    def can_export(self):
        return self.network is not None and self.plan is not None

    def export_request(self):
        return self.network.frequencies, self.plan.frequencies, self.ri_ports()

    # Preview -----------------------------------------------------------------------------

    def schedule(self, *args):
        self.timer.start()

    def quantity_changed(self):
        # Only a magnitude is always positive, so the other values start on a linear axis.
        self.y_log.setChecked(self.quantity.currentData() == "magnitude")
        self.schedule()

    def scales_changed(self):
        self.plot.set_scales(self.x_log.isChecked(), self.y_log.isChecked())
        self.schedule()

    def plot_reset(self):
        self.plot.reset_view()
        self.plot.setFocus()

    def update_plan(self):
        self.plan = None
        results = [""] * self.bands.rowCount()
        if self.network is None:
            message, error = "", False
        else:
            source = self.network.frequencies
            try:
                bands = self.read_bands()
                self.plan = build_plan(bands, source, self.keep_source.isChecked())
            except ValueError as failure:
                message, error = str(failure), True
            else:
                error = False
                results = ["원본 범위 밖" if item is None else
                           f"{format_frequency(item[0])} ~ {format_frequency(item[1])} · {item[2]:,}점"
                           for item in self.plan.ranges]
                targets = self.plan.frequencies
                size = estimate_size(len(self.model.ports), len(targets))
                message = (f"원본 {len(source):,}점 ({format_frequency(source[0])} ~ {format_frequency(source[-1])})"
                           f"  →  출력 {len(targets):,}점 ({format_frequency(targets[0])} ~ "
                           f"{format_frequency(targets[-1])})\n예상 파일 크기 약 {format_bytes(size)}")
        self.bands.blockSignals(True)
        for row, text in enumerate(results):
            self.bands.item(row, 4).setText(text)
            self.bands.item(row, 4).setToolTip(text)
        self.bands.blockSignals(False)
        self.status.setText(message)
        self.status.setStyleSheet("color: #ffb86b;" if error else "")
        self.host.update_save_button()

    def plot_entries(self):
        position = {port.source_index: row for row, port in enumerate(self.model.ports)}
        selected = self.selected_sources()
        # Without a selection the plot follows the search filter.
        shown = selected or self.visible_sources()
        entries = [(source - 1, source - 1) for source in shown]
        if self.transfer.isChecked() and 2 <= len(selected) <= TRANSFER_LIMIT:
            entries += [(first - 1, second - 1) for first in selected for second in selected if first != second]
        entries.sort(key=lambda entry: (entry[0] != entry[1], position[entry[0] + 1], position[entry[1] + 1]))
        return entries[:TRACE_LIMIT], len(entries)

    def refresh(self):
        self.timer.stop()
        self.update_plan()
        checked = len(self.model.checked)
        total = len(self.model.ports)
        selected = len(self.selected_sources())
        self.channel_label.setText(
            f"RI 보간 {checked} / {total}개  ·  그래프 "
            + (f"{selected}개 선택" if selected else "전체 표시")
            + "\n체크: RI 보간 적용 · 행 선택(Ctrl/Shift): 그래프에 표시할 채널")
        network = self.network
        if network is None:
            return
        if network.matrices is None:
            self.cancel_jobs()
            self.plot.message = "메모리가 부족해 그래프를 생략했습니다.\n구간 설정과 저장은 그대로 사용할 수 있습니다."
            self.plot.set_data([], self.regions())
            self.update_plot_label()
            return
        self.entries, self.entry_total = self.plot_entries()
        targets = self.plan.frequencies if self.plan is not None else np.empty(0)
        ri_ports = self.ri_ports()
        parameter = self.parameter.currentData()
        digest = hashlib.blake2b(targets.tobytes() + ri_ports.tobytes(), digest_size=16).digest()
        key = (self.generation, parameter, tuple(self.entries), digest)
        if not self.entries:
            self.cancel_jobs()
            self.wanted = None
            self.plot.message = "검색 결과에 해당하는 채널이 없습니다." if total else ""
            self.plot.set_data([], self.regions())
            self.update_plot_label()
        elif key in self.cache:
            self.cancel_jobs()
            self.wanted = key
            self.show_traces(key)
        elif key != self.wanted or not self.jobs:
            self.cancel_jobs()
            self.wanted = key
            job = TraceJob(key, (network, targets, self.entries, parameter, ri_ports), self)
            job.done.connect(self.job_done)
            job.failed.connect(self.job_failed)
            job.progress.connect(self.job_progress)
            job.finished.connect(lambda job=job: self.job_finished(job))
            self.jobs.append(job)
            self.plot.set_busy("계산 중…")
            job.start()

    def cancel_jobs(self):
        for job in self.jobs:
            job.stop.set()
        self.plot.set_busy("")

    def job_progress(self, key, percent):
        if key == self.wanted:
            self.plot.set_busy(f"{key[1]}-parameter 계산 중… {percent}%")

    def job_done(self, key, values):
        if key[0] != self.generation:
            return
        if len(self.cache) >= 6:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = values
        if key == self.wanted:
            self.plot.set_busy("")
            self.show_traces(key)

    def job_failed(self, key, message):
        if key == self.wanted:
            self.plot.set_busy("")
            self.plot.message = f"그래프 계산 실패: {message}"
            self.plot.set_data([], self.regions())

    def job_finished(self, job):
        self.jobs.remove(job)
        job.deleteLater()

    def shutdown(self):
        """Stop background work before the window closes."""
        self.timer.stop()
        self.cancel_jobs()
        for job in list(self.jobs):
            job.wait()

    def wait(self):
        """Finish pending preview work; used by tests."""
        from PySide6.QtCore import QCoreApplication
        if self.timer.isActive():
            self.refresh()
        while self.jobs:
            self.jobs[0].wait()
            QCoreApplication.processEvents()
        QCoreApplication.processEvents()

    def regions(self):
        if self.plan is None:
            return []
        regions = []
        for row, item in enumerate(self.plan.ranges):
            if item is None or item[0] == item[1]:
                continue
            label = {LINEAR_COUNT: "Linear", DECADE_COUNT: "Decade", LINEAR_STEP: "Step"}[
                self.bands.cellWidget(row, 2).currentData()]
            regions.append(Region(item[0], item[1], f"{row + 1} · {label} {self.bands.item(row, 3).text()}"))
        return regions

    def values(self, data):
        quantity = self.quantity.currentData()
        if quantity == "magnitude":
            return np.abs(data)
        if quantity == "db":
            with np.errstate(divide="ignore"):
                return 20 * np.log10(np.abs(data))
        if quantity == "phase":
            return np.degrees(np.angle(data))
        return data.real if quantity == "real" else data.imag

    def show_traces(self, key):
        source_values, target_values = self.cache[key]
        parameter = key[1]
        network = self.network
        targets = self.plan.frequencies if self.plan is not None else np.empty(0)
        if len(targets) != len(target_values):
            return
        position = {port.source_index: row for row, port in enumerate(self.model.ports)}
        names = {port.source_index: port.name for port in self.model.ports}
        lines, markers = [], []
        show_result = self.show_result.isChecked() and len(targets) > 0
        for column, (row, other) in enumerate(self.entries):
            first, second = position[row + 1], position[other + 1]
            label = f"{parameter}({first + 1},{second + 1})  {names[row + 1]}"
            if row != other:
                label += f"  ←  {names[other + 1]}"
            color = COLORS[(first if row == other else first * 7 + second + 3) % len(COLORS)]
            if show_result:
                lines.append(Trace(label, targets, self.values(target_values[:, column]), color, note="보간"))
            if self.show_source.isChecked():
                original = self.values(source_values[:, column])
                if not show_result:
                    lines.append(Trace(label, network.frequencies, original, color, note="원본"))
                markers.append(Trace(label, network.frequencies, original, color, markers=True, note="원본"))
        quantity = self.quantity.currentData()
        symbol = f"|{parameter}|" if quantity in ("magnitude", "db") else parameter
        unit = "Ω" if parameter == "Z" else ""
        self.plot.y_title = {"magnitude": f"{symbol} ({unit})" if unit else symbol,
                             "db": f"{symbol} (dB{unit})", "phase": f"∠{parameter} (°)",
                             "real": f"Re {symbol}" + (f" ({unit})" if unit else ""),
                             "imaginary": f"Im {symbol}" + (f" ({unit})" if unit else "")}[quantity]
        self.plot.y_unit = {"magnitude": f" {unit}" if unit else "", "db": f" dB{unit}", "phase": "°"}.get(
            quantity, f" {unit}" if unit else "")
        self.plot.message = "" if lines or markers else "원본 지점 또는 보간 결과를 선택하세요."
        traces = lines + markers
        self.hidden = ((self.x_log.isChecked() and any(len(trace.x) and trace.x[0] <= 0 for trace in traces))
                       or (self.y_log.isChecked() and any((trace.y <= 0).any() for trace in traces)))
        view = (parameter, quantity)
        self.plot.set_data(traces, self.regions(), keep_view=self.plot.modified and view == self.shown)
        self.shown = view
        self.update_plot_label()

    def update_plot_label(self):
        notes = []
        if self.network is not None and self.network.matrices is not None:
            if self.entry_total > len(self.entries):
                notes.append(f"{self.entry_total:,}개 항 중 {len(self.entries):,}개만 표시")
            if self.transfer.isChecked() and not 2 <= len(self.selected_sources()) <= TRANSFER_LIMIT:
                notes.append(f"전달 항은 채널 2~{TRANSFER_LIMIT}개 선택 시 표시")
            if self.hidden:
                notes.append("로그 축에서는 0 이하 값을 생략")
        notes.append("휠: 확대·축소 (Ctrl: X축만, Shift: Y축만) · 드래그: 이동 · Ctrl+D: 보기 초기화")
        self.plot_label.setText("  ·  ".join(notes))
