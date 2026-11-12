"""Interpolation tab: channel selection, frequency bands, checks and the preview plot."""

import hashlib
import math
from html import escape
from threading import Event

import numpy as np
from PySide6.QtCore import (QAbstractTableModel, QCoreApplication, QItemSelection, QItemSelectionModel,
                           QModelIndex, QSortFilterProxyModel, Qt, QThread, QTimer, Signal)
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                              QLabel, QLineEdit, QMenu, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                              QTableView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .core import CancelledError
from .interpolation import (DECADE_COUNT, DEFAULT_BANDS, DENSITY_LIMIT, LINEAR_AXIS, LINEAR_COUNT, LINEAR_STEP,
                            LOG_AXIS, PHASE_LIMIT, Band, build_plan, check_plan, compute_traces, estimate_size,
                            format_frequency, parse_frequency, touched_intervals)
from .names import natural_key
from .plot import COLORS, PlotWidget, Region, Trace, engineering

AUTO_LOAD_BYTES = 256 * 1024 * 1024
# Matrix entries times added points above which the output check waits for its button.
CHECK_BUDGET = 60_000_000
TRANSFER_LIMIT = 8
TRACE_LIMIT = 600
_WARNING = "#ffb86b"
_MODES = ((LINEAR_COUNT, "Linear Count", "지점 수"), (DECADE_COUNT, "Decade Count", "decade당 지점 수"),
          (LINEAR_STEP, "Linear Step", "간격"))
_QUANTITIES = (("magnitude", "크기"), ("db", "dB"), ("phase", "위상 (°)"), ("real", "실수"), ("imaginary", "허수"))


def format_bytes(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def magnitude_loss(degrees):
    """Share of the magnitude that real/imaginary interpolation loses halfway through an angle step."""
    return 100 * (1 - math.cos(math.radians(degrees) / 2))


class ChannelModel(QAbstractTableModel):
    """Ports in output order with a check mark for real/imaginary interpolation."""

    checks_changed = Signal()
    headers = ["RI 보간", "포트", "채널명", "위상 변화"]
    tips = ["체크한 채널이 포함된 항은 실수·허수(RI) 보간, 나머지 채널끼리의 항은 크기·위상(MA) 보간",
            "저장될 포트 번호", "저장될 채널명",
            "지점을 추가하는 구간에서 인접한 원본 지점 사이의 최대 위상 변화 (크기 0.01 이상인 항).\n"
            f"{PHASE_LIMIT:g}°를 넘으면 RI 보간이 지점 사이의 크기를 줄입니다."]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ports = []
        self.checked = set()
        # Largest angle step per source port for the current plan; None before the data is loaded.
        self.steps = None

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.ports)

    def columnCount(self, parent=QModelIndex()):
        return 4

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.DisplayRole:
            return self.headers[section]
        if role == Qt.ToolTipRole:
            return self.tips[section]

    def step(self, port):
        return None if self.steps is None else float(self.steps[port.source_index - 1])

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        port = self.ports[index.row()]
        column = index.column()
        if column == 0:
            if role == Qt.CheckStateRole:
                return Qt.Checked if port.source_index in self.checked else Qt.Unchecked
            return None
        if column == 3:
            step = self.step(port)
            if step is None:
                return None
            if role == Qt.DisplayRole:
                return f"{step:.1f}°"
            if role == Qt.TextAlignmentRole:
                return Qt.AlignCenter
            if step > PHASE_LIMIT:
                if role == Qt.ForegroundRole and port.source_index in self.checked:
                    return QColor(_WARNING)
                if role == Qt.ToolTipRole:
                    return (f"위상 변화 {step:.1f}°: RI 보간 시 지점 사이 크기가 최대 {magnitude_loss(step):.1f}% "
                            "줄어듭니다. 이런 채널은 체크를 해제해 크기·위상 보간을 쓰세요.")
            return None
        if role == Qt.DisplayRole:
            return index.row() + 1 if column == 1 else port.name
        if role == Qt.ToolTipRole:
            return f"원본 포트 {port.source_index}: {port.original_name}"
        if role == Qt.TextAlignmentRole and column == 1:
            return Qt.AlignCenter

    def flags(self, index):
        flags = super().flags(index)
        return flags | Qt.ItemIsUserCheckable if index.column() == 0 else flags

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid() and index.column() == 0 and role == Qt.CheckStateRole:
            self.set_checked([self.ports[index.row()].source_index], Qt.CheckState(value) == Qt.Checked)
            return True
        return False

    def changed(self, column):
        if self.ports:
            self.dataChanged.emit(self.index(0, column), self.index(len(self.ports) - 1, column))

    def set_ports(self, ports):
        self.beginResetModel()
        self.ports = list(ports)
        self.endResetModel()

    def set_steps(self, steps):
        self.steps = steps
        self.changed(3)

    def set_checked(self, sources, state):
        before = set(self.checked)
        self.checked = self.checked | set(sources) if state else self.checked - set(sources)
        if self.checked != before:
            self.changed(0)
            self.changed(3)
            self.checks_changed.emit()


class ChannelProxy(QSortFilterProxyModel):
    text = ""

    def lessThan(self, left, right):
        model = self.sourceModel()
        keys = []
        for index in (left, right):
            port = model.ports[index.row()]
            if left.column() == 0:
                key = port.source_index not in model.checked
            elif left.column() == 2:
                key = natural_key(port.name)
            elif left.column() == 3:
                key = model.step(port) or 0.0
            else:
                key = 0
            keys.append((key, index.row()))
        return keys[0] < keys[1]

    def filterAcceptsRow(self, row, parent):
        return (not self.text or self.text in self.sourceModel().ports[row].name.casefold()
                or self.text == str(row + 1))


class ChannelTable(QTableView):
    """Clicks in the check column toggle one channel; Space toggles every selected channel."""

    # True from a press in the check column until its release.
    checking = False

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        self.checking = index.isValid() and index.column() == 0 and event.button() == Qt.LeftButton
        if self.checking:
            # Checking a channel must not change which channels the plot shows.
            state = Qt.Unchecked if index.data(Qt.CheckStateRole) == Qt.Checked else Qt.Checked
            self.model().setData(index, state, Qt.CheckStateRole)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # The view never saw the press, so a move or release would act on an older one.
        if not self.checking:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.checking:
            self.checking = False
            return
        super().mouseReleaseEvent(event)

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


class ModeBox(QComboBox):
    """A band's mode; the wheel is left to the table so scrolling never changes a band."""

    def wheelEvent(self, event):
        event.ignore()


class Job(QThread):
    """Run one calculation off the window thread; `key` names the request it answers."""

    done = Signal(object, object)
    failed = Signal(object, str)
    progress = Signal(object, int)

    def __init__(self, key, action, parent=None):
        super().__init__(parent)
        self.key = key
        self.action = action
        self.stop = Event()

    def run(self):
        try:
            values = self.action(lambda percent: self.progress.emit(self.key, percent), self.stop.is_set)
        except CancelledError:
            return
        except Exception as error:
            self.failed.emit(self.key, str(error))
            return
        self.done.emit(self.key, values)


class InterpolationTab(QWidget):
    """Add interpolated points to selected channels and preview the result."""

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.document = None
        self.network = None
        self.plan = None
        self.plan_bands = []
        self.plan_error = ""
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
        self.checks = None
        self.check_key = None
        self.check_wanted = None
        # "", "manual" (waits for the button), "running" or "failed", with a note for the last two.
        self.check_state = ""
        self.check_note = ""
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

        # Channels fill the left edge; the bands and checks sit under the plot whose regions they define.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.build_channels())
        self.right = QSplitter(Qt.Vertical)
        self.right.addWidget(self.build_plot())
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.build_bands())
        bottom.addWidget(self.build_report())
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        bottom.setSizes([550, 330])
        self.right.addWidget(bottom)
        self.right.setStretchFactor(0, 1)
        self.right.setStretchFactor(1, 0)
        self.right.setSizes([490, 250])
        self.splitter.addWidget(self.right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([370, 880])
        # A small window scrolls instead of squeezing the plot and tables into each other.
        self.splitter.setMinimumSize(1040, 450)
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
        group.setMinimumWidth(280)
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
        menu.addSeparator()
        menu.addAction(f"위상 변화 {PHASE_LIMIT:g}° 초과 채널 해제", self.uncheck_steep)
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
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setStretchLastSection(False)
        for column, width in ((0, 76), (1, 56), (3, 84)):
            self.table.setColumnWidth(column, width)
        self.table.setAccessibleName("보간 및 그래프 채널 목록")
        self.table.selectionModel().selectionChanged.connect(self.schedule)
        layout.addWidget(self.table, 1)
        self.channel_label = QLabel()
        self.channel_label.setObjectName("hint")
        self.channel_label.setWordWrap(True)
        layout.addWidget(self.channel_label)
        return group

    def build_bands(self):
        group = QGroupBox("추가할 주파수 구간")
        layout = QVBoxLayout(group)
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
        for column, width in enumerate((70, 70, 128, 64)):
            self.bands.setColumnWidth(column, width)
        self.bands.horizontalHeaderItem(0).setToolTip("예: 0, 1k, 10MHz, 2G")
        self.bands.horizontalHeaderItem(1).setToolTip("예: 0, 1k, 10MHz, 2G")
        self.bands.horizontalHeaderItem(3).setToolTip("Linear Count: 구간 전체 지점 수\n"
                                                      "Decade Count: decade당 지점 수\nLinear Step: 주파수 간격")
        self.bands.horizontalHeaderItem(4).setToolTip("구간의 지점 수와 원본 주파수 범위 안으로 줄인 실제 구간.\n"
                                                      "괄호 안 +는 그중 원본에 없어 새로 추가되는 지점 수")
        self.bands.setAccessibleName("추가할 주파수 구간 설정")
        self.bands.setMinimumHeight(96)
        self.bands.itemChanged.connect(self.band_edited)
        layout.addWidget(self.bands, 1)
        row = QHBoxLayout()
        for text, callback in (("구간 추가", self.add_band), ("구간 삭제", self.remove_band),
                               ("기본값", self.reset_bands)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch()
        row.addWidget(QLabel("보간 축"))
        self.axis = QComboBox()
        self.axis.addItem("로그 주파수", LOG_AXIS)
        self.axis.addItem("선형 주파수", LINEAR_AXIS)
        self.axis.setAccessibleName("보간 축")
        self.axis.setToolTip("원본 지점 사이를 직선으로 잇는 주파수 축.\n"
                             "두 축의 결과 차이는 보통 1~2% 이내이며 어느 쪽이 더 정확한지는 구간에 따라 다릅니다.\n"
                             "0 Hz와 첫 양의 주파수 사이에는 지점을 추가하지 않습니다.")
        self.axis.currentIndexChanged.connect(self.schedule)
        row.addWidget(self.axis)
        layout.addLayout(row)
        self.set_bands(DEFAULT_BANDS)
        return group

    def build_report(self):
        group = QGroupBox("출력 · 점검")
        group.setMinimumWidth(260)
        layout = QVBoxLayout(group)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setTextFormat(Qt.RichText)
        self.report.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report.setAccessibleName("출력 지점과 점검 결과")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.report)
        # The group box supplies the background.
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        layout.addWidget(scroll, 1)
        self.check_button = QPushButton("출력 점검 실행")
        self.check_button.setToolTip("추가 지점의 수동성·대칭성과 출력 격자의 MA–RI 차이를 계산합니다.")
        self.check_button.clicked.connect(lambda: self.start_check(force=True))
        self.check_button.hide()
        layout.addWidget(self.check_button, 0, Qt.AlignLeft)
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
        self.compare = QCheckBox("MA 비교")
        self.compare.setToolTip("같은 지점을 모든 채널 크기·위상(MA) 보간으로 계산한 값을 점선으로 겹쳐 그립니다.\n"
                                "결합 툴의 기본 보간이 만드는 차이를 확인하는 용도입니다.")
        self.transfer = QCheckBox("전달 항")
        self.transfer.setToolTip(f"선택한 채널 사이의 전달 항도 표시합니다. 채널 2~{TRANSFER_LIMIT}개 선택 시 적용됩니다.")
        reset = QPushButton("보기 초기화")
        reset.setToolTip("기본 위치·크기로 되돌리기 (Ctrl+D)")
        reset.clicked.connect(self.plot_reset)
        for widget in (self.parameter, self.quantity, self.x_log, self.y_log, self.show_source,
                       self.show_result, self.compare, self.transfer):
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
        for check in (self.show_source, self.show_result, self.compare, self.transfer):
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
        combo = ModeBox()
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
        self.append_band(Band(start, stop, DECADE_COUNT, 100) if start > 0 else Band(start, stop, LINEAR_COUNT, 101))
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

    def uncheck_steep(self):
        steep = [port.source_index for port in self.model.ports if (self.model.step(port) or 0) > PHASE_LIMIT]
        self.model.set_checked(steep, False)

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
        self.plan_bands = []
        self.load_tried = False
        self.generation += 1
        self.cache.clear()
        self.wanted = self.shown = None
        self.checks = self.check_key = self.check_wanted = None
        self.search.clear()
        ports = document.ports if document is not None else []
        self.model.checked = {port.source_index for port in ports}
        self.model.steps = None
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
        self.checks = self.check_key = self.check_wanted = None
        self.stack.setCurrentIndex(1)
        self.refresh()

    def can_export(self):
        return self.network is not None and self.plan is not None

    def prepare_export(self):
        """Apply an edit that is still open or waiting for the timer, so the saved file matches the table."""
        current = self.bands.currentItem()
        if current is not None and current.column() != 4:
            # Moving to the read-only column commits an open cell editor.
            self.bands.setCurrentCell(current.row(), 4)
        self.refresh()

    def export_request(self):
        """Return the bands, plan, real/imaginary port selection and axis to export with."""
        return self.plan_bands, self.plan, self.ri_ports(), self.axis.currentData()

    # Plan and checks ---------------------------------------------------------------------

    def schedule(self, *args):
        self.timer.start()

    def update_plan(self):
        self.plan = None
        self.plan_bands = []
        self.plan_error = ""
        results = [""] * self.bands.rowCount()
        steps = None
        if self.network is not None:
            source = self.network.frequencies
            try:
                self.plan_bands = self.read_bands()
                self.plan = build_plan(self.plan_bands, source)
            except ValueError as failure:
                self.plan_error = str(failure)
            else:
                results = ["원본 범위 밖" if item is None else
                           f"{item[2]:,}점 (+{item[3]:,}) · {format_frequency(item[0])} ~ {format_frequency(item[1])}"
                           for item in self.plan.ranges]
                touched = touched_intervals(self.network, self.plan)
                steps = (self.network.phase_steps[touched].max(axis=0) if len(touched)
                         else np.zeros(len(self.model.ports)))
        self.model.set_steps(steps)
        self.bands.blockSignals(True)
        for row, text in enumerate(results):
            self.bands.item(row, 4).setText(text)
            self.bands.item(row, 4).setToolTip(text)
        self.bands.blockSignals(False)
        self.host.update_save_button()

    def start_check(self, force=False):
        """Measure the planned output in the background; large problems wait for the button."""
        network, plan = self.network, self.plan
        if network is None or plan is None or network.matrices is None:
            self.check_wanted = None
            self.check_state = ""
            return
        ri_ports = self.ri_ports()
        axis = self.axis.currentData()
        digest = hashlib.blake2b(plan.frequencies.tobytes() + ri_ports.tobytes(), digest_size=16).digest()
        key = ("check", self.generation, digest, axis)
        if key == self.check_key or (key == self.check_wanted and self.check_state == "running"):
            return
        self.cancel_jobs("check")
        self.checks = self.check_key = None
        self.check_wanted = key
        self.check_note = ""
        if not force and len(ri_ports) ** 2 * int(plan.added.sum()) > CHECK_BUDGET:
            self.check_state = "manual"
            return
        self.check_state = "running"
        self.start_job(key, lambda progress, cancel: check_plan(network, plan, ri_ports, axis, progress, cancel))
        self.update_report()

    def update_report(self):
        """Show the output size, the warnings and the checks."""
        self.check_button.setVisible(self.check_state == "manual")
        network, plan = self.network, self.plan
        if network is None:
            self.report.setText("")
            return
        if plan is None:
            self.report.setText(f'<span style="color:{_WARNING}">{escape(self.plan_error)}</span>')
            return
        source, targets = network.frequencies, plan.frequencies
        added = int(plan.added.sum())
        size = format_bytes(estimate_size(len(self.model.ports), len(targets)))
        lines = [f"원본 {len(source):,}점 + 추가 {added:,}점 → 출력 <b>{len(targets):,}점</b> · 약 {size}"]

        def note(text, warning=False):
            mark, color = ("⚠", f' style="color:{_WARNING}"') if warning else ("✔", "")
            lines.append(f"<span{color}>{mark} {text}</span>")

        def span(first, last):
            return f"{format_frequency(first, 4)} ~ {format_frequency(last, 4)}"

        for number, (band, item) in enumerate(zip(self.plan_bands, plan.ranges), 1):
            if item is not None and band.mode == DECADE_COUNT and band.value < DENSITY_LIMIT:
                note(f"구간 {number}: decade당 {int(band.value)}점. {DENSITY_LIMIT}점 미만이면 툴이 크기·위상으로 "
                     "다시 보간할 때 오차가 남음 (100점 이상 권장)", True)
        ri_ports = self.ri_ports()
        checks = self.checks
        if not added:
            note("추가되는 지점이 없음. 구간이 원본 지점과 모두 겹침", True)
        if ri_ports.any():
            steps = np.where(ri_ports, self.model.steps, 0.0)
            steep = int((steps > PHASE_LIMIT).sum())
            if steep:
                note(f"위상 변화: RI 채널 {steep}개가 {PHASE_LIMIT:g}° 초과 (최대 {steps.max():.1f}°). RI 보간 시 지점 사이 "
                     f"크기가 최대 {magnitude_loss(steps.max()):.1f}% 줄어듦. 해당 채널은 체크 해제 권장", True)
            else:
                note(f"위상 변화 최대 {steps.max():.1f}° (RI 채널, 지점 추가 구간)")
            gaps = network.differences[:, ri_ports].max(axis=1) if len(network.differences) else np.zeros(0)
            if len(gaps):
                worst = int(np.argmax(gaps))
                text = f"MA–RI 차이: 원본 격자 최대 {gaps[worst]:.2g} ({span(source[worst], source[worst + 1])})"
                if checks is not None:
                    text += f"<br>&nbsp;&nbsp;&nbsp;→ 출력 격자 최대 {checks.remaining:.2g}"
                    if checks.remaining_range is not None:
                        text += f" ({span(*checks.remaining_range)})"
                references = network.references[ri_ports]
                if references.min() == references.max():
                    # Reference resistance times the gap gives the order of the false series impedance.
                    text += f"<br>&nbsp;&nbsp;&nbsp;기준 임피던스 곱 {engineering(references[0] * gaps[worst], 3)}Ω"
                    if checks is not None:
                        text += f" → {engineering(references[0] * checks.remaining, 3)}Ω"
                lines.append(text)
        else:
            lines.append("RI 보간 채널 없음. 모든 항을 크기·위상으로 보간")
        if network.matrices is None:
            lines.append("메모리 부족으로 추가 지점의 수동성·대칭성 점검 생략")
        elif checks is not None:
            note(f"수동성: 최대 특이값 원본 {checks.source_gain:.9f} → 추가 {checks.added_gain:.9f}",
                 checks.added_gain > checks.source_gain + 1e-12)
            note(f"대칭성: 비대칭 최대 원본 {network.asymmetry:.1e} → 추가 {checks.asymmetry:.1e}",
                 checks.asymmetry > network.asymmetry + 1e-12)
        elif self.check_state == "manual":
            lines.append("포트 수가 많아 추가 지점의 수동성·대칭성 점검은 버튼으로 실행")
        elif self.check_state == "running":
            lines.append(f"추가 지점 점검 중… {self.check_note}")
        elif self.check_state == "failed":
            note(f"추가 지점 점검 실패: {escape(self.check_note)}", True)
        self.report.setText("<br>".join(lines))

    # Preview -----------------------------------------------------------------------------

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
            self.update_report()
            return
        self.start_check()
        self.update_report()
        if network.matrices is None:
            self.cancel_jobs("trace")
            self.plot.message = "메모리가 부족해 그래프를 생략했습니다.\n구간 설정과 저장은 그대로 사용할 수 있습니다."
            self.plot.set_data([], self.regions())
            self.update_plot_label()
            return
        self.entries, self.entry_total = self.plot_entries()
        targets = self.plan.frequencies if self.plan is not None else np.empty(0)
        ri_ports = self.ri_ports()
        parameter = self.parameter.currentData()
        axis = self.axis.currentData()
        compare = self.compare.isChecked()
        digest = hashlib.blake2b(targets.tobytes() + ri_ports.tobytes(), digest_size=16).digest()
        key = ("trace", self.generation, parameter, tuple(self.entries), digest, axis, compare)
        if not self.entries:
            self.cancel_jobs("trace")
            self.wanted = None
            self.plot.message = "검색 결과에 해당하는 채널이 없습니다." if total else ""
            self.plot.set_data([], self.regions())
            self.update_plot_label()
        elif key in self.cache:
            self.cancel_jobs("trace")
            self.wanted = key
            self.show_traces(key)
        elif key != self.wanted or not any(job.key == key and not job.stop.is_set() for job in self.jobs):
            self.cancel_jobs("trace")
            self.wanted = key
            entries = self.entries

            def traces(progress, cancel):
                values = compute_traces(network, targets, entries, parameter, ri_ports, axis, progress, cancel)
                if not compare:
                    return values
                # The comparison interpolates every entry on magnitude and angle, as a tool does by default.
                polar = compute_traces(network, targets, entries, parameter, np.zeros_like(ri_ports), axis,
                                       None, cancel)
                return (*values, polar[1])

            self.plot.set_busy("계산 중…")
            self.start_job(key, traces)

    def start_job(self, key, action):
        job = Job(key, action, self)
        job.done.connect(self.job_done)
        job.failed.connect(self.job_failed)
        job.progress.connect(self.job_progress)
        job.finished.connect(lambda job=job: self.job_finished(job))
        self.jobs.append(job)
        job.start()

    def cancel_jobs(self, kind=None):
        for job in self.jobs:
            if kind is None or job.key[0] == kind:
                job.stop.set()
        if kind != "check":
            self.plot.set_busy("")

    def job_progress(self, key, percent):
        if key == self.wanted:
            self.plot.set_busy(f"{key[2]}-parameter 계산 중… {percent}%")
        elif key == self.check_wanted:
            self.check_note = f"{percent}%"
            self.update_report()

    def job_done(self, key, values):
        if key[1] != self.generation:
            return
        if key[0] == "check":
            if key == self.check_wanted:
                self.checks, self.check_key, self.check_state = values, key, ""
                self.update_report()
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
        elif key == self.check_wanted:
            self.check_state, self.check_note = "failed", message
            self.update_report()

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
        """Finish pending preview and check work; used by tests."""
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
        # The plan's own bands, not the table: the table may already hold a later edit.
        for number, (band, item) in enumerate(zip(self.plan_bands, self.plan.ranges), 1):
            if item is None or item[0] == item[1]:
                continue
            label = {LINEAR_COUNT: f"Linear {int(band.value)}", DECADE_COUNT: f"Decade {int(band.value)}",
                     LINEAR_STEP: f"Step {format_frequency(band.value)}"}[band.mode]
            regions.append(Region(item[0], item[1], f"{number} · {label}"))
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
        source_values, target_values, *polar_values = self.cache[key]
        parameter = key[2]
        network = self.network
        targets = self.plan.frequencies if self.plan is not None else np.empty(0)
        if len(targets) != len(target_values):
            return
        position = {port.source_index: row for row, port in enumerate(self.model.ports)}
        names = {port.source_index: port.name for port in self.model.ports}
        lines, dashes, markers = [], [], []
        show_result = self.show_result.isChecked() and len(targets) > 0
        for column, (row, other) in enumerate(self.entries):
            first, second = position[row + 1], position[other + 1]
            label = f"{parameter}({first + 1},{second + 1})  {names[row + 1]}"
            if row != other:
                label += f"  ←  {names[other + 1]}"
            color = COLORS[(first if row == other else first * 7 + second + 3) % len(COLORS)]
            if show_result:
                lines.append(Trace(label, targets, self.values(target_values[:, column]), color, note="보간"))
                if polar_values:
                    dashes.append(Trace(f"{label}  · MA", targets, self.values(polar_values[0][:, column]), color,
                                        dashed=True, note="전체 MA 보간"))
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
        traces = dashes + lines + markers
        self.plot.message = "" if traces else "원본 지점 또는 보간 결과를 선택하세요."
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
