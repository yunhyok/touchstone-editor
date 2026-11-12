"""Add interpolated frequency points to S-parameters.

A tool that combines networks interpolates between the file's frequency points. Doing that
on magnitude and angle bends every matrix entry along its own path and breaks the linear
relations between entries, so a small series impedance carried by those relations turns
into a false resonance. Interpolating the real and imaginary parts keeps the relations;
INTDATTYP=RI selects it in an HSPICE-compatible S model. This module writes that
interpolation into the file: every source point stays and bands of new points go between.
"""

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

import numpy as np

from . import __version__
from .core import (_check_cancel, _check_source, _export_lines, _lines, _points, _write_point,
                   export_touchstone, option_settings)

LINEAR_COUNT, DECADE_COUNT, LINEAR_STEP = "linear_count", "decade_count", "linear_step"
LOG_AXIS, LINEAR_AXIS = "log", "lin"
MAX_POINTS = 1_000_000
# Below this many points per decade a later magnitude/angle interpolation still leaves visible error.
DENSITY_LIMIT = 40
# Above this angle step between source points, real/imaginary interpolation loses magnitude.
PHASE_LIMIT = 20.0
_CHANGED = "The source file changed. Open it again before exporting."
# A band point this close to a source frequency, relative, is that source frequency.
_TOLERANCE = 1e-9
# Smaller magnitudes carry no usable angle.
_MEANINGFUL = 0.01
_FREQUENCY = re.compile(r"\s*((?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(hz|khz|mhz|ghz|k|meg|m|g)?\s*",
                        re.IGNORECASE)
_SCALES = {"": 1, "hz": 1, "k": 10**3, "khz": 10**3, "m": 10**6, "meg": 10**6, "mhz": 10**6,
           "g": 10**9, "ghz": 10**9}


@dataclass(frozen=True)
class Band:
    start: float
    stop: float
    mode: str
    value: float


DEFAULT_BANDS = (Band(0.0, 1e3, LINEAR_COUNT, 101),
                 Band(1e3, 1e7, DECADE_COUNT, 100),
                 Band(1e7, 2e9, LINEAR_STEP, 1e6))


@dataclass(frozen=True)
class Plan:
    # Every source frequency plus the added ones, and which of them are added.
    frequencies: np.ndarray
    added: np.ndarray
    # One (first, last, points, new points) per band, or None when the band misses the source range.
    ranges: tuple


@dataclass
class Network:
    frequencies: np.ndarray
    # Complex (ports, ports) arrays in source port order; None when they did not fit in memory.
    matrices: list | None
    references: np.ndarray
    data_format: str
    # Per source interval and port: the largest angle step in degrees, and the largest gap between
    # magnitude/angle and real/imaginary interpolation halfway through the interval.
    phase_steps: np.ndarray
    differences: np.ndarray
    # Largest |S - S transposed| over all source points.
    asymmetry: float
    # Largest singular value over all source points, once a check has measured it.
    gain: float | None = None


@dataclass(frozen=True)
class Checks:
    source_gain: float
    added_gain: float
    asymmetry: float
    # Largest gap a second magnitude/angle interpolation of the output would open, and where.
    remaining: float
    remaining_range: tuple | None


def parse_frequency(text):
    """Read a frequency such as 0, 1k, 10MHz or 2G as hertz."""
    match = _FREQUENCY.fullmatch(text)
    if match is None:
        raise ValueError(f"주파수를 해석할 수 없습니다: {text.strip() or '빈 값'}  (예: 0, 1k, 10MHz, 2G)")
    try:
        value = float(Decimal(match[1]) * _SCALES[(match[2] or "").lower()])
    except (ArithmeticError, ValueError):
        value = math.inf
    if not math.isfinite(value):
        raise ValueError("주파수가 너무 큽니다.")
    return value


def format_frequency(hertz, digits=12):
    """Write a frequency with its unit; the default digits read back to the same value."""
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(hertz) >= scale:
            return f"{hertz / scale:.{digits}g} {unit}"
    return f"{hertz:.{digits}g} Hz"


def _band_grid(band, low, high):
    if not (math.isfinite(band.start) and math.isfinite(band.stop)) or band.start < 0 or band.stop <= band.start:
        raise ValueError("끝 주파수는 시작 주파수보다 커야 합니다.")
    start, stop = max(band.start, low), min(band.stop, high)
    if start > stop:
        return np.empty(0)
    if not math.isfinite(band.value):
        raise ValueError("값이 올바르지 않습니다.")
    if band.mode == LINEAR_COUNT:
        if band.value != int(band.value) or band.value < 2:
            raise ValueError("Linear Count는 2 이상의 정수여야 합니다.")
        first, last = 0.0, band.value - 1
    elif band.mode == DECADE_COUNT:
        if band.value != int(band.value) or band.value < 1:
            raise ValueError("Decade Count는 1 이상의 정수여야 합니다.")
        if band.start <= 0:
            raise ValueError("Decade Count 구간의 시작 주파수는 0보다 커야 합니다.")
        # Anchor the grid at the nominal start so clipping keeps the same frequencies.
        first = band.value * math.log10(start / band.start)
        last = band.value * math.log10(stop / band.start)
    elif band.mode == LINEAR_STEP:
        if band.value <= 0:
            raise ValueError("Linear Step 간격은 0보다 커야 합니다.")
        first = (start - band.start) / band.value
        last = (stop - band.start) / band.value
    else:
        raise ValueError(f"Unknown band mode: {band.mode}")
    if not last - first < MAX_POINTS:
        raise ValueError(f"한 구간의 지점 수가 {MAX_POINTS:,}개를 넘습니다. 간격이나 개수를 조정하세요.")
    first, last = math.ceil(first - 1e-9), math.floor(last + 1e-9)
    steps = np.arange(first, last + 1)
    if band.mode == LINEAR_COUNT:
        grid = np.linspace(start, stop, int(band.value)) if start < stop else np.empty(0)
    elif band.mode == DECADE_COUNT:
        grid = band.start * 10.0 ** (steps / band.value)
    else:
        grid = band.start + steps * band.value
    return np.unique(np.concatenate(([start], np.clip(grid, start, stop), [stop])))


def _snap(grid, source):
    """Move grid points that match a source frequency onto it, so the source value is used."""
    if not len(grid):
        return grid
    if len(source) > 1:
        nearest = np.clip(np.searchsorted(source, grid), 1, len(source) - 1)
        below, above = source[nearest - 1], source[nearest]
        knots = np.where(grid - below <= above - grid, below, above)
    else:
        knots = np.full(len(grid), source[0])
    return np.unique(np.where(np.abs(grid - knots) <= _TOLERANCE * np.abs(knots), knots, grid))


def build_plan(bands, source):
    """Keep every source frequency and add the band points that fall inside the source range."""
    source = np.asarray(source, dtype=np.float64)
    if not len(source):
        raise ValueError("원본 주파수 지점이 없습니다.")
    if not bands:
        raise ValueError("주파수 구간을 하나 이상 추가하세요.")
    positive = source[source > 0]
    additions, ranges = [], []
    for number, band in enumerate(bands, 1):
        try:
            grid = _snap(_band_grid(band, source[0], source[-1]), source)
        except ValueError as error:
            raise ValueError(f"구간 {number}: {error}") from error
        # 0 Hz has no place on a log axis, so nothing is added below the first positive source point.
        grid = grid[(grid == 0) | (grid >= positive[0])] if len(positive) else grid[grid == 0]
        new = grid[~np.isin(grid, source)]
        additions.append(new)
        ranges.append((grid[0], grid[-1], len(grid), len(new)) if len(grid) else None)
    if all(item is None for item in ranges):
        raise ValueError("모든 구간이 원본 주파수 범위 "
                         f"({format_frequency(source[0])} ~ {format_frequency(source[-1])}) 밖에 있습니다.")
    added = np.unique(np.concatenate(additions))
    if len(added):
        added = added[np.concatenate(([True], np.diff(added) > _TOLERANCE * added[1:]))]
    frequencies = np.union1d(source, added)
    if len(frequencies) > MAX_POINTS:
        raise ValueError(f"출력 지점 수가 {MAX_POINTS:,}개를 넘습니다. 간격이나 개수를 조정하세요.")
    return Plan(frequencies, ~np.isin(frequencies, source), tuple(ranges))


def describe(document, ports, bands, plan, ri_ports, axis, now=None):
    """Return the header comment lines that record how the output was made."""
    now = now or datetime.now().astimezone()
    name = "".join(character for character in document.source.name if character.isprintable())
    added = int(plan.added.sum())
    lines = [f"Resampled by Touchstone Editor {__version__} on {now.isoformat(sep=' ', timespec='seconds')}.",
             f"Source file: {name}",
             f"Frequency points: {len(plan.frequencies) - added} from the source, {added} added, "
             f"{len(plan.frequencies)} in total."]
    selected = [position for position, port in enumerate(ports, 1) if ri_ports[port.source_index - 1]]
    where = "a log-frequency axis" if axis == LOG_AXIS else "a linear frequency axis"
    if len(selected) == len(ports):
        lines.append(f"Interpolation: real/imaginary (INTDATTYP=RI equivalent) on {where} for all ports.")
    elif selected:
        lines.append(f"Interpolation on {where}: real/imaginary (INTDATTYP=RI equivalent) for ports "
                     f"{_port_ranges(selected)}; magnitude/angle for entries between the other ports.")
    else:
        lines.append(f"Interpolation: magnitude/angle on {where} for all ports.")
    for number, (band, item) in enumerate(zip(bands, plan.ranges), 1):
        if item is None:
            continue
        rule = {LINEAR_COUNT: f"{int(band.value)} points with linear spacing",
                DECADE_COUNT: f"{int(band.value)} points per decade",
                LINEAR_STEP: f"step {format_frequency(band.value)}"}[band.mode]
        lines.append(f"Band {number}: {format_frequency(item[0])} to {format_frequency(item[1])}, {rule}, "
                     f"{item[3]} new points.")
    lines.append("Added points are interpolated values, not solver results. "
                 "Source frequencies keep their original values.")
    return lines


def estimate_size(port_count, point_count):
    """Approximate output bytes; each number takes about twenty characters."""
    return point_count * (2 * port_count * port_count * 20 + 24)


def _frequency(token, unit):
    try:
        return float(Decimal(token.decode("ascii")) * int(unit))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Invalid frequency.") from error


def _matrix(point, count, data_format):
    values = np.array(point[1:], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Frequency {point[0].decode('ascii', 'replace')}: a value exceeds the numeric range.")
    pairs = values.reshape(count, count, 2)
    if count == 2:
        # Touchstone 1.x stores two-port values as 11, 21, 12, 22.
        pairs = pairs.transpose(1, 0, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    if data_format == "RI":
        return first + 1j * second
    magnitude = first if data_format == "MA" else 10.0 ** (first / 20.0)
    return magnitude * np.exp(1j * np.deg2rad(second))


class _Segment:
    """Interpolate between two frequency points; `ri` marks the real/imaginary entries."""

    def __init__(self, start, stop, ri=None):
        self.start, self.stop = start, stop
        self.polar = None if ri is None or ri.all() else ~ri
        if self.polar is not None:
            start, stop = start[self.polar], stop[self.polar]
            self.magnitudes = np.abs(start), np.abs(stop)
            # A zero has no angle, so it takes the angle of the other end and the value does not turn.
            self.angle = np.where(start == 0, np.angle(stop), np.angle(start))
            self.turn = np.where((start == 0) | (stop == 0), 0.0, np.angle(stop * np.conj(start)))

    def at(self, fraction):
        result = self.start * (1 - fraction) + self.stop * fraction
        if self.polar is not None:
            start, stop = self.magnitudes
            result[self.polar] = ((start * (1 - fraction) + stop * fraction)
                                  * np.exp(1j * (self.angle + self.turn * fraction)))
        return result


def _position(target, start, stop, axis):
    """Return how far a target lies between two source frequencies, from 0 to 1."""
    target, start, stop = (np.asarray(value, dtype=np.float64) for value in (target, start, stop))
    position = (target - start) / (stop - start)
    if axis == LOG_AXIS:
        # The interval that starts at 0 Hz has no logarithm and stays linear.
        with np.errstate(divide="ignore", invalid="ignore"):
            position = np.where(start > 0, np.log(target / start) / np.log(stop / start), position)
    return position


def _positions(source, targets, axis):
    """Return the source interval of every target and its position inside that interval."""
    if len(source) < 2:
        return np.zeros(len(targets), dtype=int), np.zeros(len(targets))
    index = np.clip(np.searchsorted(source, targets, side="right") - 1, 0, len(source) - 2)
    return index, _position(targets, source[index], source[index + 1], axis)


def _polar_gap(start, stop):
    """Distance between magnitude/angle and real/imaginary interpolation halfway between two values."""
    polar = _Segment(start, stop, np.zeros(start.shape, dtype=bool)).at(0.5)
    return np.abs(polar - (start + stop) / 2)


def _interval_metrics(start, stop):
    """Return the largest angle step in degrees and the largest interpolation gap of every port."""
    turn = np.degrees(np.abs(np.angle(stop * np.conj(start))))
    turn[(np.abs(start) < _MEANINGFUL) | (np.abs(stop) < _MEANINGFUL)] = 0
    return [np.maximum(values.max(axis=0), values.max(axis=1)) for values in (turn, _polar_gap(start, stop))]


def s_to_z(s, references, columns=None):
    """Z = sqrt(R) (I - S)^-1 (I + S) sqrt(R) for real per-port reference resistances."""
    identity = np.eye(len(s))
    root = np.sqrt(references)
    right = identity + s if columns is None else (identity + s)[:, columns]
    try:
        solved = np.linalg.solve(identity - s, right)
    except np.linalg.LinAlgError:
        return np.full(right.shape, complex("nan"))
    return root[:, None] * solved * (root if columns is None else root[columns])


def _ri_mask(ri_ports):
    # An entry uses real/imaginary interpolation when either of its ports is selected.
    return None if ri_ports.all() else ri_ports[:, None] | ri_ports[None, :]


def compute_traces(network, targets, entries, parameter, ri_ports, axis=LOG_AXIS, progress=None, cancel=None):
    """Evaluate matrix entries at the source and target frequencies.

    Entries are (row, column) pairs in source port order. Returns two complex arrays with
    one row per frequency and one column per entry.
    """
    matrices, source = network.matrices, network.frequencies
    rows = np.array([row for row, _ in entries], dtype=int)
    columns = np.array([column for _, column in entries], dtype=int)
    used, positions = np.unique(columns, return_inverse=True)
    total = len(source) + len(targets)
    done = 0

    def pick(matrix):
        nonlocal done
        done += 1
        if done % 16 == 0:
            _check_cancel(cancel)
            if progress is not None:
                progress(min(99, 100 * done // total))
        if parameter == "S":
            return matrix[rows, columns]
        return s_to_z(matrix, network.references, used)[rows, positions]

    original = np.array([pick(matrix) for matrix in matrices]).reshape(len(source), len(entries))
    if len(source) < 2:
        return original, np.repeat(original, len(targets), axis=0)
    index, fractions = _positions(source, targets, axis)
    mask = _ri_mask(ri_ports)
    if parameter == "S":
        mask = None if mask is None else mask[rows, columns]
    result = np.empty((len(targets), len(entries)), dtype=complex)
    segment, current = None, -1
    for position, (interval, fraction) in enumerate(zip(index.tolist(), fractions.tolist())):
        if interval != current:
            current = interval
            if parameter == "S":
                segment = _Segment(original[interval], original[interval + 1], mask)
            else:
                segment = _Segment(matrices[interval], matrices[interval + 1], mask)
        result[position] = pick(segment.at(fraction)) if parameter == "Z" else segment.at(fraction)
        if parameter == "S" and position % 1024 == 0:
            _check_cancel(cancel)
    return original, result


def _available_memory():
    try:
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = ([("length", ctypes.c_ulong), ("load", ctypes.c_ulong)]
                            + [(name, ctypes.c_ulonglong) for name in
                               ("total", "available", "total_page", "available_page",
                                "total_virtual", "available_virtual", "extended")])

            status = Status()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.available
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


class _Discard:
    def write(self, data):
        pass


def load_network(document, progress=None, cancel=None, memory_limit=None):
    """Read every frequency point; keep the matrices only while they fit in memory."""
    count = document.port_count
    settings = option_settings(document.option_line, count)
    if memory_limit is None:
        available = _available_memory()
        memory_limit = None if available is None else int(available * 0.6)
    frequencies, matrices, steps, differences = [], [], [], []
    names = [port.name for port in document.ports]
    previous = None
    asymmetry = 0.0
    with document.source.open("rb") as stream:
        _check_source(document.source, document._stamp, stream)
        raw_lines = _lines(stream, document, progress, cancel)
        lines = _export_lines(raw_lines, _Discard(), list(range(count)), names, False, False,
                              [b"\n"], [True], True)
        for point in _points(lines, count, collect=True):
            frequencies.append(_frequency(point[0], settings.unit))
            matrix = _matrix(point, count, settings.data_format)
            # The interval metrics need only two neighbouring points, so they survive a memory shortage.
            if previous is not None:
                step, difference = _interval_metrics(previous, matrix)
                steps.append(step)
                differences.append(difference)
            asymmetry = max(asymmetry, float(np.abs(matrix - matrix.T).max()))
            previous = matrix
            if matrices is not None:
                if memory_limit is not None and 16 * count * count * (len(matrices) + 1) > memory_limit:
                    matrices = None
                else:
                    matrices.append(matrix)
        _check_source(document.source, document._stamp, stream)
    if not frequencies:
        raise ValueError("The file contains no S-parameter data.")
    document.point_count = len(frequencies)
    if progress is not None:
        progress(100)
    shape = (len(frequencies) - 1, count)
    return Network(np.array(frequencies), matrices, np.array(settings.references), settings.data_format,
                   np.array(steps).reshape(shape), np.array(differences).reshape(shape), asymmetry)


def _gain(matrix):
    """Largest singular value; a passive network stays at or below one."""
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def touched_intervals(network, plan):
    """Return the source intervals that receive at least one added point."""
    added = plan.frequencies[plan.added]
    return np.unique(np.searchsorted(network.frequencies, added, side="right") - 1)


def check_plan(network, plan, ri_ports, axis=LOG_AXIS, progress=None, cancel=None):
    """Measure the output: gain and symmetry of the added points, and the gap a second
    magnitude/angle interpolation between neighbouring output points would still open."""
    matrices, source = network.matrices, network.frequencies
    if network.gain is None:
        gain = 0.0
        for matrix in matrices:
            _check_cancel(cancel)
            gain = max(gain, _gain(matrix))
        network.gain = gain
    mask = _ri_mask(ri_ports)
    # Only entries interpolated on real and imaginary parts are at risk from a second interpolation.
    entries = ... if mask is None else mask
    index, positions = _positions(source, plan.frequencies, axis)
    total = max(1, int(plan.added.sum()))
    done = 0
    added_gain = asymmetry = remaining = 0.0
    remaining_range = None
    previous = segment = None
    current = -1
    for frequency, new, interval, position in zip(plan.frequencies.tolist(), plan.added.tolist(),
                                                  index.tolist(), positions.tolist()):
        if new:
            if interval != current:
                current = interval
                segment = _Segment(matrices[interval], matrices[interval + 1], mask)
            matrix = segment.at(position)
            added_gain = max(added_gain, _gain(matrix))
            asymmetry = max(asymmetry, float(np.abs(matrix - matrix.T).max()))
            done += 1
            _check_cancel(cancel)
            if progress is not None:
                progress(min(99, 100 * done // total))
        else:
            matrix = matrices[int(np.searchsorted(source, frequency))]
        if previous is not None and ri_ports.any():
            start_frequency, start, start_new = previous
            if new or start_new:
                gap = float(_polar_gap(start[entries], matrix[entries]).max())
            else:
                # Between two untouched source points the load already measured the gap.
                gap = float(network.differences[int(np.searchsorted(source, start_frequency))][ri_ports].max())
            if gap > remaining:
                remaining, remaining_range = gap, (start_frequency, frequency)
        previous = frequency, matrix, new
    return Checks(network.gain, added_gain, asymmetry, remaining, remaining_range)


def _port_ranges(numbers):
    ranges = []
    for number in numbers:
        if ranges and ranges[-1][1] == number - 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


class Resampler:
    """Write the target frequency points while the exporter streams the source points."""

    def __init__(self, document, source, targets, ri_ports, axis=LOG_AXIS, notes=()):
        self.count = document.port_count
        self.settings = option_settings(document.option_line, self.count)
        self.source = np.asarray(source, dtype=np.float64)
        self.targets = np.asarray(targets, dtype=np.float64)
        self.mask = _ri_mask(np.asarray(ri_ports, dtype=bool))
        self.axis = axis
        self.notes = notes
        self.seen = 0
        self.cursor = 0
        self.previous = None
        self.segment = None
        self.templates = {}

    def _template(self, newline):
        template = self.templates.get(newline)
        if template is None:
            count = self.count
            number = "%.15g"
            if count <= 2:
                rows = [" ".join([number] * (2 * count * count))]
            else:
                widths = [min(8, 2 * count - start) for start in range(0, 2 * count, 8)]
                rows = [" ".join([number] * width) for _ in range(count) for width in widths]
            separator = newline.decode("ascii") + "\t"
            template = self.templates[newline] = "%s " + separator.join(rows) + newline.decode("ascii")
        return template

    def _write(self, output, frequency, matrix, order, newline):
        if order != list(range(self.count)):
            matrix = matrix[np.ix_(order, order)]
        pairs = np.empty((self.count, self.count, 2))
        data_format = self.settings.data_format
        if data_format == "RI":
            pairs[..., 0], pairs[..., 1] = matrix.real, matrix.imag
        else:
            magnitude = np.abs(matrix)
            # Decibels cannot express an exact zero.
            pairs[..., 0] = magnitude if data_format == "MA" else 20 * np.log10(np.maximum(magnitude, 1e-300))
            pairs[..., 1] = np.degrees(np.angle(matrix))
        if self.count == 2:
            pairs = pairs.transpose(1, 0, 2)
        text = self._template(newline) % ("%.15g" % (frequency / self.settings.unit), *pairs.ravel().tolist())
        output.write(text.encode("ascii"))

    def feed(self, output, point, order, newline, cancel=None):
        frequency = _frequency(point[0], self.settings.unit)
        if self.seen >= len(self.source) or frequency != self.source[self.seen]:
            raise ValueError(_CHANGED)
        if not self.seen:
            for note in self.notes:
                output.write(b"! " + note.encode("utf-8") + newline)
        self.seen += 1
        matrix = None
        while self.cursor < len(self.targets) and self.targets[self.cursor] <= frequency:
            target = self.targets[self.cursor]
            if target == frequency:
                _write_point(output, point, order, newline)
            elif self.previous is None:
                raise ValueError(_CHANGED)
            else:
                if self.segment is None:
                    start_frequency, start_point, start = self.previous
                    if start is None:
                        start = _matrix(start_point, self.count, self.settings.data_format)
                    matrix = _matrix(point, self.count, self.settings.data_format)
                    self.segment = start_frequency, _Segment(start, matrix, self.mask)
                start_frequency, segment = self.segment
                position = float(_position(target, start_frequency, frequency, self.axis))
                self._write(output, target, segment.at(position), order, newline)
            self.cursor += 1
            _check_cancel(cancel)
        self.previous = frequency, point, matrix
        self.segment = None

    def finish(self):
        if self.seen != len(self.source) or self.cursor != len(self.targets):
            raise ValueError(_CHANGED)


def export_interpolated(document, ports, target, source, targets, ri_ports, axis=LOG_AXIS, notes=(),
                        progress=None, cancel=None):
    """Write the resampled network with the current port names and order.

    `ri_ports` is a boolean per source port; True selects real/imaginary interpolation.
    `notes` are comment lines written ahead of the data, such as those from describe().
    """
    ri_ports = np.asarray(ri_ports, dtype=bool)
    if len(ri_ports) != document.port_count:
        raise ValueError("The interpolation selection does not match the port count.")
    export_touchstone(document, ports, target, progress, cancel,
                      Resampler(document, source, targets, ri_ports, axis, notes))
