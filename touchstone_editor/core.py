"""Stream Touchstone 1.x S-parameter data without rounding its values."""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .names import extract_channel_name


class CancelledError(Exception):
    """The caller cancelled a load or export."""


@dataclass
class Port:
    source_index: int
    name: str
    original_name: str


@dataclass
class TouchstoneDocument:
    source: Path
    port_count: int
    ports: list[Port]
    point_count: int | None
    option_line: str
    _stamp: tuple[int, ...] = field(repr=False)
    _header_size: int = field(repr=False)
    _header_digest: bytes = field(repr=False)
    _names: tuple[str, ...] = field(repr=False)


_SUFFIX = re.compile(r"\.s([1-9][0-9]*)p", re.IGNORECASE)
_PORT = re.compile(rb"![ \t]*port(?:\[([0-9]+)\]|[ \t]+([0-9]+))[ \t]*=[ \t]*(.*)", re.IGNORECASE)
_LEGACY_PORT = re.compile(rb"![ \t]*(?:port[0-9]+_|nets?list\b)", re.IGNORECASE)
_PORT_VALUES = re.compile(rb"![ \t]*(?:port[ \t]+impedance|gamma)\b", re.IGNORECASE)
_DIFF_CHANNEL = re.compile(rb"![ \t]*\.(end)?diffchannels\b", re.IGNORECASE)
_DIFF_ROW = re.compile(rb'![ \t]*(?:"[^"]*"[ \t]*)+')
_NUMBER = rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?"
_NUMERIC_LINE = re.compile(_NUMBER + rb"(?:[ \t]+" + _NUMBER + rb")*")
_OPTION_LINE = re.compile(rb"[ \t!-~]+")
_WHITESPACE = b" \t\r\n"
_BOM = b"\xef\xbb\xbf"
_LINE_LIMIT = 16 * 1024 * 1024


def _port_count(path):
    match = _SUFFIX.fullmatch(path.suffix)
    if match is None:
        raise ValueError("Expected a Touchstone .sNp extension, such as .s80p.")
    return int(match[1])


def _stamp(stat):
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    # On Windows, ctime is creation time and can differ between stat and fstat.
    return stamp if os.name == "nt" else (*stamp, stat.st_ctime_ns)


def _check_source(source, stamp, stream=None):
    # shortcut: body edits preserving file identity/size/mtime evade detection; use a snapshot if required.
    if _stamp(source.stat()) != stamp or (stream is not None and _stamp(os.fstat(stream.fileno())) != stamp):
        raise ValueError("The source file changed. Open it again before exporting.")


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise CancelledError()


def _header_lines(stream, cancel):
    # Use an unbuffered stream so opening stops at the first non-header byte.
    line = bytearray()
    byte = stream.read(1)
    if byte == _BOM[:1]:
        bom = byte + stream.read(2)
        if bom != _BOM:
            return
        line.extend(bom)
        byte = stream.read(1)
    prefix = True
    line_number = 1
    while byte:
        if prefix and byte not in b" \t":
            if byte == b"[":
                raise ValueError(f"Line {line_number}: Touchstone 2.x keywords are unsupported.")
            if byte not in b"!#\r\n":
                return
            prefix = False
        line.extend(byte)
        if byte in (b"\r", b"\n"):
            following = stream.read(1) if byte == b"\r" else b""
            if following == b"\n":
                line.extend(following)
                following = b""
            yield bytes(line)
            _check_cancel(cancel)
            line.clear()
            prefix = True
            line_number += 1
            byte = following or stream.read(1)
        else:
            if len(line) % 4096 == 0:
                _check_cancel(cancel)
                if len(line) >= _LINE_LIMIT:
                    raise ValueError("A Touchstone line exceeds the 16 MiB limit.")
            byte = stream.read(1)
    if line:
        yield bytes(line)


def _lines(stream, document, progress, cancel):
    size = document._stamp[2]
    header_size = document._header_size
    digest = hashlib.sha256()
    percent = 0
    if progress is not None:
        progress(percent)
    _check_cancel(cancel)
    tail = b""
    position = 0
    while block := stream.read(256 * 1024):
        if position < header_size:
            digest.update(block[:header_size - position])
            if position + len(block) >= header_size and digest.digest() != document._header_digest:
                raise ValueError("The source header changed. Open the file again before exporting.")
        position += len(block)
        current = min(99, 100 * position // max(1, size))
        if progress is not None and current > percent:
            progress(current)
            percent = current
        _check_cancel(cancel)
        chunk = tail + block
        lines = chunk.splitlines(keepends=True)
        # shortcut: cap one physical line at 16 MiB, revisit for larger vendor comments.
        if len(chunk) > _LINE_LIMIT and any(len(line) > _LINE_LIMIT for line in lines):
            raise ValueError("A Touchstone line exceeds the 16 MiB limit.")
        # Keep a final CR until the next block distinguishes CR from CRLF.
        tail = b"" if lines[-1].endswith(b"\n") else lines.pop()
        yield from lines
    if tail:
        yield tail
    if position < header_size:
        raise ValueError("The source header changed. Open the file again before exporting.")


def _parts(raw):
    code, marker, comment = raw.partition(b"!")
    return code.strip(_WHITESPACE), marker + comment.rstrip(b"\r\n")


def _diff_channel_state(code, comment, in_block, line_number):
    marker = _DIFF_CHANNEL.match(comment)
    if marker:
        if code or comment[marker.end():].strip(b" \t") or bool(marker[1]) != in_block:
            raise ValueError(f"Line {line_number}: malformed PowerSI .DiffChannels block.")
        return not in_block
    if in_block and (code or not _DIFF_ROW.fullmatch(comment)):
        raise ValueError(f"Line {line_number}: malformed PowerSI .DiffChannels block.")
    return in_block


def _text(raw):
    for encoding in ("utf-8", "cp949", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue


def _positive_number(token):
    if not _NUMERIC_LINE.fullmatch(token):
        return False
    try:
        return Decimal(token.decode("ascii")) > 0
    except InvalidOperation:
        return False


def _option_references(code, port_count, line_number):
    error = f"Line {line_number}: invalid option line. Only S-parameters are supported."
    if not _OPTION_LINE.fullmatch(code):
        raise ValueError(error)
    tokens = code[1:].split()
    categories = {
        b"HZ": "unit", b"KHZ": "unit", b"MHZ": "unit", b"GHZ": "unit",
        b"S": "parameter", b"RI": "format", b"MA": "format", b"DB": "format",
    }
    seen = set()
    offset = 0
    while offset < len(tokens):
        token = tokens[offset].upper()
        if token == b"R" and "reference" not in seen:
            offset += 1
            start = offset
            while offset < len(tokens) and _positive_number(tokens[offset]):
                offset += 1
            count = offset - start
            if count == 1:
                seen.add("reference")
                continue
            if count == port_count and offset == len(tokens):
                return start, tokens
        else:
            category = categories.get(token)
            if category is not None and category not in seen:
                seen.add(category)
                offset += 1
                continue
        raise ValueError(error)
    return None


def _points(lines, port_count, collect=False):
    row_size = 4 if port_count == 2 else port_count
    point_size = port_count * port_count
    filled = 0
    previous_frequency = None
    point = []
    for line_number, code in lines:
        values = code.split(maxsplit=9)
        first_line = filled == 0
        if port_count == 2 and len(values) == 5:
            raise ValueError(f"Line {line_number}: two-port noise data or an incomplete network point.")
        pairs, remainder = divmod(len(values) - first_line, 2)
        if (remainder or not 1 <= pairs <= 4 or filled % row_size + pairs > row_size
                or (port_count <= 4 and pairs != row_size)):
            raise ValueError(f"Line {line_number}: incorrect number of S-parameter values in the matrix row.")
        if not _NUMERIC_LINE.fullmatch(code):
            raise ValueError(f"Line {line_number}: invalid numeric data.")
        if first_line:
            try:
                frequency = Decimal(values[0].decode("ascii"))
            except InvalidOperation as error:
                raise ValueError(f"Line {line_number}: invalid frequency.") from error
            if frequency < 0 or (previous_frequency is not None and frequency <= previous_frequency):
                raise ValueError(f"Line {line_number}: frequencies must be nonnegative and strictly increasing.")
            previous_frequency = frequency
        if collect:
            point.extend(values)
        filled += pairs
        if filled == point_size:
            yield point
            point = []
            filled = 0
    if filled:
        raise ValueError("The file ends with an incomplete S-parameter matrix.")


def read_touchstone(path, progress=None, cancel=None):
    """Read leading metadata without consuming the numeric records."""
    path = Path(path)
    port_count = _port_count(path)
    source = path.resolve(strict=True)
    option_line = ""
    ports = {}
    digest = hashlib.sha256()
    header_size = 0
    in_diff_channels = False
    if progress is not None:
        progress(0)
    _check_cancel(cancel)
    with source.open("rb", buffering=0) as stream:
        stamp = _stamp(os.fstat(stream.fileno()))
        for line_number, raw in enumerate(_header_lines(stream, cancel), 1):
            header_size += len(raw)
            digest.update(raw)
            if line_number == 1:
                raw = raw.removeprefix(_BOM)
            code, comment = _parts(raw)
            in_diff_channels = _diff_channel_state(code, comment, in_diff_channels, line_number)
            match = _PORT.fullmatch(comment)
            if match:
                index = int(match[1] or match[2])
                if not 1 <= index <= port_count or index in ports or not match[3].strip():
                    raise ValueError(f"Line {line_number}: missing, duplicate or out-of-range port name.")
                original = _text(match[3].strip())
                name = extract_channel_name(original) if match[2] else original
                ports[index] = Port(index, name, original)
            if code.startswith(b"#"):
                if option_line:
                    raise ValueError(f"Line {line_number}: multiple option lines are unsupported.")
                _option_references(code, port_count, line_number)
                option_line = code.decode("ascii")
        _check_cancel(cancel)
        _check_source(source, stamp, stream)
    if in_diff_channels:
        raise ValueError("The PowerSI .DiffChannels block is missing its end marker.")
    if not option_line:
        raise ValueError("Expected an option line before the network data.")
    port_list = [ports.get(index, Port(index, f"Port {index}", f"Port {index}"))
                 for index in range(1, port_count + 1)]
    if progress is not None:
        progress(100)
    return TouchstoneDocument(source, port_count, port_list, None, option_line,
                              stamp, header_size, digest.digest(), tuple(port.name for port in port_list))


def _check_target(target, source):
    if target.resolve() == source:
        raise ValueError("Choose an output file different from the source file.")
    try:
        same = target.samefile(source)
    except FileNotFoundError:
        return
    if same:
        raise ValueError("Choose an output file different from the source file.")


def _write_point(output, point, order, newline):
    port_count = len(order)
    if port_count <= 2:
        # Touchstone 1.x stores two-port values as 11, 21, 12, 22.
        offsets = [1 + 2 * (column * port_count + row) for column in order for row in order]
        values = [value for offset in offsets for value in point[offset:offset + 2]]
        output.write(b" ".join([point[0], *values]) + newline)
        return
    rows = []
    for row in order:
        offset = 1 + 2 * row * port_count
        values = [value for column in order for value in point[offset + 2 * column:offset + 2 * column + 2]]
        rows.extend(b" ".join(values[i:i + 8]) for i in range(0, len(values), 8))
    output.write(point[0] + b" " + (newline + b"\t").join(rows) + newline)


def _export_lines(raw_lines, output, order, names, changed_order, changed_names, newline):
    names_written = False
    in_data = False
    option_written = False
    in_diff_channels = False
    for line_number, raw in enumerate(raw_lines, 1):
        if line_number == 1 and raw.startswith(_BOM):
            output.write(_BOM)
            raw = raw[len(_BOM):]
        line_ending = raw[len(raw.rstrip(b"\r\n")):]
        if line_ending:
            newline[0] = line_ending
        code, comment = _parts(raw)
        if comment or in_diff_channels:
            previous_state = in_diff_channels
            in_diff_channels = _diff_channel_state(code, comment, in_diff_channels, line_number)
            if previous_state or in_diff_channels:
                if not changed_order and not changed_names:
                    output.write(raw)
                continue
        if code.startswith(b"["):
            raise ValueError(f"Line {line_number}: Touchstone 2.x keywords are unsupported.")
        is_option = code.startswith(b"#")
        if is_option:
            if option_written:
                raise ValueError(f"Line {line_number}: multiple option lines are unsupported.")
            option_written = True
        if changed_order and comment and _PORT_VALUES.match(comment):
            raise ValueError("Port reordering with HFSS Gamma/Port Impedance extensions is unsupported. "
                             "Name-only export preserves these comments.")
        is_data = bool(code) and not is_option
        is_port = bool(comment and _PORT.fullmatch(comment))
        if is_port and (in_data or is_data):
            raise ValueError(f"Line {line_number}: numbered port names must precede the network data.")
        if not names_written and (is_port or is_data):
            for index, name in enumerate(names, 1):
                output.write(f"! Port[{index}] = {name}".encode("utf-8") + newline[0])
            names_written = True
        drop_comment = is_port or ((changed_order or changed_names) and comment and _LEGACY_PORT.match(comment))
        if drop_comment:
            if not code:
                continue
            raw = raw.split(b"!", 1)[0].rstrip(b" \t") + line_ending
        if is_data:
            in_data = True
            if not changed_order:
                output.write(raw)
            yield line_number, code
            continue
        if changed_order and in_data:
            continue
        if changed_order and is_option:
            reference = _option_references(code, len(order), line_number)
            if reference is not None:
                start, tokens = reference
                raw = b"# " + b" ".join(tokens[:start] + [tokens[start + index] for index in order])
                if comment and not drop_comment:
                    raw += b" " + comment
                raw += line_ending
        output.write(raw)
    if in_diff_channels:
        raise ValueError("The PowerSI .DiffChannels block is missing its end marker.")


def export_touchstone(document, ports, target, progress=None, cancel=None):
    """Write Ansys-style names and reorder both axes using the selected source indices."""
    target = Path(target).absolute()
    source = document.source
    port_count = document.port_count
    if _port_count(target) != port_count:
        raise ValueError(f"The output file must have the .s{port_count}p extension.")
    _check_target(target, source)
    indices = [port.source_index for port in ports]
    if (any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(1, port_count + 1))):
        raise ValueError("The output must include each source port exactly once.")
    names = []
    for port in ports:
        name = port.name.strip()
        if not name or not port.name.isprintable() or "!" in name:
            raise ValueError(f"Port {port.source_index}: enter a nonempty name without ! or control characters.")
        names.append(name)
    order = [index - 1 for index in indices]
    changed_order = order != list(range(port_count))
    changed_names = tuple(names) != document._names
    newline = [b"\n"]
    temporary_path = None
    point_count = 0
    try:
        with source.open("rb") as stream:
            _check_source(source, document._stamp, stream)
            _check_cancel(cancel)
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, suffix=".tmp", delete=False) as output:
                temporary_path = Path(output.name)
                raw_lines = _lines(stream, document, progress, cancel)
                lines = _export_lines(raw_lines, output, order, names, changed_order, changed_names, newline)
                for point in _points(lines, port_count, collect=changed_order):
                    if changed_order:
                        _write_point(output, point, order, newline[0])
                    point_count += 1
                if not point_count:
                    raise ValueError("The file contains no S-parameter data.")
                _check_source(source, document._stamp, stream)
                output.flush()
                os.fsync(output.fileno())
        _check_cancel(cancel)
        _check_source(source, document._stamp)
        _check_target(target, source)
        os.replace(temporary_path, target)
        document.point_count = point_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if progress is not None:
        progress(100)
