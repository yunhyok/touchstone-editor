import cmath
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from touchstone_editor.core import CancelledError, option_settings, read_touchstone
from touchstone_editor.interpolation import (DECADE_COUNT, DEFAULT_BANDS, LINEAR_AXIS, LINEAR_COUNT, LINEAR_STEP,
                                             LOG_AXIS, PHASE_LIMIT, Band, build_plan, check_plan, compute_traces,
                                             describe, export_interpolated, format_frequency, load_network,
                                             parse_frequency, s_to_z, touched_intervals)


def write(path, text):
    path.write_bytes(text.encode())
    return read_touchstone(path)


def touchstone(frequencies, matrices, reference=50.0):
    """Return RI-format Touchstone text with full precision; two-port rows follow the 11, 21, 12, 22 order."""
    count = len(matrices[0])
    lines = [f"# Hz S RI R {reference:g}"]
    for frequency, matrix in zip(frequencies, matrices):
        cells = matrix.T.ravel() if count == 2 else matrix.ravel()
        values = [f"{part!r}" for cell in cells for part in (float(cell.real), float(cell.imag))]
        rows = [" ".join(values)] if count <= 2 else [
            " ".join(values[start:min(start + 8, 2 * count * (row + 1))])
            for row in range(count) for start in range(2 * count * row, 2 * count * (row + 1), 8)]
        lines.append(f"{float(frequency)!r} " + "\n\t".join(rows))
    return "\n".join(lines) + "\n"


def numbers(path):
    """Return {frequency text: [values]} for the data rows of a written file."""
    points, current = {}, None
    count = int(path.suffix[2:-1])
    for line in path.read_bytes().splitlines():
        code = line.split(b"!", 1)[0].split()
        if not code or code[0].startswith(b"#"):
            continue
        if current is None or len(points[current]) == 2 * count * count:
            current = code[0].decode()
            points[current] = []
            code = code[1:]
        points[current].extend(float(value) for value in code)
    return points


def solver_grid():
    """A 626-point grid: 0 Hz, eight points per decade up to 1 MHz, then linear steps up to 2 GHz."""
    return np.concatenate(([0.0], 10 ** (np.arange(-8, 49) / 8), np.arange(1.5e6, 10.01e6, 0.5e6),
                           np.arange(11e6, 20.5e6, 1e6), np.arange(22e6, 100.5e6, 2e6),
                           np.arange(102e6, 400.5e6, 2e6), np.arange(404e6, 1000.5e6, 4e6),
                           np.arange(1005e6, 2000.5e6, 5e6)))


def lumped_block(frequency, reference=1.0, capacitance=1.5e-6, resistance=4e-3, inductance=80e-12):
    """S matrix of a shunt capacitor at port 1 followed by a small series impedance to port 2."""
    omega = 2 * math.pi * frequency
    y = 1j * omega * capacitance * reference
    z = (resistance + 1j * omega * inductance) / reference
    d = 2 + z + y + y * z
    return np.array([[(z - y - y * z) / d, 2 / d], [2 / d, (z - y + y * z) / d]])


def series_resistance(s, reference=1.0):
    """Resistance seen at port 1 with port 2 shorted."""
    z = s_to_z(s, np.array([reference, reference]))
    return (z[0, 0] - z[0, 1] * z[1, 0] / z[1, 1]).real


# Reference implementation of the two core steps, kept independent of the code under test.

def reference_grid(source, per_decade=100, low=1e3, high=1e7, tolerance=1e-9):
    source = np.asarray(source, float)
    positive = source[source > 0]
    count = int(round(per_decade * np.log10(high / low))) + 1
    add = np.logspace(np.log10(low), np.log10(high), count)
    add = add[(add > positive.min()) & (add < positive.max())]
    index = np.searchsorted(positive, add)
    near = np.zeros(len(add), bool)
    for k in (index - 1, index):
        ok = (k >= 0) & (k < len(positive))
        near[ok] |= np.abs(add[ok] - positive[k[ok]]) <= tolerance * add[ok]
    return np.unique(np.concatenate([source, add[~near]]))


def reference_resample(source, matrices, targets, axis):
    source, targets = np.asarray(source, float), np.asarray(targets, float)
    ps, pn = source > 0, targets > 0
    x, xp = (np.log(targets[pn]), np.log(source[ps])) if axis == "log" else (targets[pn], source[ps])
    flat = matrices.reshape(len(source), -1)
    out = np.empty((len(targets), flat.shape[1]), complex)
    for column in range(flat.shape[1]):
        out[pn, column] = (np.interp(x, xp, flat[ps, column].real)
                           + 1j * np.interp(x, xp, flat[ps, column].imag))
    result = out.reshape((len(targets),) + matrices.shape[1:])
    result[np.searchsorted(targets, source)] = matrices
    return result


# Frequency grid ---------------------------------------------------------------------------


def test_every_source_point_stays_and_a_decade_band_adds_the_expected_points():
    source = solver_grid()
    assert len(source) == 626
    plan = build_plan([Band(1e3, 1e7, DECADE_COUNT, 100)], source)
    grid = plan.frequencies
    assert len(grid) == 1013 and int(plan.added.sum()) == 387
    assert np.array_equal(grid[~plan.added], source)
    assert plan.ranges == ((1e3, 1e7, 401, 387),)
    assert int(((source >= 1e5) & (source <= 1e6)).sum()) == 9
    assert int(((grid >= 1e5) & (grid <= 1e6)).sum()) == 105
    # Ascending, inside the source range and free of near duplicates.
    assert grid[0] == source[0] and grid[-1] == source[-1]
    assert (np.diff(grid)[1:] > 1e-9 * grid[2:]).all()
    assert np.allclose(grid, reference_grid(source), rtol=1e-12, atol=0)


def test_default_bands_keep_the_source_points_and_report_the_new_ones():
    plan = build_plan(DEFAULT_BANDS, solver_grid())
    assert [item[2:] for item in plan.ranges] == [(101, 97), (401, 387), (1991, 1440)]
    assert len(plan.frequencies) == 626 + 97 + 387 + 1440
    # 0, 10, 100 and 1000 Hz are source points already, so the first new point is 20 Hz.
    assert plan.frequencies[plan.added][:3].tolist() == [20.0, 30.0, 40.0]
    assert not plan.added[0] and plan.frequencies[0] == 0.0
    # A source that already sits on every band point gains nothing.
    grid = np.unique(np.concatenate((np.arange(101) * 10.0, 1e3 * 10 ** (np.arange(401) / 100),
                                     1e7 + np.arange(1991) * 1e6)))
    plan = build_plan(DEFAULT_BANDS, grid)
    assert len(grid) == 2491 and not plan.added.any() and np.array_equal(plan.frequencies, grid)


def test_bands_shrink_to_a_narrower_source_range():
    source = np.array([150.0, 5e5, 1.2345e9])
    plan = build_plan(DEFAULT_BANDS, source)
    grid = plan.frequencies
    assert grid[0] == 150.0 and grid[-1] == 1.2345e9 and np.isin(source, grid).all()
    # The linear count keeps its number of points inside the remaining range.
    assert plan.ranges[0] == (150.0, 1e3, 101, 100)
    assert np.allclose(np.diff(grid[:101]), 8.5)
    # Step and decade grids stay anchored at their nominal start.
    assert plan.ranges[2] == (1e7, 1.2345e9, 1226, 1225)
    assert grid[-2] == 1.234e9
    inside = build_plan(DEFAULT_BANDS, np.array([2.5e3, 4e6]))
    assert inside.frequencies[0] == 2.5e3 and inside.frequencies[-1] == 4e6
    assert math.isclose(inside.frequencies[1], 1e3 * 10 ** 0.40, rel_tol=1e-15)
    assert inside.ranges[0] is None and inside.ranges[2] is None


def test_nothing_is_added_between_zero_and_the_first_positive_source_frequency():
    plan = build_plan([Band(0, 1e3, LINEAR_COUNT, 11)], np.array([0.0, 250.0, 5e3]))
    assert plan.frequencies.tolist() == [0.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 5e3]
    assert plan.ranges[0] == (0.0, 1e3, 9, 8)
    only_dc = build_plan([Band(0, 1e3, LINEAR_COUNT, 11)], np.array([0.0]))
    assert only_dc.frequencies.tolist() == [0.0] and not only_dc.added.any()


def test_band_points_next_to_a_source_frequency_give_way_to_it():
    near, far = 2000.0 * (1 + 5e-10), 3000.0 * (1 + 5e-9)
    plan = build_plan([Band(1e3, 5e3, LINEAR_STEP, 1e3)], np.array([1e3, near, far, 5e3]))
    # 2 kHz matches the source within 1e-9 and is dropped; 3 kHz does not and stays next to it.
    assert plan.frequencies.tolist() == [1e3, near, 3e3, far, 4e3, 5e3]
    assert plan.added.tolist() == [False, False, True, False, True, False]
    assert plan.ranges[0] == (1e3, 5e3, 5, 2)
    overlapping = build_plan([Band(1e3, 4e3, LINEAR_STEP, 1e3), Band(1e3, 4e3, LINEAR_COUNT, 4)],
                             np.array([1e3, 4e3]))
    assert overlapping.frequencies.tolist() == [1e3, 2e3, 3e3, 4e3]


def test_plan_rejects_invalid_or_excessive_bands():
    source = np.array([0.0, 1e9])
    for band, message in ((Band(10, 10, LINEAR_COUNT, 5), "끝 주파수"),
                          (Band(0, 10, LINEAR_COUNT, 1), "2 이상"),
                          (Band(0, 10, DECADE_COUNT, 10), "0보다"),
                          (Band(1, 10, DECADE_COUNT, 0.5), "1 이상"),
                          (Band(0, 10, LINEAR_STEP, 0), "0보다"),
                          (Band(0, 1e9, LINEAR_STEP, 1e-300), "넘습니다"),
                          (Band(0, 1e9, LINEAR_STEP, 1), "넘습니다")):
        with pytest.raises(ValueError, match="구간 1: .*" + message):
            build_plan([band], source)
    with pytest.raises(ValueError, match="밖에"):
        build_plan(DEFAULT_BANDS, np.array([3e9, 5e9]))
    with pytest.raises(ValueError, match="하나 이상"):
        build_plan([], source)


def test_frequency_text_round_trips():
    for text, value in (("0", 0.0), ("1k", 1e3), ("10MHz", 1e7), ("2G", 2e9), (" 0.1 khz ", 100.0),
                        ("1.5meg", 1.5e6), ("1e3", 1e3), ("7 Hz", 7.0), ("3M", 3e6)):
        assert parse_frequency(text) == value
        assert parse_frequency(format_frequency(value)) == value
    assert format_frequency(123456789.0) == "123.456789 MHz"
    for text in ("", "abc", "-1", "1 kk", "1e999", "1e999999k", "1e9999999999999999999"):
        with pytest.raises(ValueError):
            parse_frequency(text)


def test_option_settings_defaults_and_per_port_references():
    assert option_settings("#", 2) == option_settings("# GHZ S MA R 50", 2)
    settings = option_settings("# kHz DB S R 25 50 75", 3)
    assert (settings.unit, settings.data_format, settings.references) == (1e3, "DB", (25.0, 50.0, 75.0))
    assert option_settings("# R 75 MHz RI", 1).references == (75.0,)


# Interpolation ----------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [LOG_AXIS, LINEAR_AXIS])
def test_values_match_the_reference_implementation(tmp_path, axis):
    source = np.concatenate(([0.0], 10 ** (np.arange(8, 49) / 8)))
    generator = np.random.default_rng(3)
    matrices = generator.standard_normal((len(source), 3, 3)) + 1j * generator.standard_normal((len(source), 3, 3))
    document = write(tmp_path / "source.s3p", touchstone(source, matrices))
    network = load_network(document)
    assert np.array_equal(np.array(network.matrices), matrices)
    plan = build_plan([Band(1e3, 1e6, DECADE_COUNT, 100)], network.frequencies)
    expected = reference_resample(source, matrices, plan.frequencies, axis)
    entries = [(row, column) for row in range(3) for column in range(3)]
    _, preview = compute_traces(network, plan.frequencies, entries, "S", np.ones(3, bool), axis)
    assert np.allclose(preview.reshape(-1, 3, 3), expected, rtol=1e-12, atol=1e-13)
    target = tmp_path / "result.s3p"
    export_interpolated(document, document.ports, target, network.frequencies, plan.frequencies,
                        np.ones(3, bool), axis)
    result = load_network(read_touchstone(target))
    assert np.array_equal(result.frequencies[~plan.added], source)
    assert np.allclose(result.frequencies, plan.frequencies, rtol=1e-14, atol=0)
    assert np.allclose(np.array(result.matrices), expected, rtol=1e-12, atol=1e-13)
    # Source points are copied, not interpolated: their values are bit-identical.
    assert np.array_equal(np.array(result.matrices)[~plan.added], matrices)


def test_axis_changes_where_a_point_lies_between_its_neighbours(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n0 4 0\n100 0 0\n10000 2 -2\n")
    network = load_network(document)
    targets = np.array([0.0, 50.0, 100.0, 1000.0, 5050.0, 10000.0])
    ones = np.array([True])
    log = compute_traces(network, targets, [(0, 0)], "S", ones, LOG_AXIS)[1][:, 0]
    linear = compute_traces(network, targets, [(0, 0)], "S", ones, LINEAR_AXIS)[1][:, 0]
    # 1 kHz is halfway from 100 Hz to 10 kHz on a log axis; 5.05 kHz is halfway on a linear axis.
    assert log[3] == pytest.approx(1 - 1j) and linear[4] == pytest.approx(1 - 1j)
    assert linear[3] == pytest.approx((2 - 2j) * 900 / 9900)
    # The interval that starts at 0 Hz is linear on both axes.
    assert log[1] == linear[1] == 2
    assert log[[0, 2, 5]].tolist() == linear[[0, 2, 5]].tolist() == [4, 0, 2 - 2j]


def test_export_keeps_source_text_and_writes_linear_real_imaginary_values(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n0 1.000 0.0\n100 -1.0 0.50\n300 0.0 -1.0\n")
    network = load_network(document)
    assert network.frequencies.tolist() == [0.0, 100.0, 300.0] and document.point_count == 3
    bands = [Band(0, 300, LINEAR_STEP, 50)]
    plan = build_plan(bands, network.frequencies)
    assert plan.frequencies.tolist() == [0, 100, 150, 200, 250, 300]
    notes = describe(document, document.ports, bands, plan, [True], LINEAR_AXIS,
                     datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    target = tmp_path / "result.s1p"
    export_interpolated(document, document.ports, target, network.frequencies, plan.frequencies, [True],
                        LINEAR_AXIS, notes)
    lines = target.read_bytes().decode().splitlines()
    assert lines[:2] == ["# Hz S RI R 50", "! Port[1] = Port 1"]
    assert lines[2].startswith("! Resampled by Touchstone Editor ") and lines[2].endswith(" on 2026-01-02 03:04:05+00:00.")
    assert lines[3:8] == [
        "! Source file: source.s1p",
        "! Frequency points: 3 from the source, 3 added, 6 in total.",
        "! Interpolation: real/imaginary (INTDATTYP=RI equivalent) on a linear frequency axis for all ports.",
        "! Band 1: 0 Hz to 300 Hz, step 50 Hz, 3 new points.",
        "! Added points are interpolated values, not solver results. Source frequencies keep their original values."]
    # Source frequencies keep their text; the others are linear in the real and imaginary parts.
    assert lines[8:] == ["0 1.000 0.0", "100 -1.0 0.50", "150 -0.75 0.125", "200 -0.5 -0.25",
                         "250 -0.25 -0.625", "300 0.0 -1.0"]
    assert read_touchstone(target).ports[0].name == "Port 1"


def test_header_notes_name_the_selected_ports_and_the_axis(tmp_path):
    source = np.array([1e3, 1e4])
    document = write(tmp_path / "이름.s3p", touchstone(source, np.zeros((2, 3, 3), complex)))
    bands = [Band(1e3, 1e4, DECADE_COUNT, 10), Band(1e5, 1e6, LINEAR_COUNT, 5), Band(1e3, 1e4, LINEAR_COUNT, 3)]
    plan = build_plan(bands, source)
    ports = [document.ports[2], document.ports[0], document.ports[1]]
    mixed = describe(document, ports, bands, plan, [True, False, True], LOG_AXIS)
    assert "! Source file: 이름.s3p".removeprefix("! ") in mixed
    assert ("Interpolation on a log-frequency axis: real/imaginary (INTDATTYP=RI equivalent) for ports 1-2; "
            "magnitude/angle for entries between the other ports.") in mixed
    assert "Band 1: 1 kHz to 10 kHz, 10 points per decade, 9 new points." in mixed
    assert "Band 3: 1 kHz to 10 kHz, 3 points with linear spacing, 1 new points." in mixed
    assert not any(line.startswith("Band 2") for line in mixed)
    none = describe(document, ports, bands, plan, [False] * 3, LINEAR_AXIS)
    assert "Interpolation: magnitude/angle on a linear frequency axis for all ports." in none


def test_unselected_ports_use_magnitude_and_shortest_arc_angle(tmp_path):
    # Port 2 rotates through the 180-degree wrap, where magnitude/angle and real/imaginary differ most.
    document = write(tmp_path / "source.s2p",
                     "# Hz S MA R 50\n"
                     "1 1.0 0  0.1 90  0.1 90  1.0 170\n"
                     "3 0.5 90 0.3 90  0.3 90  0.5 -170\n")
    network = load_network(document)
    target = tmp_path / "result.s2p"
    export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0, 2.0, 3.0]),
                        [True, False], LINEAR_AXIS)
    values = numbers(target)
    assert list(values) == ["1", "2", "3"]
    assert values["1"] == [1.0, 0, 0.1, 90, 0.1, 90, 1.0, 170]
    middle = [cmath.rect(values["2"][i], math.radians(values["2"][i + 1])) for i in range(0, 8, 2)]
    s11, s21, s12, s22 = middle
    assert s11 == pytest.approx(0.5 + 0.25j)
    assert s21 == pytest.approx(0.2j) and s12 == pytest.approx(0.2j)
    # Port 2's own term keeps its magnitude path and turns through 180 degrees, not through zero.
    assert abs(s22) == pytest.approx(0.75)
    assert abs(math.degrees(cmath.phase(s22))) == pytest.approx(180)
    # With real/imaginary interpolation the same term shrinks towards the chord.
    all_ri = compute_traces(network, np.array([2.0]), [(1, 1)], "S", np.array([True, True]), LINEAR_AXIS)[1][0, 0]
    assert all_ri == pytest.approx(complex(0.75 * math.cos(math.radians(170)), 0.25 * math.sin(math.radians(170))))
    assert abs(all_ri) < 0.74
    mixed = compute_traces(network, np.array([2.0]), [(1, 1), (0, 0), (1, 0)], "S", np.array([True, False]),
                           LINEAR_AXIS)[1][0]
    assert mixed.tolist() == pytest.approx([s22, s11, s21])


@pytest.mark.parametrize("value", [2j, -0.3 - 0.4j, 0.3 + 0.4j, -0.3 + 0.4j, 0.3 - 0.4j, -1, 1])
def test_zero_magnitude_end_keeps_the_other_angle(tmp_path, value):
    # A zero has no angle; in every quadrant the magnitude must ramp along the other end's direction.
    document = write(tmp_path / "source.s1p",
                     f"# Hz S RI R 50\n1 0 0\n3 {value.real} {value.imag}\n5 0 0\n7 -0.0 -0.0\n")
    network = load_network(document)
    values = compute_traces(network, np.array([2.0, 4.0, 6.0]), [(0, 0)], "S", np.array([False]),
                            LINEAR_AXIS)[1][:, 0]
    assert values.tolist() == pytest.approx([value / 2, value / 2, 0])


@pytest.mark.parametrize("data_format", ["RI", "MA", "DB"])
def test_every_number_format_round_trips_through_interpolation(tmp_path, data_format):
    truth = {1.0: np.array([[0.2 + 0.1j, 0.05j, -0.3], [0.01, -0.9 + 0.01j, 0.2 - 0.2j], [0.4j, 0.3, 0.6]]),
             2.0: np.array([[0.1 - 0.3j, 0.02, 0.1j], [0.5, -0.8 - 0.02j, 0.1], [-0.2, 0.3j, 0.5 + 0.1j]])}

    def text(value):
        value = complex(value)
        if data_format == "RI":
            return f"{value.real!r} {value.imag!r}"
        magnitude = abs(value) if data_format == "MA" else 20 * math.log10(abs(value))
        return f"{magnitude!r} {math.degrees(cmath.phase(value))!r}"

    rows = "".join(f"{frequency} " + "\n ".join(" ".join(text(value) for value in row) for row in matrix) + "\n"
                   for frequency, matrix in truth.items())
    document = write(tmp_path / "source.s3p", f"# kHz S {data_format} R 50\n" + rows)
    network = load_network(document)
    assert network.frequencies.tolist() == [1e3, 2e3]
    assert np.allclose(network.matrices[0], truth[1.0], rtol=1e-14, atol=1e-16)
    target = tmp_path / "result.s3p"
    export_interpolated(document, document.ports, target, network.frequencies, np.array([1e3, 1.25e3, 2e3]),
                        [True, True, True], LINEAR_AXIS)
    result = load_network(read_touchstone(target))
    # The option line, and with it the number format and unit, is unchanged.
    assert read_touchstone(target).option_line == document.option_line
    assert result.frequencies.tolist() == [1e3, 1.25e3, 2e3]
    assert np.allclose(result.matrices[1], 0.75 * truth[1.0] + 0.25 * truth[2.0], rtol=1e-13, atol=1e-15)
    assert np.array_equal(result.matrices[0], network.matrices[0])
    assert np.array_equal(result.matrices[2], network.matrices[1])


def test_two_port_order_and_reordered_export(tmp_path):
    # Two-port rows are stored as 11, 21, 12, 22.
    document = write(tmp_path / "source.s2p",
                     "! Port[1] = A\n! Port[2] = B\n# Hz S RI R 10 40\n"
                     "0 11 0 21 0 12 0 22 0\n10 111 0 121 0 112 0 122 0\n")
    network = load_network(document)
    assert network.matrices[0].real.tolist() == [[11, 12], [21, 22]]
    assert network.references.tolist() == [10, 40]
    target = tmp_path / "result.s2p"
    export_interpolated(document, document.ports[::-1], target, network.frequencies, np.array([0.0, 5.0, 10.0]),
                        [True, True])
    lines = target.read_bytes().splitlines()
    assert lines == [b"! Port[1] = B", b"! Port[2] = A", b"# Hz S RI R 40 10",
                     b"0 22 0 12 0 21 0 11 0", b"5 72 0 62 0 71 0 61 0", b"10 122 0 112 0 121 0 111 0"]


def test_wide_matrix_layout_and_dropped_point_comments(tmp_path):
    count = 5
    rows = lambda scale: "\n".join(" ".join(f"{scale * (10 * r + c)} 0" for c in range(1, count + 1)[start:start + 4])
                                   for r in range(1, count + 1) for start in (0, 4))
    document = write(tmp_path / "source.s5p",
                     "! Port 1 = U1-A1 NET/0\n# Hz S RI R 50\n"
                     f"1 {rows(1)}\n! Gamma 1 0 1 0 1 0 1 0 1 0\n! Port Impedance 5 0 5 0 5 0 5 0 5 0\n"
                     f"3 {rows(3)} ! trailing\n! closing comment\n")
    network = load_network(document)
    target = tmp_path / "result.s5p"
    export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0, 2.0, 3.0]),
                        [True] * count, LINEAR_AXIS)
    lines = target.read_bytes().splitlines()
    assert b"Gamma" not in target.read_bytes() and b"closing" not in target.read_bytes()
    assert lines[0] == b"! Port[1] = NET/0"
    # Each matrix row starts a line and holds at most four pairs per line.
    middle = lines[lines.index(b"\t51 0 52 0 53 0 54 0") + 2:]
    assert middle[0] == b"2 22 0 24 0 26 0 28 0"
    assert middle[1] == b"\t30 0"
    assert middle[2] == b"\t42 0 44 0 46 0 48 0"
    assert len(middle) == 2 * (2 * count)
    assert np.allclose(load_network(read_touchstone(target)).matrices[1], 2 * network.matrices[0])


def test_export_requires_the_loaded_frequencies_and_can_be_cancelled(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n1 0 0\n2 1 0\n3 2 0\n")
    network = load_network(document)
    target = tmp_path / "result.s1p"
    target.write_bytes(b"keep")
    for source, targets in ((np.array([1.0, 2.0]), np.array([1.5])),
                            (np.array([1.0, 2.5, 3.0]), np.array([1.5])),
                            (network.frequencies, np.array([0.5, 1.0])),
                            (network.frequencies, np.array([1.0, 4.0]))):
        with pytest.raises(ValueError, match="source file changed"):
            export_interpolated(document, document.ports, target, source, targets, [True])
        assert target.read_bytes() == b"keep"
    with pytest.raises(ValueError, match="port count"):
        export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0]), [True, True])
    with pytest.raises(CancelledError):
        export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0, 1.5]), [True],
                            cancel=lambda: True)
    assert target.read_bytes() == b"keep"
    assert list(tmp_path.glob("*.tmp")) == []


def test_memory_limit_keeps_frequencies_metrics_and_streaming_export(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n1 0 0\n2 1 0\n3 2 0\n")
    network = load_network(document, memory_limit=20)
    assert network.matrices is None
    assert network.frequencies.tolist() == [1.0, 2.0, 3.0]
    assert network.differences.shape == network.phase_steps.shape == (2, 1)
    target = tmp_path / "result.s1p"
    export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0, 2.5]), [True],
                        LINEAR_AXIS)
    assert target.read_bytes().splitlines()[-2:] == [b"1 0 0", b"2.5 1.5 0"]


# Why real/imaginary: the false series resistance ---------------------------------------------


def test_magnitude_angle_interpolation_invents_series_resistance_and_real_imaginary_does_not():
    """Between two points of a shunt-capacitor block, magnitude/angle interpolation inflates the small
    series resistance by more than half; real/imaginary interpolation keeps it within a percent."""
    from touchstone_editor.interpolation import _Segment

    first, second = 10 ** (41 / 8), 10 ** (42 / 8)
    start, stop = lumped_block(first), lumped_block(second)
    # The shunt capacitor is a thousand times larger than the series path, so port 1 sees the 4 mOhm.
    assert series_resistance(start) == pytest.approx(4e-3, rel=1e-3)
    polar = np.zeros((2, 2), dtype=bool)
    linear = [series_resistance(_Segment(start, stop).at(position)) for position in np.linspace(0, 1, 21)]
    bent = [series_resistance(_Segment(start, stop, polar).at(position)) for position in np.linspace(0, 1, 21)]
    assert max(linear) == pytest.approx(4e-3, rel=0.01) and min(linear) == pytest.approx(4e-3, rel=0.01)
    assert max(bent) > 1.5 * 4e-3
    assert bent[0] == pytest.approx(series_resistance(start)) and bent[-1] == pytest.approx(series_resistance(stop))
    # The resistance sits in how far S21 - S11 is from one; real/imaginary interpolation keeps that relation.
    relation = lambda s: 1 - (s[1, 0] - s[0, 0]).real
    exact = relation(lumped_block(math.sqrt(first * second)))
    assert relation(_Segment(start, stop).at(0.5)) == pytest.approx(exact, rel=0.03)
    assert relation(_Segment(start, stop, polar).at(0.5)) > 1.5 * exact


def test_resampled_file_survives_a_second_magnitude_angle_interpolation(tmp_path):
    """After real/imaginary resampling at 100 points per decade, a tool that interpolates the new file on
    magnitude and angle stays close to the true resistance; on the original file it does not."""
    from touchstone_editor.interpolation import _Segment

    source = 10 ** (np.arange(24, 57) / 8)
    document = write(tmp_path / "source.s2p", touchstone(source, [lumped_block(f) for f in source], 1.0))
    network = load_network(document)
    plan = build_plan([Band(1e3, 1e7, DECADE_COUNT, 100)], network.frequencies)
    target = tmp_path / "result.s2p"
    export_interpolated(document, document.ports, target, network.frequencies, plan.frequencies, [True, True])
    resampled = load_network(read_touchstone(target))
    polar = np.zeros((2, 2), dtype=bool)

    def worst(data):
        """Largest resistance a tool finds halfway between neighbouring points from 100 kHz to 1 MHz."""
        pairs = [(k, k + 1) for k in range(len(data.frequencies) - 1)
                 if 1e5 <= data.frequencies[k] and data.frequencies[k + 1] <= 1e6]
        return max(series_resistance(_Segment(data.matrices[a], data.matrices[b], polar).at(0.5)) for a, b in pairs)

    assert worst(network) > 1.5 * 4e-3
    assert worst(resampled) == pytest.approx(4e-3, rel=0.02)


def test_load_measures_angle_steps_and_the_interpolation_gap(tmp_path):
    # Port 1 turns 90 degrees per point at full magnitude; port 2 barely moves; the coupling is negligible.
    matrices = [np.array([[cmath.rect(1, math.radians(90 * k)), 1e-3], [1e-3, 0.5 + 0.01 * k]]) for k in range(3)]
    document = write(tmp_path / "source.s2p", touchstone([1e3, 2e3, 4e3], matrices))
    network = load_network(document)
    assert network.phase_steps[:, 0] == pytest.approx([90, 90])
    assert network.phase_steps[:, 1] == pytest.approx([0, 0], abs=1e-9)
    assert (network.phase_steps[:, 0] > PHASE_LIMIT).all()
    # Halfway through a quarter turn the chord is cos(45 degrees) of the arc.
    assert network.differences[:, 0] == pytest.approx([1 - math.cos(math.pi / 4)] * 2)
    assert network.differences[:, 1] == pytest.approx([0, 0], abs=1e-12)
    assert network.asymmetry == 0
    plan = build_plan([Band(2e3, 4e3, LINEAR_COUNT, 3)], network.frequencies)
    assert touched_intervals(network, plan).tolist() == [1]


def test_check_reports_gain_symmetry_and_the_remaining_gap(tmp_path):
    source = 10 ** (np.arange(24, 57) / 8)
    document = write(tmp_path / "source.s2p", touchstone(source, [lumped_block(f) for f in source], 1.0))
    network = load_network(document)
    both = np.array([True, True])
    original = float(network.differences.max())
    progress = []
    plan = build_plan([Band(1e3, 1e7, DECADE_COUNT, 100)], network.frequencies)
    checks = check_plan(network, plan, both, LOG_AXIS, progress.append)
    assert progress and progress[-1] <= 99
    # Real/imaginary interpolation is a convex combination: it cannot raise the gain or break symmetry.
    assert checks.added_gain <= checks.source_gain <= 1 + 1e-12
    assert checks.source_gain == network.gain == pytest.approx(max(np.linalg.norm(m, 2) for m in network.matrices))
    assert checks.asymmetry <= 1e-15
    # Twelve times denser points shrink the gap by about the square of that.
    assert checks.remaining == pytest.approx(original * (8 / 100) ** 2, rel=0.25)
    start, stop = checks.remaining_range
    assert stop / start == pytest.approx(10 ** 0.01, rel=1e-6)
    # A band away from the worst interval leaves the gap where it was.
    elsewhere = check_plan(network, build_plan([Band(5e6, 1e7, DECADE_COUNT, 100)], network.frequencies), both)
    assert elsewhere.remaining == pytest.approx(original)
    # With no real/imaginary entries nothing is at risk from a second interpolation.
    assert check_plan(network, plan, np.array([False, False])).remaining == 0
    with pytest.raises(CancelledError):
        check_plan(network, plan, both, cancel=lambda: True)


# S to Z -----------------------------------------------------------------------------------


def test_z_conversion_matches_circuit_theory():
    # One port: Z = R (1 + S) / (1 - S).
    assert s_to_z(np.array([[0.2 - 0.4j]]), np.array([50.0]))[0, 0] == pytest.approx(50 * (1.2 - 0.4j) / (0.8 + 0.4j))
    # A shunt impedance between two ports has Z11 = Z12 = Z21 = Z22 = Zs.
    shunt = 3 + 4j
    for references in ((50.0, 50.0), (10.0, 75.0)):
        root = np.sqrt(np.array(references))
        impedance = np.full((2, 2), shunt)
        normal = impedance / root[:, None] / root[None, :]
        s = (normal - np.eye(2)) @ np.linalg.inv(normal + np.eye(2))
        assert np.allclose(s_to_z(s, np.array(references)), impedance)
        assert np.allclose(s_to_z(s, np.array(references), np.array([1])), impedance[:, [1]])
    # An ideal open has no finite impedance.
    assert np.isnan(s_to_z(np.eye(2, dtype=complex), np.array([50.0, 50.0]))).all()


def test_z_traces_convert_the_interpolated_matrix(tmp_path):
    # Interpolating S and converting differs from interpolating Z; the preview must show the former.
    document = write(tmp_path / "source.s2p",
                     "# Hz S RI R 50\n1 -0.9 0.01 0.05 0.02 0.05 0.02 -0.8 0.03\n"
                     "3 -0.5 0.20 0.10 0.05 0.10 0.05 -0.6 0.10\n")
    network = load_network(document)
    progress = []
    source, result = compute_traces(network, np.array([1.0, 2.0, 3.0]), [(0, 0), (1, 0), (1, 1)], "Z",
                                    np.array([True, True]), LINEAR_AXIS, progress.append)
    middle = s_to_z((network.matrices[0] + network.matrices[1]) / 2, network.references)
    assert result[1].tolist() == pytest.approx([middle[0, 0], middle[1, 0], middle[1, 1]])
    assert np.allclose(result[[0, 2]], source)
    assert not np.allclose(result[1], (source[0] + source[1]) / 2, rtol=1e-3)
    with pytest.raises(CancelledError):
        compute_traces(network, np.arange(1.0, 3.0, 0.01), [(0, 0)], "Z", np.array([True, True]),
                       cancel=lambda: True)
