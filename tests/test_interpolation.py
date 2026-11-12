import cmath
import math

import numpy as np
import pytest

from touchstone_editor.core import CancelledError, option_settings, read_touchstone
from touchstone_editor.interpolation import (DECADE_COUNT, DEFAULT_BANDS, LINEAR_COUNT, LINEAR_STEP, Band,
                                             build_plan, compute_traces, export_interpolated,
                                             format_frequency, load_network, parse_frequency, s_to_z)


def write(path, text):
    path.write_bytes(text.encode())
    return read_touchstone(path)


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


def test_default_bands_match_the_requested_sweeps():
    plan = build_plan(DEFAULT_BANDS, np.array([0.0, 5e4, 2e9, 2e10]))
    grid = plan.frequencies
    assert len(grid) == 101 + 400 + 1990
    assert np.array_equal(grid[:101], np.arange(101) * 10.0)
    assert np.allclose(grid[100:501], 1e3 * 10 ** (np.arange(401) / 100), rtol=1e-15)
    assert np.array_equal(grid[500:], 1e7 + np.arange(1991) * 1e6)
    assert [count for _, _, count in plan.ranges] == [101, 401, 1991]
    assert grid[200] == 1e4 and grid[500] == 1e7


def test_bands_shrink_to_a_narrower_source_range():
    plan = build_plan(DEFAULT_BANDS, np.array([150.0, 5e5, 1.2345e9]))
    grid = plan.frequencies
    assert grid[0] == 150.0 and grid[-1] == 1.2345e9
    # The linear count keeps its number of points inside the remaining range.
    assert plan.ranges[0] == (150.0, 1e3, 101)
    assert np.allclose(np.diff(grid[:101]), 8.5)
    # Step and decade grids stay anchored at their nominal start.
    assert plan.ranges[2] == (1e7, 1.2345e9, 1226)
    assert grid[-2] == 1.234e9
    inside = build_plan(DEFAULT_BANDS, np.array([2.5e3, 4e6])).frequencies
    assert inside[0] == 2.5e3 and inside[-1] == 4e6
    assert math.isclose(inside[1], 1e3 * 10 ** 0.40, rel_tol=1e-15)
    assert build_plan(DEFAULT_BANDS, np.array([2.5e3, 4e6])).ranges[0] is None


def test_plan_rejects_invalid_or_excessive_bands():
    source = np.array([0.0, 1e9])
    for band, message in ((Band(10, 10, LINEAR_COUNT, 5), "끝 주파수"),
                          (Band(0, 10, LINEAR_COUNT, 1), "2 이상"),
                          (Band(0, 10, DECADE_COUNT, 10), "0보다"),
                          (Band(1, 10, DECADE_COUNT, 0.5), "1 이상"),
                          (Band(0, 10, LINEAR_STEP, 0), "0보다"),
                          (Band(0, 1e9, LINEAR_STEP, 1e-300), "넘습니다"),
                          (Band(0, 1e9, LINEAR_STEP, 1), "넘습니다")):
        with pytest.raises(ValueError, match=message):
            build_plan([band], source)
    with pytest.raises(ValueError, match="밖에"):
        build_plan(DEFAULT_BANDS, np.array([3e9, 5e9]))
    with pytest.raises(ValueError, match="하나 이상"):
        build_plan([], source)


def test_keep_source_adds_every_original_frequency_and_snaps_near_duplicates():
    source = np.array([0.0, 123.0, 1000.0 * (1 + 1e-14), 5e9])
    plan = build_plan([Band(0, 1e3, LINEAR_COUNT, 3)], source, keep_source=True)
    assert plan.frequencies.tolist() == [0.0, 123.0, 500.0, source[2], 5e9]
    assert build_plan([], source, keep_source=True).frequencies.tolist() == source.tolist()


def test_frequency_text_round_trips():
    for text, value in (("0", 0.0), ("1k", 1e3), ("10MHz", 1e7), ("2G", 2e9), (" 0.1 khz ", 100.0),
                        ("1.5meg", 1.5e6), ("1e3", 1e3), ("7 Hz", 7.0), ("3M", 3e6)):
        assert parse_frequency(text) == value
        assert parse_frequency(format_frequency(value)) == value
    assert format_frequency(123456789.0) == "123.456789 MHz"
    for text in ("", "abc", "-1", "1 kk", "1e999"):
        with pytest.raises(ValueError):
            parse_frequency(text)


def test_option_settings_defaults_and_per_port_references():
    assert option_settings("#", 2) == option_settings("# GHZ S MA R 50", 2)
    settings = option_settings("# kHz DB S R 25 50 75", 3)
    assert (settings.unit, settings.data_format, settings.references) == (1e3, "DB", (25.0, 50.0, 75.0))
    assert option_settings("# R 75 MHz RI", 1).references == (75.0,)


def test_real_imaginary_interpolation_and_verbatim_source_points(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n0 1.000 0.0\n100 -1.0 0.50\n300 0.0 -1.0\n")
    network = load_network(document)
    assert network.frequencies.tolist() == [0.0, 100.0, 300.0]
    assert document.point_count == 3
    plan = build_plan([Band(0, 300, LINEAR_STEP, 50)], network.frequencies)
    target = tmp_path / "result.s1p"
    export_interpolated(document, document.ports, target, network.frequencies, plan.frequencies, [True])
    lines = target.read_bytes().splitlines()
    assert lines[:2] == [b"# Hz S RI R 50", b"! Port[1] = Port 1"]
    assert lines[2].startswith(b"! Resampled by Touchstone Editor ")
    assert lines[2].endswith(b": 3 -> 7 frequency points.")
    assert b"INTDATTYP=RI equivalent" in lines[3] and b"all ports" in lines[3]
    # Source frequencies keep their text; the others are linear in the real and imaginary parts.
    assert lines[4:] == [b"0 1.000 0.0", b"50 0 0.25", b"100 -1.0 0.50", b"150 -0.75 0.125",
                         b"200 -0.5 -0.25", b"250 -0.25 -0.625", b"300 0.0 -1.0"]
    source, result = compute_traces(network, plan.frequencies, [(0, 0)], "S", np.array([True]))
    assert source[:, 0].tolist() == [1, -1 + 0.5j, -1j]
    assert result[:, 0].tolist() == [1, 0.25j, -1 + 0.5j, -0.75 + 0.125j, -0.5 - 0.25j, -0.25 - 0.625j, -1j]


def test_unselected_ports_use_magnitude_and_shortest_arc_angle(tmp_path):
    # Port 2 rotates through the 180-degree wrap, where magnitude/angle and real/imaginary differ most.
    document = write(tmp_path / "source.s2p",
                     "# Hz S MA R 50\n"
                     "1 1.0 0  0.1 90  0.1 90  1.0 170\n"
                     "3 0.5 90 0.3 90  0.3 90  0.5 -170\n")
    network = load_network(document)
    target = tmp_path / "result.s2p"
    export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0, 2.0, 3.0]),
                        [True, False])
    comment = target.read_bytes().splitlines()[4]
    assert b"for ports 1;" in comment and b"magnitude/angle linear for the other entries" in comment
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
    # With real/imaginary interpolation the same term would shrink towards the chord.
    all_ri = compute_traces(network, np.array([2.0]), [(1, 1)], "S", np.array([True, True]))[1][0, 0]
    assert all_ri == pytest.approx(complex(0.75 * math.cos(math.radians(170)), 0.25 * math.sin(math.radians(170))))
    assert abs(all_ri) < 0.74
    mixed = compute_traces(network, np.array([2.0]), [(1, 1), (0, 0), (1, 0)], "S", np.array([True, False]))[1][0]
    assert mixed.tolist() == pytest.approx([s22, s11, s21])


def test_zero_magnitude_end_keeps_the_other_angle(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n1 0 0\n3 0 2\n5 0 0\n")
    network = load_network(document)
    values = compute_traces(network, np.array([2.0, 4.0]), [(0, 0)], "S", np.array([False]))[1][:, 0]
    assert values.tolist() == pytest.approx([1j, 1j])


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
                        [True, True, True])
    result = load_network(read_touchstone(target))
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
    assert lines[:3] == [b"! Port[1] = B", b"! Port[2] = A", b"# Hz S RI R 40 10"]
    assert lines[5:] == [b"0 22 0 12 0 21 0 11 0", b"5 72 0 62 0 71 0 61 0", b"10 122 0 112 0 121 0 111 0"]


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
                        [True] * count)
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


def test_memory_limit_keeps_frequencies_and_export_still_streams(tmp_path):
    document = write(tmp_path / "source.s1p", "# Hz S RI R 50\n1 0 0\n2 1 0\n3 2 0\n")
    network = load_network(document, memory_limit=20)
    assert network.matrices is None
    assert network.frequencies.tolist() == [1.0, 2.0, 3.0]
    target = tmp_path / "result.s1p"
    export_interpolated(document, document.ports, target, network.frequencies, np.array([1.0, 2.5]), [True])
    assert target.read_bytes().splitlines()[-2:] == [b"1 0 0", b"2.5 1.5 0"]


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
                                    np.array([True, True]), progress.append)
    middle = s_to_z((network.matrices[0] + network.matrices[1]) / 2, network.references)
    assert result[1].tolist() == pytest.approx([middle[0, 0], middle[1, 0], middle[1, 1]])
    assert np.allclose(result[[0, 2]], source)
    assert not np.allclose(result[1], (source[0] + source[1]) / 2, rtol=1e-3)
    with pytest.raises(CancelledError):
        compute_traces(network, np.arange(1.0, 3.0, 0.01), [(0, 0)], "Z", np.array([True, True]),
                       cancel=lambda: True)
