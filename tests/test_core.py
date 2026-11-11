import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from touchstone_editor.core import CancelledError, export_touchstone, read_touchstone


def matrix_bytes(count, frequency=b"+1.00E+9"):
    def value(row, column):
        return f"{row}{column}.000e-3 -.{column}{row}E+0".encode()

    if count <= 2:
        cells = [value(row, column) for column in range(1, count + 1) for row in range(1, count + 1)]
        return b" ".join([frequency, *cells]) + b"\n"
    rows = [[value(row, column) for column in range(1, count + 1)] for row in range(1, count + 1)]
    lines = [b" ".join(row[start:start + 4]) for row in rows for start in range(0, count, 4)]
    return frequency + b" " + b"\n\t".join(lines) + b"\n"


class TouchstoneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def source(self, count, contents):
        path = self.directory / f"source.s{count}p"
        path.write_bytes(contents)
        return path

    def target(self, count):
        return self.directory / f"output.s{count}p"

    def test_names_only_preserves_data_and_unchanged_legacy_metadata(self):
        header = (b"\xef\xbb\xbf! Synthesized comments\r\n"
                  b"! Port101_A_0::SAMPLE\r\n! NETSLIST SAMPLE/0, OTHER\r\n"
                  b"! Port 1 = X4_0-J2 SAMPLE/0\r\n! Port 2 = U1-A1 OTHER CHANNEL\r\n"
                  b"#\tHz\tS\tRI\tR\t1\r\n")
        data = b"1.000 1e0 -0 +02 .3 -3E0 0 4. 0 ! preserve this\r\n\r\n"
        document = read_touchstone(self.source(2, header + data))
        self.assertEqual([port.name for port in document.ports], ["SAMPLE/0", "OTHER CHANNEL"])
        self.assertEqual(document.ports[0].original_name, "X4_0-J2 SAMPLE/0")
        self.assertIsNone(document.point_count)
        target = self.target(2)
        export_touchstone(document, document.ports, target)
        self.assertEqual(document.point_count, 1)
        expected = header.replace(b"Port 1 = X4_0-J2", b"Port[1] =").replace(b"Port 2 = U1-A1", b"Port[2] =")
        self.assertEqual(target.read_bytes(), expected + data)

        document.ports[0].name = "변경/0"
        export_touchstone(document, document.ports, target)
        output = target.read_bytes()
        self.assertTrue(output.endswith(data))
        self.assertNotIn(b"Port101_", output)
        self.assertNotIn(b"NETSLIST", output)
        self.assertEqual(read_touchstone(target).ports[0].name, "변경/0")

    def test_two_port_and_row_major_permutations(self):
        for count, order in ((2, [2, 1]), (3, [2, 3, 1]), (4, [3, 1, 4, 2]), (5, [4, 5, 1, 3, 2])):
            with self.subTest(count=count):
                original = matrix_bytes(count)
                references = b" ".join(str(25 * index).encode() for index in range(1, count + 1))
                path = self.source(count, b"! Port101_A_0::SAMPLE\n# KHZ DB S R " + references + b"\n" + original)
                document = read_touchstone(path)
                target = self.target(count)
                export_touchstone(document, [document.ports[index - 1] for index in order], target)
                result = read_touchstone(target)
                self.assertEqual(result.option_line, "# KHZ DB S R " + " ".join(str(25 * index) for index in order))
                self.assertEqual([port.name for port in result.ports], [f"Port {index}" for index in order])
                self.assertNotIn(b"Port101_", target.read_bytes())
                values = original.split()
                flattened = target.read_bytes().splitlines()[count + 1:]
                new_values = b" ".join(flattened).split()
                self.assertEqual(new_values[0], values[0])
                for new_row, old_row in enumerate(order):
                    for new_column, old_column in enumerate(order):
                        if count == 2:
                            new_offset = new_column * count + new_row
                            old_offset = (old_column - 1) * count + old_row - 1
                        else:
                            new_offset = new_row * count + new_column
                            old_offset = (old_row - 1) * count + old_column - 1
                        self.assertEqual(new_values[1 + 2 * new_offset:3 + 2 * new_offset],
                                         values[1 + 2 * old_offset:3 + 2 * old_offset])

    def test_partial_options_comments_and_mixed_line_endings(self):
        contents = b"# ma\r! Port[1] = Leading name\r\n1 .5 -12\r\n2 1 0\n"
        source = self.source(1, contents)
        document = read_touchstone(source)
        self.assertEqual(document.option_line, "# ma")
        self.assertEqual(document.ports[0].name, "Leading name")
        self.assertIsNone(document.point_count)
        target = self.target(1)
        export_touchstone(document, document.ports, target)
        self.assertEqual(document.point_count, 2)
        self.assertEqual(target.read_bytes(), contents)

    def test_open_reads_only_the_header_and_first_body_byte(self):
        header = b"! Port 1 = PKG-A12 CHANNEL\r\n# Hz S RI R 50\r\n! After option\r\n"
        source = self.source(1, header + b"1")
        with source.open("r+b") as stream:
            stream.seek(32 * 1024 * 1024)
            stream.write(b"\n")
        test = self
        with source.open("rb", buffering=0) as real_stream:
            class GuardedStream:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return None

                def fileno(self):
                    return real_stream.fileno()

                def read(self, size):
                    test.assertGreater(size, 0)
                    test.assertLessEqual(real_stream.tell() + size, len(header) + 1)
                    return real_stream.read(size)

            with patch.object(Path, "open", return_value=GuardedStream()) as open_file:
                document = read_touchstone(source)
            open_file.assert_called_once_with("rb", buffering=0)
            self.assertEqual(real_stream.tell(), len(header) + 1)
        self.assertEqual(document.ports[0].name, "CHANNEL")
        self.assertIsNone(document.point_count)

    def test_power_prefixes_cleaned_only_for_numbered_names(self):
        header = (b"! Port 1 = Z9-3K23 SIGNAL1\n! Port 2 = PKG-QB991 SIGNAL2\n"
                  b"! Port 3 = Net-input description\n! Port[4] = PKG-QB991 SIGNAL4\n#\n")
        document = read_touchstone(self.source(4, header + matrix_bytes(4)))
        self.assertEqual([port.name for port in document.ports],
                         ["SIGNAL1", "SIGNAL2", "Net-input description", "PKG-QB991 SIGNAL4"])
        self.assertEqual(document.ports[0].original_name, "Z9-3K23 SIGNAL1")

    def test_open_rejects_unsupported_or_malformed_headers(self):
        cases = [
            (1, b"[Version] 2.0\n#\n1 0 0\n", "Touchstone 2.x"),
            (1, b"1 0 0\n", "Expected an option line"),
            (1, b"# GHz Z RI R 50\n1 0 0\n", "option line"),
            (1, b"# S R -1\n1 0 0\n", "option line"),
            (1, b"# R 1 2\n1 0 0\n", "option line"),
            (1, b"# MHz\n# GHz\n1 0 0\n", "multiple option lines"),
            (1, b"! Port[2] = Other\n#\n1 0 0\n", "out-of-range"),
            (1, b"! Port 1 = \n#\n1 0 0\n", "missing"),
            (1, b"! Port 1 = First\n! Port[1] = Second\n#\n1 0 0\n", "duplicate"),
            (1, b"!.DiffChannels\n!.DiffChannels\n#\n", "malformed PowerSI"),
            (1, b"#\n!.DiffChannels\n1 0 0\n", "missing its end marker"),
            (1, b"!.EndDiffChannels\n#\n", "malformed PowerSI"),
        ]
        for count, contents, error in cases:
            with self.subTest(contents=contents):
                with self.assertRaisesRegex(ValueError, error):
                    read_touchstone(self.source(count, contents))

    def test_save_validates_data_before_replacing_output(self):
        cases = [
            (1, b"# MHz\n1 0 0\n# GHz\n2 0 0\n", "multiple option lines"),
            (1, b"#\n", "no S-parameter data"),
            (1, b"#\n1 NaN 0\n", "invalid numeric"),
            (1, b"#\n1 2e- 0\n", "invalid numeric"),
            (1, b"#\n+1 1 2\n1.0 3 4\n", "strictly increasing"),
            (1, b"#\n1 0 0\n! Port[1] = Late name\n", "must precede"),
            (1, b"#\n1 0 0 ! Port[1] = Inline name\n", "must precede"),
            (1, b"#\n1 0 0\n[End]\n", "Touchstone 2.x"),
            (2, b"#\n" + matrix_bytes(2) + b"1 2 3 4 5\n", "noise data"),
            (3, b"#\n1 1 0 2 0 3 0\n", "incomplete S-parameter matrix"),
            (3, b"#\n1 1 0 2 0\n", "incorrect number"),
            (5, b"#\n1 1 0 2 0 3 0 4 0\n5 0 6 0\n", "incorrect number"),
        ]
        for count, contents, error in cases:
            with self.subTest(contents=contents):
                source = self.source(count, contents)
                document = read_touchstone(source)
                target = self.target(count)
                target.write_bytes(b"existing output")
                orders = [document.ports]
                if count > 1:
                    orders.append(list(reversed(document.ports)))
                for order in orders:
                    with self.assertRaisesRegex(ValueError, error):
                        export_touchstone(document, order, target)
                    self.assertEqual(target.read_bytes(), b"existing output")
                    self.assertIsNone(document.point_count)
                    self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_rejects_invalid_permutations_names_and_target_aliases(self):
        source = self.source(2, b"#\n" + matrix_bytes(2))
        document = read_touchstone(source)
        first, second = document.ports
        target = self.target(2)
        for order in ([first], [first, first], [first, replace(second, source_index=3)],
                      [replace(first, source_index=True), second]):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                export_touchstone(document, order, target)
        for name in ("", "   ", "X!Y", "X\nY", "X\rY", "X\0Y"):
            with self.assertRaisesRegex(ValueError, "nonempty name"):
                export_touchstone(document, [replace(first, name=name), second], target)
        with self.assertRaisesRegex(ValueError, "extension"):
            export_touchstone(document, document.ports, self.target(1))
        with self.assertRaisesRegex(ValueError, "different from the source"):
            export_touchstone(document, document.ports, source)
        os.link(source, target)
        with self.assertRaisesRegex(ValueError, "different from the source"):
            export_touchstone(document, document.ports, target)
        target.unlink()
        try:
            target.symlink_to(source)
        except OSError:
            return  # Creating symlinks can require privileges on Windows.
        with self.assertRaisesRegex(ValueError, "different from the source"):
            export_touchstone(document, document.ports, target)

    def test_source_changes_before_or_during_export(self):
        contents = b"#\n" + matrix_bytes(2)
        source = self.source(2, contents)
        document = read_touchstone(source)
        target = self.target(2)
        target.write_bytes(b"existing output")
        source.write_bytes(contents + b"! changed\n")
        with self.assertRaisesRegex(ValueError, "source file changed"):
            export_touchstone(document, document.ports, target)

        source.write_bytes(contents)
        document = read_touchstone(source)

        def mutate(percent):
            if percent == 99:
                stamp = source.stat()
                source.write_bytes(contents.replace(b"+1.00", b"+2.00"))
                # Windows can give two writes in the same timer tick the same mtime.
                os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))

        with self.assertRaisesRegex(ValueError, "source file changed"):
            export_touchstone(document, document.ports, target, progress=mutate)
        self.assertEqual(target.read_bytes(), b"existing output")
        self.assertEqual(set(self.directory.iterdir()), {source, target})

    def test_changed_header_rejected_even_with_matching_file_stamp(self):
        source = self.source(1, b"! Port[1] = AAA\n#\n" + matrix_bytes(1))
        document = read_touchstone(source)
        source.write_bytes(b"! Port[1] = BBB\n#\n" + matrix_bytes(1))
        document._stamp = read_touchstone(source)._stamp
        with self.assertRaisesRegex(ValueError, "source header changed"):
            export_touchstone(document, document.ports, self.target(1))
        self.assertFalse(self.target(1).exists())

    def test_diff_channel_metadata_retained_for_default_names_and_removed_for_edits(self):
        block = (b"!.DiffChannels\r\n"
                 b'!"Pair" "A" "B" "PKG::A" "PKG::B" "0" "1"\r\n'
                 b"!.EndDiffChannels\r\n")
        source = self.source(2, block + b"! Port 1 = PKG-A1 A\n! Port[2] = B\n#\n" + matrix_bytes(2))
        document = read_touchstone(source)
        target = self.target(2)
        export_touchstone(document, document.ports, target)
        self.assertTrue(target.read_bytes().startswith(block))
        export_touchstone(document, [replace(document.ports[0], name="Changed"), document.ports[1]], target)
        self.assertNotIn(b".DiffChannels", target.read_bytes())
        self.assertNotIn(b"PKG::", target.read_bytes())
        export_touchstone(document, list(reversed(document.ports)), target)
        self.assertNotIn(b".DiffChannels", target.read_bytes())
        self.assertNotIn(b"PKG::", target.read_bytes())
        self.assertEqual([port.name for port in read_touchstone(target).ports], ["B", "A"])

    def test_cancellation_and_failed_replace_are_atomic(self):
        source = self.source(2, b"#\n" + matrix_bytes(2))
        target = self.target(2)
        target.write_bytes(b"existing output")
        document = read_touchstone(source)
        progress = []
        with self.assertRaises(CancelledError):
            read_touchstone(source, cancel=lambda: True)
        with self.assertRaises(CancelledError):
            export_touchstone(document, document.ports, target, progress.append, lambda: 99 in progress)
        self.assertEqual(progress, [0, 99])
        with patch("touchstone_editor.core.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                export_touchstone(document, document.ports, target)
        self.assertEqual(target.read_bytes(), b"existing output")
        self.assertEqual(set(self.directory.iterdir()), {source, target})

    def test_hfss_impedance_comments_preserved_or_reordered(self):
        data = matrix_bytes(2) + b"! Port Impedance 45 0 75 0\n! Gamma 1 0 2 0\n"
        source = self.source(2, b"# Hz S RI R 50\n" + data)
        document = read_touchstone(source)
        target = self.target(2)
        export_touchstone(document, document.ports, target)
        self.assertTrue(target.read_bytes().endswith(data))
        export_touchstone(document, list(reversed(document.ports)), target)
        self.assertTrue(target.read_bytes().endswith(b"! Port Impedance 75 0 45 0\n! Gamma 2 0 1 0\n"))

    def test_hfss_vectors_follow_matrix_order_and_frequency(self):
        for count, order in ((2, [1, 0]), (3, [2, 0, 1])):
            with self.subTest(count=count):
                data = bytearray()
                expected_blocks = []
                original_points = []
                for frequency in (b"1e+0", b"2.00"):
                    matrix = matrix_bytes(count, frequency).replace(b"\n", b"\r\n")
                    original_points.append(matrix.split())
                    if frequency == b"2.00":
                        matrix = matrix.replace(b"\r\n", b" ! 2026\r\n", 1)
                    data.extend(matrix)
                    block = []
                    for label in (b"! Gamma", b"! Port Impedance"):
                        pairs = [b"+" + frequency + b" -." + str(i + 1).encode() + b"E-3" for i in range(count)]
                        data.extend(label + b"\t" + b"\t".join(pairs) + b"\r\n")
                        block.append(label + b" " + b" ".join(pairs[i] for i in order))
                    expected_blocks.append(block)
                source = self.source(count, b"# GHz S MA R 50\r\n" + data)
                document = read_touchstone(source)
                target = self.target(count)
                export_touchstone(document, document.ports, target)
                self.assertTrue(target.read_bytes().endswith(data))
                export_touchstone(document, [document.ports[i] for i in order], target)
                lines = target.read_bytes().splitlines()[count + 1:]
                for point_number in range(2):
                    row_count = 1 if count == 2 else count
                    offset = point_number * (row_count + 2)
                    actual = b" ".join(lines[offset:offset + row_count]).split()
                    original = original_points[point_number]
                    axes = [(r, c) for c in order for r in order] if count == 2 else [(r, c) for r in order for c in order]
                    expected = [original[0]]
                    for row, column in axes:
                        cell = column * count + row if count == 2 else row * count + column
                        expected.extend(original[1 + 2 * cell:3 + 2 * cell])
                    self.assertEqual(actual, expected)
                    self.assertEqual(lines[offset + row_count:offset + row_count + 2], expected_blocks[point_number])
                self.assertEqual(document.point_count, 2)

    def test_hfss_unsupported_metadata_rejected_without_replacing_output(self):
        matrix = matrix_bytes(3)
        valid = b"! Port Impedance 50 0 60 0 70 0\n"
        cases = [
            valid + matrix,
            matrix + b"! Gamma 1 0 2 0\n",
            matrix + b"! Gamma 1 0 NaN 0 3 0\n",
            matrix + valid + b"! 1 0 2 0 3 0\n",
            matrix + valid + b"! 1 0 Inf 0 3 0\n",
            matrix + valid + b"! NaN 0 2 0 3 0\n",
            matrix + b"! Port Impedance " + b"50 0 " * 9 + b"\n",
            matrix.replace(b"\n", b"\n" + valid, 1),
            matrix.rstrip(b"\n") + b" " + valid,
        ]
        for body in cases:
            with self.subTest(body=body):
                document = read_touchstone(self.source(3, b"#\n" + body))
                target = self.target(3)
                target.write_bytes(b"existing output")
                with self.assertRaisesRegex(ValueError, "HFSS"):
                    export_touchstone(document, list(reversed(document.ports)), target)
                self.assertEqual(target.read_bytes(), b"existing output")
                self.assertEqual(list(self.directory.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
