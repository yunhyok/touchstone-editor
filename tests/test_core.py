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
        self.assertEqual(document.point_count, 1)
        target = self.target(2)
        export_touchstone(document, document.ports, target)
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
        source = self.source(1, b"# ma\r1 .5 -12\r\n2 1 0\n! Port[1] = Late name")
        document = read_touchstone(source)
        self.assertEqual(document.option_line, "# ma")
        self.assertEqual(document.ports[0].name, "Late name")
        self.assertEqual(document.point_count, 2)
        target = self.target(1)
        export_touchstone(document, document.ports, target)
        self.assertEqual(target.read_bytes(), b"# ma\r! Port[1] = Late name\r\n1 .5 -12\r\n2 1 0\n")

    def test_rejects_unsupported_or_malformed_sources(self):
        cases = [
            (1, b"[Version] 2.0\n#\n1 0 0\n", "Touchstone 2.x"),
            (1, b"1 0 0\n", "expected an option line"),
            (1, b"# GHz Z RI R 50\n1 0 0\n", "option line"),
            (1, b"# S R -1\n1 0 0\n", "option line"),
            (1, b"# R 1 2\n1 0 0\n", "option line"),
            (1, b"# MHz\n1 0 0\n# GHz\n2 0 0\n", "multiple option lines"),
            (1, b"#\n", "no S-parameter data"),
            (1, b"#\n1 NaN 0\n", "invalid numeric"),
            (1, b"#\n1 2e- 0\n", "invalid numeric"),
            (1, b"#\n+1 1 2\n1.0 3 4\n", "strictly increasing"),
            (1, b"! Port[2] = Other\n#\n1 0 0\n", "out-of-range"),
            (1, b"! Port 1 = \n#\n1 0 0\n", "missing"),
            (1, b"! Port 1 = First\n! Port[1] = Second\n#\n1 0 0\n", "duplicate"),
            (2, b"#\n" + matrix_bytes(2) + b"1 2 3 4 5\n", "noise data"),
            (3, b"#\n1 1 0 2 0 3 0\n", "incomplete S-parameter matrix"),
            (3, b"#\n1 1 0 2 0\n", "incorrect number"),
            (5, b"#\n1 1 0 2 0 3 0 4 0\n5 0 6 0\n", "incorrect number"),
        ]
        for count, contents, error in cases:
            with self.subTest(contents=contents):
                with self.assertRaisesRegex(ValueError, error):
                    read_touchstone(self.source(count, contents))

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
            if percent == 0:
                source.write_bytes(contents.replace(b"+1.00", b"+2.00"))

        with self.assertRaisesRegex(ValueError, "source file changed"):
            export_touchstone(document, document.ports, target, progress=mutate)
        self.assertEqual(target.read_bytes(), b"existing output")
        self.assertEqual(set(self.directory.iterdir()), {source, target})

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

    def test_hfss_impedance_comments_preserved_or_reorder_rejected(self):
        data = matrix_bytes(2) + b"! Port Impedance 45 0 75 0\n! Gamma 1 0 2 0\n"
        source = self.source(2, b"# Hz S RI R 50\n" + data)
        document = read_touchstone(source)
        target = self.target(2)
        export_touchstone(document, document.ports, target)
        self.assertTrue(target.read_bytes().endswith(data))
        previous = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "HFSS"):
            export_touchstone(document, list(reversed(document.ports)), target)
        self.assertEqual(target.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
