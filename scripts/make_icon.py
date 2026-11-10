"""Render the app icon from its SVG source."""

from pathlib import Path

from PIL import Image
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    renderer = QSvgRenderer(str(assets / "icon.svg"))
    assert renderer.isValid(), "Invalid icon SVG"
    canvas = QImage(renderer.defaultSize(), QImage.Format.Format_ARGB32)
    canvas.fill(0)
    painter = QPainter(canvas)
    renderer.render(painter)
    painter.end()

    png = assets / "icon.png"
    if not canvas.save(str(png)):
        raise OSError(f"Could not write {png}")

    ico = assets / "icon.ico"
    sizes = [(size, size) for size in (16, 24, 32, 48, 64, 128, 256)]
    with Image.open(png) as image:
        image.save(ico, sizes=sizes)
    with Image.open(ico) as image:
        assert image.ico.sizes() == set(sizes), "Incomplete icon size set"


if __name__ == "__main__":
    main()
