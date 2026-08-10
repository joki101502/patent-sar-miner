"""Compound-number cell OCR (PRD §8.2, R8.4, EC-4).

A Table 1 number cell is a single small glyph adrift in a tall ruled box: on the
reference patent's page 63 the cell is 190x510 px and the digit occupies about
40x49 of it. Handing that region straight to tesseract returns punctuation noise
(`.`, `:`) at every page-segmentation mode, because the ruling lines dominate the
region and the glyph is a rounding error within it.

The fix is to inset past the rulings, crop to the ink, and upscale before OCR.
Getting this wrong is expensive: the compound number is the PRIMARY join key
(PRD R11.1), and an unread number costs a whole SAR row.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sarmine.ocr.tesseract import ocr_number_cell

REPO = Path(__file__).resolve().parent.parent
REFERENCE_PDF = REPO / "data" / "patents" / "WO2024097932A1.pdf"
NEEDS_PDF = pytest.mark.skipif(not REFERENCE_PDF.is_file(), reason="reference PDF absent")


def synthetic_number_cell(tmp_path: Path, digits: str, *, size=(190, 510)) -> Path:
    """A ruled cell with one small centred number, mimicking the real geometry."""
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=0, width=4)
    draw.text((size[0] // 2 - 6 * len(digits), size[1] // 2 - 8), digits, fill=0)
    path = tmp_path / f"cell_{digits}.png"
    img.save(path)
    return path


def test_reads_a_small_digit_adrift_in_a_ruled_cell(tmp_path):
    assert ocr_number_cell(synthetic_number_cell(tmp_path, "5"), work_dir=tmp_path) == 5


def test_reads_a_two_digit_number(tmp_path):
    assert ocr_number_cell(synthetic_number_cell(tmp_path, "47"), work_dir=tmp_path) == 47


def test_returns_none_on_an_empty_cell_rather_than_guessing(tmp_path):
    """PRD R8.4 / EC-4 — an invented compound number silently corrupts the join."""
    img = Image.new("L", (190, 510), color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 189, 509], outline=0, width=4)
    path = tmp_path / "blank.png"
    img.save(path)

    assert ocr_number_cell(path, work_dir=tmp_path) is None


def test_ruling_lines_alone_are_not_read_as_a_number(tmp_path):
    """The borders are the densest ink in the region; they must not become a digit."""
    img = Image.new("L", (190, 510), color=255)
    ImageDraw.Draw(img).rectangle([0, 0, 189, 509], outline=0, width=8)
    path = tmp_path / "rulings_only.png"
    img.save(path)

    assert ocr_number_cell(path, work_dir=tmp_path) is None


@pytest.mark.slow
@NEEDS_PDF
def test_reads_compounds_5_and_6_from_the_real_page_63(tmp_path):
    """Ground truth: PRD Appendix B.2 places compound 5 in page 63's first data
    row, and the page carries compounds 5 and 6."""
    from sarmine.segment.rulings import detect_grid

    subprocess.run(
        ["pdftoppm", "-r", "200", "-gray", "-f", "63", "-l", "63",
         str(REFERENCE_PDF), str(tmp_path / "r")],
        check=True, capture_output=True,
    )
    raw = sorted(tmp_path.glob("r-*"))[0]
    page = tmp_path / "p63.png"
    Image.open(raw).rotate(-90, expand=True).save(page)

    grid = detect_grid(page)
    numbers = [
        ocr_number_cell(page, cell.bbox, work_dir=tmp_path)
        for cell in sorted(grid.cells, key=lambda c: c.row)
        if cell.col == 0
    ]

    assert numbers == [5, 6]
