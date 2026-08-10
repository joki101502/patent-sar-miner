"""A blank activity cell stays blank (PRD EC-7, R12.9, AC-6.2).

Measured on page 187 of the reference patent, where compounds 33-38 have no HbF
value: the crop contains **exactly zero** dark pixels, and Tesseract returns
`Be` for it anyway. Every one of the eleven blank HbF cells hallucinated the
same letter, which is worse than a wrong value — it is a fabricated measurement
attached to a real compound, and it would rank that compound as if it had been
tested.

Ink coverage separates the two cases cleanly: 0.00000 for a blank cell against
0.008-0.012 for a cell with a letter in it. So the pixels decide, not the OCR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sarmine.pipeline import BLANK_CELL_MIN_DARK_PIXELS, cell_is_blank


def _blank(tmp_path: Path) -> Path:
    path = tmp_path / "blank.png"
    Image.new("L", (200, 120), 255).save(path)
    return path


def _with_letter(tmp_path: Path) -> Path:
    path = tmp_path / "letter.png"
    img = Image.new("L", (200, 120), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 30, 140, 90), outline=0, width=6)
    img.save(path)
    return path


def test_an_empty_crop_is_blank(tmp_path):
    assert cell_is_blank(_blank(tmp_path), (0, 0, 200, 120)) is True


def test_a_crop_with_a_letter_is_not_blank(tmp_path):
    assert cell_is_blank(_with_letter(tmp_path), (0, 0, 200, 120)) is False


def test_the_threshold_separates_the_measured_populations():
    """Measured dark-pixel counts: blank cells 0; real compound-number cells 148-450.

    An *absolute* floor is the right shape for this. As a fraction of its cell a
    compound number is only 0.0007-0.0019 — below any fraction threshold that
    also excludes a blank activity cell — because the number column is narrow
    and tall with one small glyph in it.
    """
    assert 0 < BLANK_CELL_MIN_DARK_PIXELS < 148


@pytest.mark.parametrize(
    "dark_pixels, ocr_reading",
    [(148, "1"), (194, "5"), (296, "11"), (392, "47"), (450, "49")],
)
def test_real_compound_number_cells_are_never_blank(tmp_path, dark_pixels, ocr_reading):
    """These exact cells were measured on the reference patent and were being
    discarded as blank, which is why compounds 11, 47 and 51 went missing."""
    path = tmp_path / f"num_{ocr_reading}.png"
    img = Image.new("L", (285, 800), 255)
    # A block of the measured area, in a cell of the measured size.
    side = int(dark_pixels**0.5)
    ImageDraw.Draw(img).rectangle((100, 380, 100 + side, 380 + side), fill=0)
    img.save(path)

    assert cell_is_blank(path, (0, 0, 285, 800)) is False


def test_a_zero_area_bbox_is_treated_as_blank(tmp_path):
    assert cell_is_blank(_blank(tmp_path), (10, 10, 10, 10)) is True


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("data/patents/WO2024097932A1.pdf").is_file(), reason="reference PDF absent"
)
def test_the_real_blank_hbf_cells_are_detected_as_blank(tmp_path):
    """The measured case: page 187, compounds 33-38 have no HbF value."""
    from sarmine.segment.rulings import detect_grid
    from sarmine.sources.pdf import extract_page_images

    page = extract_page_images(
        Path("data/patents/WO2024097932A1.pdf"), first=187, last=187, out_dir=tmp_path
    )[0].path
    grid = detect_grid(page)

    # Row 0 is compound 32 (HbF = A); rows 1-6 are compounds 33-38 (all blank).
    assert cell_is_blank(page, grid.cell(0, 1).bbox) is False
    for row in range(1, 7):
        assert cell_is_blank(page, grid.cell(row, 1).bbox) is True
    # The WIZ column on those same rows does carry values.
    for row in range(1, 7):
        assert cell_is_blank(page, grid.cell(row, 2).bbox) is False
