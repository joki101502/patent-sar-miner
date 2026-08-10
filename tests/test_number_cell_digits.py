"""A compound number must not lose a digit (PRD R8.4, R11.1, EC-4).

`11` was read as `1` on page 66 of the reference patent. That is worse than an
unreadable cell: the number is the primary join key, so a dropped digit points
the row at a different compound's activity data, and it took compound 11 out of
the run entirely.

The fixtures are the real cells, cropped from the page the failure was measured
on: `number_11.png` is the one that failed, `number_12.png` its neighbour that
always read correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sarmine.ocr.tesseract import ocr_number_cell

CELLS = Path(__file__).parent / "fixtures" / "cells"


@pytest.mark.parametrize("expected", [11, 12])
def test_real_compound_number_cells_read_every_digit(expected):
    path = CELLS / f"number_{expected}.png"
    if not path.is_file():
        pytest.skip(f"fixture {path.name} missing")

    assert ocr_number_cell(path) == expected
