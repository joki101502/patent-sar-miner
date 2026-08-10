"""Part 4 — document segmentation (PRD §8, Plan Part 4).

The measured justification for every test in this file is PRD §8.1: whole-page
OCR of Table 1 made OPSIN parse 0 of 61 names, because atom labels from the
structure drawing interleave into the name text; segmenting the name cell first
made it parse 33 of 37. `test_ac_2_3_*` below reproduces that contrast directly.

Two families of real fixtures are used:
  * `imgf000*.png`   — the Google Patents images (8-bit grayscale, PRD §3.2)
  * `pages/p-*.png`  — the same pages rendered from the PDF at 300 dpi
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
PAGES = FIXTURES / "pages"

# Table 1's pages are stored rotated 90° (PRD §3.3). Determined empirically:
# a clockwise quarter turn OCRs to readable chemical text, the other direction
# to character-reversed gibberish. ROTATE_270 is the lossless clockwise turn;
# `Image.rotate(-90, expand=True)` is the same rotation.
CLOCKWISE = Image.Transpose.ROTATE_270

# PRD R8.2 / §8.3 — spike S2's recorded rulings for `rot/p-063.png`, which was
# the reference PDF rendered at 200 dpi. The committed page fixtures are 300 dpi
# (1.5x), and the Google image is a tighter crop again, so only the STRUCTURE
# and the relative proportions transfer.
SPIKE_Y_RULINGS_200DPI = [208, 718, 1439]
SPIKE_X_RULINGS_200DPI = [157, 347, 1397, 2012]
FIXTURE_DPI_SCALE = 1.5


# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------


def _derotate(src: Path, dest: Path) -> Path:
    with Image.open(src) as im:
        im.convert("L").transpose(CLOCKWISE).save(dest)
    return dest


@pytest.fixture(scope="session")
def derotated_page_63(tmp_path_factory) -> Path:
    """De-rotated `imgf000063_0001.png` — Table 1, compounds 5 and 6."""
    out = tmp_path_factory.mktemp("derot63") / "p-063-derotated.png"
    return _derotate(FIXTURES / "imgf000063_0001.png", out)


@pytest.fixture(scope="session")
def derotated_page_62(tmp_path_factory) -> Path:
    """De-rotated `imgf000062_0001.png` — Table 1, compounds 3 and 4."""
    out = tmp_path_factory.mktemp("derot62") / "p-062-derotated.png"
    return _derotate(FIXTURES / "imgf000062_0001.png", out)


@pytest.fixture(scope="session")
def derotated_pdf_page_63(tmp_path_factory) -> Path:
    """The same page rendered from the PDF at 300 dpi, de-rotated."""
    out = tmp_path_factory.mktemp("derot_pdf63") / "p-063-000-derotated.png"
    return _derotate(PAGES / "p-063-000.png", out)


@pytest.fixture(scope="session")
def page_186() -> Path:
    return FIXTURES / "imgf000186_0001.png"


@pytest.fixture(scope="session")
def page_187() -> Path:
    return FIXTURES / "imgf000187_0001.png"


def _gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def _fractions(rulings: list[int]) -> list[float]:
    spans = [b - a for a, b in zip(rulings, rulings[1:])]
    total = sum(spans)
    return [s / total for s in spans]


# --------------------------------------------------------------------------
# rulings.py — morphological grid detection (PRD R8.2)
# --------------------------------------------------------------------------


def test_find_rulings_recovers_two_rows_by_three_columns(derotated_page_63):
    """PRD §8.3: the de-rotated Table 1 page is a 2×3 grid.

    The PRD's recorded values came from a differently-scaled render, so assert
    the structure, not the pixels.
    """
    from sarmine.segment.rulings import find_rulings

    y_rulings, x_rulings = find_rulings(_gray(derotated_page_63))

    assert len(y_rulings) == 3, f"expected 3 horizontal rulings, got {y_rulings}"
    assert len(x_rulings) == 4, f"expected 4 vertical rulings, got {x_rulings}"
    assert y_rulings == sorted(y_rulings)
    assert x_rulings == sorted(x_rulings)


def test_find_rulings_proportions_match_the_prd_measurement(derotated_page_63):
    """Relative geometry must match PRD §8.3's `rot/p-063.png` measurement.

    PRD x=[157,347,1397,2012] -> column width fractions 0.102 / 0.566 / 0.332.
    PRD y=[208,718,1439]      -> row height fractions   0.414 / 0.586.
    """
    from sarmine.segment.rulings import find_rulings

    y_rulings, x_rulings = find_rulings(_gray(derotated_page_63))

    assert _fractions(x_rulings) == pytest.approx(_fractions(SPIKE_X_RULINGS_200DPI), abs=0.02)
    assert _fractions(y_rulings) == pytest.approx(_fractions(SPIKE_Y_RULINGS_200DPI), abs=0.02)
    assert _fractions(x_rulings) == pytest.approx([0.102, 0.566, 0.332], abs=0.02)
    assert _fractions(y_rulings) == pytest.approx([0.414, 0.586], abs=0.02)


def test_find_rulings_on_the_pdf_render_scales_with_the_spike_measurement(derotated_pdf_page_63):
    """The 300 dpi page fixture is 1.5x the raster the spike measured on."""
    from sarmine.segment.rulings import find_rulings

    y_rulings, x_rulings = find_rulings(_gray(derotated_pdf_page_63))

    assert y_rulings == [312, 1077, 2158]
    assert x_rulings == [237, 522, 2096, 3019]
    for got, spike in zip(y_rulings, SPIKE_Y_RULINGS_200DPI):
        assert abs(got - spike * FIXTURE_DPI_SCALE) <= 2
    for got, spike in zip(x_rulings, SPIKE_X_RULINGS_200DPI):
        assert abs(got - spike * FIXTURE_DPI_SCALE) <= 2


def test_find_rulings_on_a_blank_image_returns_nothing():
    from sarmine.segment.rulings import find_rulings

    assert find_rulings(np.full((400, 400), 255, dtype=np.uint8)) == ([], [])


def test_find_rulings_accepts_a_colour_image():
    from sarmine.segment.rulings import find_rulings

    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
    canvas[50:53, 20:280] = 0
    canvas[240:243, 20:280] = 0
    canvas[50:243, 20:23] = 0
    canvas[50:243, 277:280] = 0

    y_rulings, x_rulings = find_rulings(canvas)

    assert len(y_rulings) == 2
    assert len(x_rulings) == 2


def test_build_cells_addresses_every_row_and_column():
    from sarmine.segment.rulings import build_cells

    cells = build_cells([0, 10, 20], [0, 5, 15, 30])

    assert len(cells) == 6
    assert {(c.row, c.col) for c in cells} == {(r, c) for r in range(2) for c in range(3)}
    by_address = {(c.row, c.col): c.bbox for c in cells}
    assert by_address[(0, 0)] == (0, 0, 5, 10)
    assert by_address[(1, 2)] == (15, 10, 30, 20)


def test_build_cells_needs_two_rulings_on_each_axis():
    from sarmine.segment.rulings import build_cells

    assert build_cells([5], [0, 10]) == []
    assert build_cells([0, 10], []) == []


def test_detect_grid_builds_addressable_cells(derotated_page_63):
    from sarmine.segment.rulings import Grid, detect_grid, find_rulings

    grid = detect_grid(derotated_page_63)

    assert isinstance(grid, Grid)
    assert (grid.n_rows, grid.n_cols) == (2, 3)
    assert len(grid.cells) == 6
    assert grid.detector == "morphology"

    with Image.open(derotated_page_63) as im:
        assert (grid.width, grid.height) == im.size

    y_rulings, x_rulings = find_rulings(_gray(derotated_page_63))
    assert (grid.y_rulings, grid.x_rulings) == (y_rulings, x_rulings)

    top_left = grid.cell(0, 0)
    assert top_left is not None
    assert top_left.bbox == (x_rulings[0], y_rulings[0], x_rulings[1], y_rulings[1])
    assert grid.cell(9, 9) is None


def test_detect_grid_finds_the_activity_table_rows(page_186, page_187):
    """Table 2: page 186 is header + 31 compounds, page 187 is compounds 32-54."""
    from sarmine.segment.rulings import detect_grid

    g186 = detect_grid(page_186)
    g187 = detect_grid(page_187)

    assert (g186.n_rows, g186.n_cols) == (32, 4)
    assert (g187.n_rows, g187.n_cols) == (23, 4)
    assert len(g186.cells) == 32 * 4
    assert len(g187.cells) == 23 * 4


def test_detect_grid_on_a_page_without_rulings_yields_an_empty_grid(tmp_path):
    from sarmine.segment.rulings import detect_grid

    blank = tmp_path / "blank.png"
    Image.new("L", (600, 800), 255).save(blank)

    grid = detect_grid(blank)

    assert (grid.n_rows, grid.n_cols) == (0, 0)
    assert grid.cells == []
    assert (grid.width, grid.height) == (600, 800)


def test_detect_grid_raises_on_a_missing_file(tmp_path):
    from sarmine.segment.rulings import detect_grid

    with pytest.raises(FileNotFoundError):
        detect_grid(tmp_path / "nope.png")


# --------------------------------------------------------------------------
# column roles (Plan 4.2) — geometry-driven, never hard-coded indices
# --------------------------------------------------------------------------


def test_assign_column_roles_on_the_real_table_1_grid(derotated_page_63):
    from sarmine.segment.rulings import assign_column_roles, detect_grid

    grid = detect_grid(derotated_page_63)
    widths = {c: grid.x_rulings[c + 1] - grid.x_rulings[c] for c in range(grid.n_cols)}

    ocr_by_col = {
        0: "5\n6",
        1: "O N HN NH F O O",
        2: "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-"
        "1H-benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione",
    }
    roles = assign_column_roles(grid, ocr_by_col)

    assert sorted(roles.values()) == ["name", "number", "structure"]
    inverse = {role: col for col, role in roles.items()}
    assert widths[inverse["structure"]] == max(widths.values())
    assert widths[inverse["number"]] == min(widths.values())
    assert inverse["name"] == 2


def test_assign_column_roles_does_not_hard_code_column_order():
    """PRD Plan 4.2: column order varies by filer."""
    from sarmine.segment.rulings import Grid, assign_column_roles, build_cells

    # name | structure | number — the reverse of the reference patent's layout
    x_rulings = [0, 600, 1600, 1700]
    y_rulings = [0, 100]
    grid = Grid(
        n_rows=1,
        n_cols=3,
        cells=build_cells(y_rulings, x_rulings),
        y_rulings=y_rulings,
        x_rulings=x_rulings,
        width=1700,
        height=100,
        detector="morphology",
    )
    roles = assign_column_roles(grid, {0: "isoindoline dione name text", 1: "N O", 2: "12"})

    assert roles == {0: "name", 1: "structure", 2: "number"}


# --------------------------------------------------------------------------
# compound numbers (PRD R8.4 / EC-4) — flag, never interpolate
# --------------------------------------------------------------------------


def test_parse_compound_numbers_reads_digits_and_reports_failures():
    from sarmine.segment.rulings import parse_compound_numbers

    assert parse_compound_numbers(["1", " 2 ", "3\n"]) == [1, 2, 3]
    assert parse_compound_numbers(["", "  ", "~"]) == [None, None, None]
    assert parse_compound_numbers(["l2", "O7"]) == [12, 7]


def test_validate_monotonic_clean_run_has_no_anomalies():
    from sarmine.segment.rulings import validate_monotonic

    assert validate_monotonic([1, 2, 3]) == []
    assert validate_monotonic([]) == []
    assert validate_monotonic([7]) == []


def test_validate_monotonic_flags_a_gap_and_never_interpolates():
    from sarmine.segment.rulings import validate_monotonic

    numbers = [1, 2, 4, 5]
    anomalies = validate_monotonic(numbers)

    assert len(anomalies) == 1
    assert anomalies[0].kind == "compound_number_gap"
    assert anomalies[0].severity == "warning"
    assert "3" in anomalies[0].message
    # EC-4: an invented compound number silently corrupts the join. The function
    # returns anomalies only — it never hands back a repaired sequence.
    assert numbers == [1, 2, 4, 5]
    assert all(type(a).__name__ == "DocumentAnomaly" for a in anomalies)


def test_validate_monotonic_flags_an_unreadable_entry():
    from sarmine.segment.rulings import validate_monotonic

    anomalies = validate_monotonic([1, 2, None, 4])

    assert [a.kind for a in anomalies] == ["compound_number_gap"]
    assert "unreadable" in anomalies[0].message.lower()
    assert len(validate_monotonic([None])) == 1


def test_validate_monotonic_flags_non_monotonicity():
    from sarmine.segment.rulings import validate_monotonic

    anomalies = validate_monotonic([1, 5, 3])

    assert any("monotonic" in a.message.lower() for a in anomalies)
    assert all(a.kind == "compound_number_gap" for a in anomalies)


def test_validate_monotonic_flags_a_duplicate():
    from sarmine.segment.rulings import validate_monotonic

    anomalies = validate_monotonic([1, 2, 2, 3])

    assert any("duplicate" in a.message.lower() for a in anomalies)


# --------------------------------------------------------------------------
# tatr.py — the second detector (PRD R8.3)
# --------------------------------------------------------------------------


def _transformers_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("transformers") is not None


def test_tatr_module_is_import_safe():
    """No model loading at import time (PRD §17.5)."""
    import sarmine.segment.tatr as tatr

    assert callable(tatr.is_available)
    assert callable(tatr.detect_grid_tatr)


def test_tatr_availability_tracks_the_transformers_install():
    from sarmine.segment import tatr

    assert tatr.is_available() == _transformers_installed()


@pytest.mark.skipif(_transformers_installed(), reason="transformers IS installed")
def test_tatr_returns_none_cleanly_when_unavailable(derotated_page_63):
    """PRD R8.3: the second detector degrades to `None`, it never raises."""
    from sarmine.segment import tatr

    assert tatr.is_available() is False
    assert tatr.detect_grid_tatr(derotated_page_63) is None


def test_tatr_returns_none_rather_than_raising_on_a_missing_file(tmp_path):
    from sarmine.segment import tatr

    assert tatr.detect_grid_tatr(tmp_path / "does-not-exist.png") is None


def test_tatr_grid_from_boxes_intersects_rows_and_columns():
    """The row×column intersection that turns TATR's predictions into cells."""
    from sarmine.segment.tatr import grid_from_boxes

    rows = [(0.0, 0.0, 300.0, 100.0), (0.0, 100.0, 300.0, 200.0)]
    columns = [(0.0, 0.0, 120.0, 200.0), (120.0, 0.0, 300.0, 200.0)]

    grid = grid_from_boxes(rows, columns, width=300, height=200)

    assert (grid.n_rows, grid.n_cols) == (2, 2)
    assert grid.detector == "tatr"
    assert len(grid.cells) == 4
    assert grid.cell(0, 0).bbox == (0, 0, 120, 100)
    assert grid.cell(1, 1).bbox == (120, 100, 300, 200)


# --------------------------------------------------------------------------
# reconcile.py — two detectors, one grid (PRD R8.3, EC-26)
# --------------------------------------------------------------------------


def _grid(y_rulings, x_rulings, detector="morphology", width=1000, height=1000):
    from sarmine.segment.rulings import Grid, build_cells

    return Grid(
        n_rows=max(len(y_rulings) - 1, 0),
        n_cols=max(len(x_rulings) - 1, 0),
        cells=build_cells(y_rulings, x_rulings),
        y_rulings=list(y_rulings),
        x_rulings=list(x_rulings),
        width=width,
        height=height,
        detector=detector,
    )


def test_cell_iou_identical_boxes():
    from sarmine.segment.reconcile import cell_iou

    assert cell_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_cell_iou_disjoint_boxes():
    from sarmine.segment.reconcile import cell_iou

    assert cell_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert cell_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0  # touching, not overlapping


def test_cell_iou_partial_overlap():
    from sarmine.segment.reconcile import cell_iou

    # 50% overlap in x, full in y: intersection 50, union 150.
    assert cell_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_cell_iou_zero_area_box():
    from sarmine.segment.reconcile import cell_iou

    assert cell_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_reconcile_agreeing_detectors_report_no_disagreement():
    from sarmine.segment.reconcile import reconcile

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")
    tatr = _grid([1, 101, 201], [0, 51, 149], detector="tatr")

    result = reconcile(morph, tatr)

    assert result.disagreement is False
    assert result.anomalies == []
    assert (result.grid.n_rows, result.grid.n_cols) == (2, 2)
    assert result.grid.detector == "reconciled"


def test_reconcile_disagreement_flags_a_detector_disagreement_anomaly():
    """EC-26: disagreement is a review-queue trigger, and the more complete grid
    wins."""
    from sarmine.segment.reconcile import reconcile

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")  # 2×2 = 4 cells
    tatr = _grid([0, 50, 100, 150, 200], [0, 50, 150], detector="tatr")  # 4×2 = 8 cells

    result = reconcile(morph, tatr)

    assert result.disagreement is True
    assert "detector_disagreement" in [a.kind for a in result.anomalies]
    assert all(a.severity == "warning" for a in result.anomalies)
    # EC-26 — prefer the detector with the more complete grid.
    assert result.grid.n_rows == 4
    assert result.notes


def test_reconcile_prefers_the_grid_closest_to_the_expected_row_count():
    from sarmine.segment.reconcile import reconcile

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")  # 2 rows
    tatr = _grid([0, 20, 40, 60, 80, 100, 200], [0, 50, 150], detector="tatr")  # 6 rows

    result = reconcile(morph, tatr, expected_rows=2)

    assert result.grid.n_rows == 2
    assert result.disagreement is True


def test_reconcile_with_tatr_none_returns_the_morphology_grid():
    from sarmine.segment.reconcile import reconcile

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")

    result = reconcile(morph, None)

    assert result.grid is morph
    assert result.grid.detector == "morphology"
    assert result.disagreement is False
    assert result.anomalies == []


def test_reconcile_with_morphology_none_returns_the_tatr_grid():
    from sarmine.segment.reconcile import reconcile

    tatr = _grid([0, 100, 200], [0, 50, 150], detector="tatr")

    result = reconcile(None, tatr)

    assert result.grid is tatr
    assert result.disagreement is False


def test_reconcile_with_both_none_raises():
    from sarmine.segment.reconcile import reconcile

    with pytest.raises(ValueError):
        reconcile(None, None)


def test_reconcile_on_the_real_incomplete_table_1_page(derotated_page_62, derotated_page_63):
    """Page 62's top ruling is missing, so morphology finds 1 row where page 63
    finds 2 — the incomplete-grid case that motivates the second detector."""
    from sarmine.segment.rulings import detect_grid

    incomplete = detect_grid(derotated_page_62)
    complete = detect_grid(derotated_page_63)

    assert incomplete.n_rows == 1
    assert complete.n_rows == 2
    assert incomplete.completeness == 1.0  # the lattice is full; the page is not


# --------------------------------------------------------------------------
# stitch.py — multi-page tables (PRD R8.5, EC-3, AC-2.4)
# No surveyed tool does this; it is our own code.
# --------------------------------------------------------------------------

TABLE_2_HEADER = ["Compound No.", "HbF Induction (%)", "WIZ EC50 (uM)", "ZBTB7A EC50 (uM)"]

# PRD Appendix B.1 — the verified contents of Table 2. "" is a blank cell.
TABLE_2_GROUND_TRUTH: list[tuple[int, str, str, str]] = [
    (1, "A", "E", "G"), (2, "A", "D", "G"), (3, "A", "E", "G"), (4, "A", "E", "H"),
    (5, "A", "E", "H"), (6, "A", "D", "H"), (7, "A", "E", "H"), (8, "A", "E", "H"),
    (9, "A", "D", "H"), (10, "A", "D", "I"), (11, "A", "F", "I"), (12, "A", "D", "H"),
    (13, "A", "E", "H"), (14, "A", "F", "I"), (15, "A", "D", "H"), (16, "A", "D", "I"),
    (17, "A", "E", "H"), (18, "A", "D", "H"), (19, "A", "F", "I"), (20, "B", "D", "I"),
    (21, "B", "D", "H"), (22, "B", "E", "I"), (23, "B", "F", "I"), (24, "B", "F", "I"),
    (25, "B", "E", "I"), (26, "C", "E", "I"), (27, "C", "E", "I"), (28, "C", "F", "I"),
    (29, "C", "E", "I"), (30, "C", "E", "H"), (31, "B", "E", "I"), (32, "A", "D", "H"),
    (33, "", "E", "H"), (34, "", "E", "G"), (35, "", "E", "H"), (36, "", "E", "G"),
    (37, "", "D", "G"), (38, "", "E", "G"), (39, "B", "E", "H"), (40, "A", "D", "G"),
    (41, "A", "D", "G"), (42, "B", "F", "H"), (43, "A", "D", "G"), (44, "A", "E", "H"),
    (45, "A", "D", "G"), (46, "A", "D", "G"), (47, "A", "D", "G"), (48, "A", "D", "G"),
    (49, "", "D", "G"), (50, "", "E", "G"), (51, "", "E", "H"), (52, "A", "D", "I"),
    (53, "", "F", "I"), (54, "", "F", "I"),
]


def _row_texts(first: int, last: int) -> list[list[str]]:
    return [
        [str(number), hbf, wiz, zbtb]
        for number, hbf, wiz, zbtb in TABLE_2_GROUND_TRUTH
        if first <= number <= last
    ]


def test_looks_like_header_accepts_the_real_table_2_header():
    from sarmine.segment.stitch import looks_like_header

    assert looks_like_header(TABLE_2_HEADER) is True


def test_looks_like_header_rejects_a_data_row():
    from sarmine.segment.stitch import looks_like_header

    # EC-3: page 187 opens directly at compound 32 with no header.
    assert looks_like_header(["32", "A", "D", "H"]) is False
    assert looks_like_header(["33", "", "E", "H"]) is False


def test_looks_like_header_rejects_empty_input():
    from sarmine.segment.stitch import looks_like_header

    assert looks_like_header([]) is False
    assert looks_like_header(["", "  "]) is False


def test_looks_like_header_accepts_a_split_header_fragment():
    """PRD EC-27 — the second reference patent wraps its header."""
    from sarmine.segment.stitch import looks_like_header

    assert looks_like_header(["Cmpd. No.", "Compound Structure", "HiBiT DC50 (nM)"]) is True


def test_columns_match_on_the_real_pages_186_and_187(page_186, page_187):
    """AC-2.4 exercises `columns_match` against genuinely detected grids."""
    from sarmine.segment.rulings import detect_grid
    from sarmine.segment.stitch import columns_match

    assert columns_match(detect_grid(page_186), detect_grid(page_187)) is True


def test_columns_match_rejects_a_different_column_layout(page_186, derotated_page_63):
    from sarmine.segment.rulings import detect_grid
    from sarmine.segment.stitch import columns_match

    assert columns_match(detect_grid(page_186), detect_grid(derotated_page_63)) is False


def test_columns_match_is_scale_normalized():
    from sarmine.segment.stitch import columns_match

    a = _grid([0, 100], [0, 250, 500, 1000], width=1000, height=100)
    b = _grid([0, 50], [0, 125, 250, 500], width=500, height=50)

    assert columns_match(a, b) is True


def test_ac_2_4_pages_186_and_187_stitch_into_one_logical_table(page_186, page_187):
    """AC-2.4 / EC-3 — Table 2 spans two pages and page 187 has NO header row."""
    from sarmine.segment.rulings import detect_grid
    from sarmine.segment.stitch import TablePage, stitch_tables

    grid_186 = detect_grid(page_186)
    grid_187 = detect_grid(page_187)

    rows_186 = [TABLE_2_HEADER] + _row_texts(1, 31)
    rows_187 = _row_texts(32, 54)
    assert len(rows_186) == grid_186.n_rows
    assert len(rows_187) == grid_187.n_rows

    pages = [
        TablePage(page_no=186, grid=grid_186, header_texts=rows_186[0], rows=rows_186),
        TablePage(page_no=187, grid=grid_187, header_texts=rows_187[0], rows=rows_187),
    ]
    tables = stitch_tables(pages)

    assert len(tables) == 1
    table = tables[0]
    assert table.page_nos == [186, 187]
    assert table.header == TABLE_2_HEADER
    assert len(table.rows) == 54

    source_pages = [page_no for page_no, _ in table.rows]
    assert source_pages.count(186) == 31
    assert source_pages.count(187) == 23
    assert table.rows[0] == (186, ["1", "A", "E", "G"])
    assert table.rows[30] == (186, ["31", "B", "E", "I"])
    assert table.rows[31] == (187, ["32", "A", "D", "H"])
    assert table.rows[-1] == (187, ["54", "", "F", "I"])
    assert table.stitch_uncertain is False


def test_stitch_starts_a_new_table_when_the_next_page_has_its_own_header(page_186, page_187):
    from sarmine.segment.rulings import detect_grid
    from sarmine.segment.stitch import TablePage, stitch_tables

    pages = [
        TablePage(
            page_no=186,
            grid=detect_grid(page_186),
            header_texts=TABLE_2_HEADER,
            rows=[TABLE_2_HEADER] + _row_texts(1, 31),
        ),
        TablePage(
            page_no=187,
            grid=detect_grid(page_187),
            header_texts=TABLE_2_HEADER,
            rows=[TABLE_2_HEADER] + _row_texts(32, 53),
        ),
    ]
    tables = stitch_tables(pages)

    assert len(tables) == 2
    assert tables[0].page_nos == [186]
    assert tables[1].page_nos == [187]


def test_stitch_starts_a_new_table_when_the_columns_do_not_match():
    from sarmine.segment.stitch import TablePage, stitch_tables

    a = _grid([0, 50, 100], [0, 250, 500, 1000], width=1000, height=100)
    b = _grid([0, 50, 100], [0, 700, 800, 1000], width=1000, height=100)

    pages = [
        TablePage(page_no=1, grid=a, header_texts=["Compound No.", "IC50 (nM)", "Ki (nM)"],
                  rows=[["Compound No.", "IC50 (nM)", "Ki (nM)"], ["1", "5", "6"]]),
        TablePage(page_no=2, grid=b, header_texts=["2", "7", "8"],
                  rows=[["2", "7", "8"], ["3", "9", "10"]]),
    ]

    assert len(stitch_tables(pages)) == 2


def test_stitch_flags_a_marginal_geometry_match_as_uncertain():
    """A column match that only just clears tolerance is not a confident stitch."""
    from sarmine.segment.stitch import TablePage, stitch_tables

    a = _grid([0, 50, 100], [0, 250, 500, 1000], width=1000, height=100)
    b = _grid([0, 50, 100], [0, 290, 540, 1000], width=1000, height=100)

    pages = [
        TablePage(page_no=1, grid=a, header_texts=["Compound No.", "IC50 (nM)", "Ki (nM)"],
                  rows=[["Compound No.", "IC50 (nM)", "Ki (nM)"], ["1", "5", "6"]]),
        TablePage(page_no=2, grid=b, header_texts=["2", "7", "8"],
                  rows=[["2", "7", "8"], ["3", "9", "10"]]),
    ]
    tables = stitch_tables(pages)

    assert len(tables) == 1
    assert tables[0].stitch_uncertain is True
    assert [a.kind for a in tables[0].anomalies] == ["table_stitch_uncertain"]


def test_stitch_of_a_single_headerless_page_keeps_every_row():
    from sarmine.segment.stitch import TablePage, stitch_tables

    grid = _grid([0, 50, 100], [0, 250, 500, 1000], width=1000, height=100)
    pages = [
        TablePage(page_no=9, grid=grid, header_texts=["1", "A", "D"],
                  rows=[["1", "A", "D"], ["2", "B", "E"]])
    ]
    tables = stitch_tables(pages)

    assert len(tables) == 1
    assert tables[0].header == []
    assert tables[0].rows == [(9, ["1", "A", "D"]), (9, ["2", "B", "E"])]


def test_stitch_of_no_pages_returns_no_tables():
    from sarmine.segment.stitch import stitch_tables

    assert stitch_tables([]) == []


# --------------------------------------------------------------------------
# crops.py — crops to disk, paths in memory (PRD R17.3, §15.3, §15.5)
# --------------------------------------------------------------------------


def test_write_crop_produces_a_png_and_a_full_provenance(derotated_page_63, tmp_path):
    from sarmine.artifacts.schema import Provenance
    from sarmine.segment.crops import write_crop
    from sarmine.segment.rulings import assign_column_roles, detect_grid

    grid = detect_grid(derotated_page_63)
    roles = assign_column_roles(grid, {0: "5", 1: "N O", 2: "isoindoline dione"})
    name_col = next(col for col, role in roles.items() if role == "name")
    cell = grid.cell(0, name_col)

    crops_dir = tmp_path / "artifacts" / "run" / "crops"
    provenance = write_crop(
        derotated_page_63,
        cell.bbox,
        crops_dir,
        page_no=63,
        kind="name",
        idx=0,
        source="structured",
        extractor="tesseract@5.5.3",
        rotation_applied=90,
    )

    assert isinstance(provenance, Provenance)

    written = crops_dir / "p063_name_0.png"
    assert written.is_file()
    with Image.open(written) as crop:
        assert crop.format == "PNG"
        assert crop.size == (cell.bbox[2] - cell.bbox[0], cell.bbox[3] - cell.bbox[1])

    # PRD §15.5 — the path is relative to the bundle root, not absolute.
    assert provenance.crop_path == "crops/p063_name_0.png"
    assert not Path(provenance.crop_path).is_absolute()

    # PRD §15.3 — raster dimensions are the SOURCE page's, so the bbox rescales.
    with Image.open(derotated_page_63) as page:
        assert (provenance.raster_width, provenance.raster_height) == page.size

    assert provenance.page_no == 63
    assert provenance.bbox == cell.bbox
    assert provenance.source == "structured"
    assert provenance.extractor == "tesseract@5.5.3"
    assert provenance.rotation_applied == 90


def test_write_crop_pads_and_clamps_to_the_page(derotated_page_63, tmp_path):
    from sarmine.segment.crops import write_crop

    provenance = write_crop(
        derotated_page_63,
        (0, 0, 100, 100),
        tmp_path / "crops",
        page_no=63,
        kind="number",
        idx=7,
        source="structured",
        extractor="tesseract@5.5.3",
        pad=25,
    )

    assert provenance.bbox == (0, 0, 125, 125)
    assert (tmp_path / "crops" / "p063_number_7.png").is_file()


def test_write_crop_rejects_a_degenerate_bbox(derotated_page_63, tmp_path):
    from sarmine.segment.crops import write_crop

    with pytest.raises(ValueError):
        write_crop(
            derotated_page_63,
            (50, 50, 50, 90),
            tmp_path / "crops",
            page_no=63,
            kind="name",
            idx=0,
            source="structured",
            extractor="tesseract@5.5.3",
        )


# --------------------------------------------------------------------------
# AC-2.3 — the whole point of Part 4 (PRD §8.1)
# --------------------------------------------------------------------------

# Atom labels bleed out of the structure drawing as isolated all-caps tokens
# ("NN", "HN", "NH") alongside bond-line garbage ("re)fe)").
_STANDALONE_ATOM_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}(?![A-Za-z0-9])")


def _tesseract(image_path: Path, psm: str = "6") -> str:
    result = subprocess.run(
        ["tesseract", str(image_path), "-", "--psm", psm],
        capture_output=True,
        check=True,
    )
    # PRD Plan 5.1 — Tesseract emits non-UTF-8 bytes on some crops.
    return result.stdout.decode("utf-8", errors="replace")


def _alpha_ratio(text: str) -> float:
    meaningful = [ch for ch in text if not ch.isspace()]
    if not meaningful:
        return 0.0
    return sum(1 for ch in meaningful if ch.isalpha()) / len(meaningful)


@pytest.mark.slow
def test_ac_2_3_name_cell_ocr_carries_no_atom_labels(derotated_page_63, tmp_path):
    """AC-2.3 / PRD §8.1 — segmentation is what makes the name channel work.

    Whole-page OCR interleaves atom labels from the structure drawing into the
    name text and OPSIN parsed 0 of 61 names. OCR'ing only the name cell parsed
    33 of 37. The contrast below IS the test.
    """
    from sarmine.segment.crops import write_crop
    from sarmine.segment.rulings import assign_column_roles, detect_grid

    grid = detect_grid(derotated_page_63)
    assert (grid.n_rows, grid.n_cols) == (2, 3)

    crops_dir = tmp_path / "crops"
    column_text: dict[int, str] = {}
    for col in range(grid.n_cols):
        cell = grid.cell(0, col)
        prov = write_crop(
            derotated_page_63,
            cell.bbox,
            crops_dir,
            page_no=63,
            kind=f"col{col}",
            idx=0,
            source="structured",
            extractor="tesseract@5.5.3",
            rotation_applied=90,
        )
        column_text[col] = _tesseract(crops_dir.parent / prov.crop_path)

    roles = assign_column_roles(grid, column_text)
    name_col = next(col for col, role in roles.items() if role == "name")
    name_text = column_text[name_col]

    # --- the segmented read: a real, complete IUPAC name -------------------
    lowered = name_text.lower()
    assert "dioxopiperidin" in lowered or "isoindoline" in lowered
    assert len(name_text.strip()) > 60
    assert _alpha_ratio(name_text) > 0.5
    assert _STANDALONE_ATOM_TOKEN.search(name_text) is None, (
        f"atom labels bled into the name cell: {name_text!r}"
    )

    # --- the contrast: whole-page OCR of the same page ---------------------
    whole_page = _tesseract(derotated_page_63)
    assert "dioxopiperidin" in whole_page.lower()  # the name is in there...
    # ...but so are the drawing's atom labels, which is exactly why OPSIN
    # parsed 0/61 on this input (PRD §8.1).
    contamination = _STANDALONE_ATOM_TOKEN.findall(whole_page)
    assert contamination, (
        "expected whole-page OCR to bleed atom labels; if this ever stops being "
        "true the AC-2.3 contrast is no longer meaningful"
    )
    assert _alpha_ratio(name_text) > _alpha_ratio(whole_page)

    print("\n--- AC-2.3 segmented name cell ---\n" + name_text)
    print("--- AC-2.3 whole-page contamination tokens ---")
    print(sorted(set(contamination)))
    print("--- AC-2.3 whole-page OCR ---\n" + whole_page)


@pytest.mark.slow
def test_ac_2_3_holds_for_the_second_name_cell(derotated_page_63, tmp_path):
    """The second compound on the same page — one clean cell is not luck."""
    from sarmine.segment.crops import write_crop
    from sarmine.segment.rulings import detect_grid

    grid = detect_grid(derotated_page_63)
    cell = grid.cell(1, 2)
    prov = write_crop(
        derotated_page_63,
        cell.bbox,
        tmp_path / "crops",
        page_no=63,
        kind="name",
        idx=1,
        source="structured",
        extractor="tesseract@5.5.3",
        rotation_applied=90,
    )
    text = _tesseract(tmp_path / prov.crop_path)

    assert "dioxopiperidin" in text.lower()
    assert "isoindoline" in text.lower()
    assert _STANDALONE_ATOM_TOKEN.search(text) is None, text
    print("\n--- AC-2.3 second name cell ---\n" + text)


# --------------------------------------------------------------------------
# The spike's own raster: PDF page 63 at 200 dpi (PRD R8.2, Plan 4.1)
# --------------------------------------------------------------------------

REFERENCE_PDF = Path(__file__).parent.parent / "data" / "patents" / "WO2024097932A1.pdf"
HAVE_PDFTOPPM = shutil.which("pdftoppm") is not None
NEEDS_PDF = pytest.mark.skipif(
    not (HAVE_PDFTOPPM and REFERENCE_PDF.is_file()),
    reason="needs poppler and the reference PDF",
)


def _render_pdf_page(page_no: int, dest_dir: Path, dpi: int = 200) -> Path:
    """Render one page of the reference PDF and de-rotate it clockwise."""
    prefix = dest_dir / f"r{page_no}"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", "-f", str(page_no), "-l", str(page_no),
         str(REFERENCE_PDF), str(prefix)],
        check=True,
        capture_output=True,
    )
    raw = sorted(dest_dir.glob(f"r{page_no}-*"))[0]
    out = dest_dir / f"p{page_no:03d}-derotated.png"
    _derotate(raw, out)
    raw.unlink()
    return out


@pytest.mark.slow
@NEEDS_PDF
def test_find_rulings_reproduces_the_spike_measurement_exactly(tmp_path):
    """PRD R8.2 / Plan 4.1 — the recorded rulings for `rot/p-063.png`, to the pixel.

    Spike S2 rendered the PDF at 200 dpi and rotated it clockwise. Reproducing
    that raster reproduces its numbers exactly, which is what makes the
    committed 300 dpi fixture's 1.5x scaling above a check and not a fudge.
    """
    from sarmine.segment.rulings import find_rulings

    y_rulings, x_rulings = find_rulings(_gray(_render_pdf_page(63, tmp_path)))

    assert y_rulings == SPIKE_Y_RULINGS_200DPI
    assert x_rulings == SPIKE_X_RULINGS_200DPI


# --------------------------------------------------------------------------
# the names the pipeline imports (contract surface)
# --------------------------------------------------------------------------


def test_cells_from_rulings_addresses_every_row_and_column():
    from sarmine.segment.rulings import cells_from_rulings

    cells = cells_from_rulings([0, 10, 20], [0, 5, 15, 30])

    assert len(cells) == 6
    by_address = {(c.row, c.col): c.bbox for c in cells}
    assert by_address[(0, 0)] == (0, 0, 5, 10)
    assert by_address[(1, 2)] == (15, 10, 30, 20)
    assert cells_from_rulings([5], [0, 10]) == []


def test_grid_derives_its_shape_from_the_rulings():
    from sarmine.segment.rulings import Grid, cells_from_rulings

    y_rulings, x_rulings = [0, 100, 200], [0, 50, 150]
    grid = Grid(
        y_rulings=y_rulings,
        x_rulings=x_rulings,
        cells=cells_from_rulings(y_rulings, x_rulings),
        width=150,
        height=200,
    )

    assert (grid.n_rows, grid.n_cols) == (2, 2)
    assert grid.detector == "morphology"
    assert grid.completeness == 1.0


def test_validate_number_sequence_flags_a_gap_and_never_interpolates():
    """EC-4 / AC-5.4 under the name the pipeline imports."""
    from sarmine.segment.rulings import validate_number_sequence

    numbers = [1, 2, 4, 5]
    anomalies = validate_number_sequence(numbers)

    assert len(anomalies) == 1
    assert anomalies[0].kind == "compound_number_gap"
    assert numbers == [1, 2, 4, 5]
    assert validate_number_sequence([1, 2, 3]) == []


def test_crop_filename_follows_the_prd_bundle_layout():
    from sarmine.segment.crops import crop_filename

    assert crop_filename(63, "name", 0) == "p063_name_0.png"
    assert crop_filename(7, "structure", 12) == "p007_structure_12.png"


def test_tatr_model_id_is_the_pinned_prd_model():
    from sarmine.segment import tatr

    assert tatr.TATR_MODEL_ID == "microsoft/table-transformer-structure-recognition-v1.1-all"


def test_detect_grid_tatr_takes_a_detection_threshold(tmp_path):
    from sarmine.segment.tatr import detect_grid_tatr

    assert detect_grid_tatr(tmp_path / "missing.png", threshold=0.9) is None


def test_importing_tatr_pulls_in_neither_torch_nor_transformers():
    """PRD §17.5 / R17.3 — heavyweight imports stay inside the functions."""
    probe = (
        "import sys; import sarmine.segment.tatr; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False False"


@pytest.mark.slow
@pytest.mark.skipif(not _transformers_installed(), reason="transformers is not installed")
def test_tatr_detects_a_grid_on_a_real_page(page_186):
    """AC-2.2's second detector, exercised for real when the weights exist."""
    from sarmine.segment.tatr import detect_grid_tatr

    grid = detect_grid_tatr(page_186)

    assert grid is not None
    assert grid.detector == "tatr"
    assert grid.n_cols >= 3


def test_reconcile_result_carries_the_cells_and_the_detector_counts():
    from sarmine.segment.reconcile import reconcile

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")  # 4 cells
    tatr = _grid([0, 50, 100, 150, 200], [0, 50, 150], detector="tatr")  # 8 cells

    result = reconcile(morph, tatr)

    assert result.n_morph == 4
    assert result.n_tatr == 8
    assert result.cells == result.grid.cells
    assert len(result.cells) == 8  # EC-26 — the more complete grid wins
    assert "morphology" in result.detail and "tatr" in result.detail


def test_merge_row_boundaries_takes_rows_from_one_grid_and_columns_from_the_other():
    """PRD R8.3 — the two detectors fail in opposite directions on this document.

    Morphology gets the three column rules right and misses row separators;
    TATR finds the rows and over-segments the columns.
    """
    from sarmine.segment.reconcile import merge_row_boundaries

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")
    tatr = _grid([0, 50, 100, 150, 200], [0, 30, 60, 90, 150], detector="tatr")

    merged = merge_row_boundaries(morph, tatr)

    assert merged.x_rulings == [0, 50, 150]
    assert merged.y_rulings == [0, 50, 100, 150, 200]
    assert (merged.n_rows, merged.n_cols) == (4, 2)
    assert len(merged.cells) == 8
    assert merged.detector == "reconciled"
    assert (merged.width, merged.height) == (morph.width, morph.height)


def test_merge_row_boundaries_collapses_near_coincident_rows():
    from sarmine.segment.reconcile import merge_row_boundaries

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")
    tatr = _grid([2, 98, 202], [0, 50, 150], detector="tatr")

    merged = merge_row_boundaries(morph, tatr)

    assert merged.n_rows == 2


def test_merge_row_boundaries_drops_bands_thinner_than_a_real_row():
    from sarmine.segment.reconcile import merge_row_boundaries

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")
    tatr = _grid([0, 20, 100, 200], [0, 50, 150], detector="tatr")

    merged = merge_row_boundaries(morph, tatr, min_band_px=40)

    assert merged.y_rulings == [0, 100, 200]


def test_merge_row_boundaries_without_a_second_grid_changes_nothing():
    from sarmine.segment.reconcile import merge_row_boundaries

    morph = _grid([0, 100, 200], [0, 50, 150], detector="morphology")

    merged = merge_row_boundaries(morph, None)

    assert merged.y_rulings == morph.y_rulings
    assert merged.cells == morph.cells


def test_reconcile_counts_a_missing_detector_as_zero():
    from sarmine.segment.reconcile import reconcile

    result = reconcile(_grid([0, 100, 200], [0, 50, 150]), None)

    assert result.n_tatr == 0
    assert result.n_morph == 4
    assert result.disagreement is False
    assert result.detail


# --------------------------------------------------------------------------
# roles.py — column roles, including the ink-density signal (Plan 4.2)
# --------------------------------------------------------------------------


def _role_grid(x_rulings: list[int]):
    return _grid([0, 100], x_rulings, width=x_rulings[-1], height=100)


def test_roles_module_labels_the_reference_layout():
    from sarmine.segment.roles import assign_column_roles

    grid = _role_grid([0, 285, 1859, 2782])  # number | structure | name
    column_text = {0: "", 1: "O N HN NH F", 2: "isoindoline-1,3-dione name text"}

    assert assign_column_roles(grid, column_text) == {0: "number", 1: "structure", 2: "name"}


def test_roles_module_keeps_the_structure_column_even_when_it_ocrs_to_more_letters():
    """A drawing full of atom labels can out-score the name cell on letters.

    Geometry decides the structure column first; the name is then the wordiest
    of what is left. Getting this backwards would send the drawing to Tesseract
    and the name to OCSR — the exact swap PRD R8.1 forbids.
    """
    from sarmine.segment.roles import assign_column_roles

    grid = _role_grid([0, 285, 1859, 2782])  # number | structure | name
    column_text = {
        0: "5",
        1: "O N HN NH F O O N H N O F HN NH O N N H O F N O H N HN NH O N",
        2: "isoindoline-1,3-dione",
    }

    assert assign_column_roles(grid, column_text) == {0: "number", 1: "structure", 2: "name"}


def test_roles_module_prefers_ink_density_over_width_for_the_structure():
    from sarmine.segment.roles import assign_column_roles

    grid = _role_grid([0, 100, 900, 2000])  # the widest column is the last
    column_text = {0: "3", 1: "N O", 2: "dioxopiperidinyl isoindoline dione"}
    column_ink = {0: 0.01, 1: 0.42, 2: 0.05}

    roles = assign_column_roles(grid, column_text, column_ink)

    assert roles == {0: "number", 1: "structure", 2: "name"}


def test_roles_module_marks_columns_it_cannot_name():
    from sarmine.segment.roles import assign_column_roles

    grid = _role_grid([0, 120, 900, 1600, 2200])
    column_text = {0: "5", 1: "N O", 2: "isoindoline dione name", 3: "0.5"}

    roles = assign_column_roles(grid, column_text)

    assert roles[0] == "number"
    assert roles[1] == "structure"
    assert roles[2] == "name"
    assert roles[3] == "unknown"


def test_roles_module_labels_a_two_column_table():
    from sarmine.segment.roles import assign_column_roles

    grid = _role_grid([0, 600, 700])

    assert assign_column_roles(grid, {0: "isoindoline dione", 1: "12"}) == {
        0: "name",
        1: "number",
    }


def test_roles_module_on_a_grid_without_columns_is_empty():
    from sarmine.segment.roles import assign_column_roles

    assert assign_column_roles(_grid([0, 100], []), {}) == {}


# --------------------------------------------------------------------------
# AC-2.2 — measured cell coverage over the compound table
# --------------------------------------------------------------------------

# Measured with the PRD R8.2 parameters, not assumed. Pages 61-64 carry
# compounds 1-8 and yield 7 row bands, one of which is a spurious 117-px sliver
# at the top of page 61.
FIXTURE_ROW_BANDS = {61: 3, 62: 1, 63: 2, 64: 1}
# Over the whole compound table (PDF pages 61-88). Spike S2 recorded 37 of 54.
CORPUS_ROW_BANDS = 38
AC_2_2_TARGET = 50


def test_ac_2_2_morphology_alone_falls_short_on_the_committed_pages(tmp_path):
    """The baseline the second detector has to beat, pinned so it cannot slip."""
    from sarmine.segment.rulings import detect_grid

    found = {}
    for page_no in (61, 62, 63, 64):
        page = _derotate(PAGES / f"p-{page_no:03d}-000.png", tmp_path / f"p{page_no}.png")
        grid = detect_grid(page)
        assert grid.n_cols == 3, f"page {page_no} is not a 3-column compound table"
        found[page_no] = grid.n_rows

    assert found == FIXTURE_ROW_BANDS
    assert sum(found.values()) < 8  # compounds 1-8 live on these four pages


@pytest.mark.slow
@NEEDS_PDF
def test_ac_2_2_morphology_only_coverage_over_the_whole_compound_table(tmp_path):
    """AC-2.2 wants >=50 of 54 cells; morphology alone reaches 38, and that is
    the honest number for this detector.

    It reproduces spike S2's 37 (the extra band is the page-61 sliver). Closing
    the gap needs the second detector (`tatr.py`, PRD R8.3), which cannot run
    here because `transformers` is not installed.
    """
    from sarmine.segment.rulings import detect_grid

    total = 0
    for page_no in range(61, 89):  # Table 1's full extent in the reference PDF
        page = _render_pdf_page(page_no, tmp_path)
        grid = detect_grid(page)
        page.unlink()
        if grid.n_cols == 3:
            total += grid.n_rows

    assert total == CORPUS_ROW_BANDS
    assert total < AC_2_2_TARGET


MIN_NAME_BAND_PX = 150  # a compound row at 200 dpi is 300-900 px tall
_IUPAC_HINT = re.compile(r"dione|amino|yl\)|carbox|amide")


@pytest.mark.slow
@NEEDS_PDF
@pytest.mark.skipif(not _transformers_installed(), reason="transformers is not installed")
def test_ac_2_2_two_detectors_recover_at_least_fifty_of_the_fifty_four_name_cells(tmp_path):
    """AC-2.2 — >=50 of the 54 compound-table cells, measured end to end.

    A cell counts as detected when the segmentation isolates a region whose OCR
    reads as a chemical name, which is the property the rest of the pipeline
    depends on. Morphology alone reaches 37, exactly spike S2's number; adding
    TATR's row boundaries takes it past the target.
    """
    from sarmine.segment.crops import write_crop
    from sarmine.segment.reconcile import merge_row_boundaries
    from sarmine.segment.roles import assign_column_roles
    from sarmine.segment.rulings import detect_grid
    from sarmine.segment.tatr import detect_grid_tatr

    names: list[str] = []
    crops_dir = tmp_path / "crops"
    for page_no in range(61, 89):  # Table 1's full extent in the reference PDF
        page = _render_pdf_page(page_no, tmp_path)
        morph = detect_grid(page)
        if morph.n_cols != 3:
            page.unlink()
            continue

        grid = merge_row_boundaries(morph, detect_grid_tatr(page), min_band_px=MIN_NAME_BAND_PX)
        name_col = next(
            col
            for col, role in assign_column_roles(
                grid, {0: "", 1: "N O HN", 2: "isoindoline dione name"}
            ).items()
            if role == "name"
        )
        for row in range(grid.n_rows):
            cell = grid.cell(row, name_col)
            prov = write_crop(
                page, cell.bbox, crops_dir,
                page_no=page_no, kind="name", idx=row,
                source="pdf_ocr", extractor="segment.crops", rotation_applied=90,
            )
            text = " ".join(_tesseract(tmp_path / prov.crop_path).split())
            if len(text) > 40 and _IUPAC_HINT.search(text):
                names.append(re.sub(r"[^a-z0-9]", "", text.lower()))
        page.unlink()

    distinct: list[str] = []
    for name in names:
        if not any(name in seen or seen in name for seen in distinct):
            distinct.append(name)

    print(f"\nAC-2.2: {len(names)} name cells detected, {len(distinct)} distinct, of 54")
    assert len(distinct) >= AC_2_2_TARGET
