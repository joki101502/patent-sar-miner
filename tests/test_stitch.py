"""Multi-page table stitching (PRD R8.5, EC-3, AC-2.4, Plan 4.5).

No surveyed commercial tool provides this, so it is our own code and it gets
its own test module. The required case is Table 2 of the reference patent:
pages 186 and 187 are one logical table of 54 rows, and page 187 has **no
header row** — it opens directly at compound 32.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
PAGES = FIXTURES / "pages"

TABLE_2_HEADER = ["Compound No.", "HbF Induction (%)", "WIZ EC50 (uM)", "ZBTB7A EC50 (uM)"]
PAGE_187_FIRST_ROW = ["32", "A", "D", "H"]


def _grid(x_rulings: list[int], n_rows: int = 2, *, width: int = 1000, height: int = 1000):
    from sarmine.segment.rulings import Grid, cells_from_rulings

    y_rulings = [int(i * height / n_rows) for i in range(n_rows + 1)]
    return Grid(
        y_rulings=y_rulings,
        x_rulings=list(x_rulings),
        cells=cells_from_rulings(y_rulings, list(x_rulings)),
        width=width,
        height=height,
    )


def _page(page_no: int, grid, first_row_text: list[str]):
    from sarmine.segment.stitch import TablePage

    return TablePage(page_no=page_no, grid=grid, first_row_text=first_row_text)


# ---------------------------------------------------------------------------
# column geometry
# ---------------------------------------------------------------------------


def test_columns_match_accepts_an_identical_layout():
    from sarmine.segment.stitch import columns_match

    grid = _grid([0, 250, 500, 1000])

    assert columns_match(grid, _grid([0, 250, 500, 1000])) is True


def test_columns_match_tolerates_a_small_shift():
    from sarmine.segment.stitch import columns_match

    assert columns_match(_grid([0, 250, 500, 1000]), _grid([0, 262, 512, 1000])) is True


def test_columns_match_rejects_a_shift_beyond_tolerance():
    from sarmine.segment.stitch import columns_match

    assert columns_match(_grid([0, 250, 500, 1000]), _grid([0, 400, 700, 1000])) is False


def test_columns_match_rejects_a_different_column_count():
    from sarmine.segment.stitch import columns_match

    assert columns_match(_grid([0, 250, 500, 1000]), _grid([0, 500, 1000])) is False


def test_columns_match_honours_an_explicit_tolerance():
    from sarmine.segment.stitch import columns_match

    a, b = _grid([0, 250, 500, 1000]), _grid([0, 290, 540, 1000])

    assert columns_match(a, b, tol_frac=0.01) is False
    assert columns_match(a, b, tol_frac=0.10) is True


def test_columns_match_is_scale_normalized():
    """Two renders of the same table at different dpi still match."""
    from sarmine.segment.stitch import columns_match

    a = _grid([0, 250, 500, 1000], width=1000)
    b = _grid([0, 125, 250, 500], width=500)

    assert columns_match(a, b) is True


# ---------------------------------------------------------------------------
# header signature
# ---------------------------------------------------------------------------


def test_looks_like_header_recognizes_the_real_table_2_header():
    from sarmine.segment.stitch import looks_like_header

    assert looks_like_header(TABLE_2_HEADER) is True


def test_looks_like_header_rejects_the_first_row_of_page_187():
    """EC-3 — page 187 opens at compound 32 with no header."""
    from sarmine.segment.stitch import looks_like_header

    assert looks_like_header(PAGE_187_FIRST_ROW) is False
    assert looks_like_header(["33", "", "E", "H"]) is False


def test_looks_like_header_rejects_nothing_at_all():
    from sarmine.segment.stitch import looks_like_header

    assert looks_like_header([]) is False
    assert looks_like_header(["", "   "]) is False


# ---------------------------------------------------------------------------
# continuation decision
# ---------------------------------------------------------------------------


def test_is_continuation_when_geometry_matches_and_there_is_no_header():
    from sarmine.segment.stitch import is_continuation

    prev = _page(186, _grid([0, 250, 500, 1000]), TABLE_2_HEADER)
    nxt = _page(187, _grid([0, 250, 500, 1000]), PAGE_187_FIRST_ROW)

    assert is_continuation(prev, nxt) is True


def test_is_not_a_continuation_when_the_next_page_has_its_own_header():
    from sarmine.segment.stitch import is_continuation

    prev = _page(186, _grid([0, 250, 500, 1000]), TABLE_2_HEADER)
    nxt = _page(187, _grid([0, 250, 500, 1000]), TABLE_2_HEADER)

    assert is_continuation(prev, nxt) is False


def test_is_not_a_continuation_when_the_columns_differ():
    from sarmine.segment.stitch import is_continuation

    prev = _page(186, _grid([0, 250, 500, 1000]), TABLE_2_HEADER)
    nxt = _page(187, _grid([0, 700, 800, 1000]), PAGE_187_FIRST_ROW)

    assert is_continuation(prev, nxt) is False


def test_is_not_a_continuation_across_a_page_gap():
    """A table that resumes three pages later is a new table until proven otherwise."""
    from sarmine.segment.stitch import is_continuation

    prev = _page(186, _grid([0, 250, 500, 1000]), TABLE_2_HEADER)
    nxt = _page(190, _grid([0, 250, 500, 1000]), PAGE_187_FIRST_ROW)

    assert is_continuation(prev, nxt) is False


# ---------------------------------------------------------------------------
# stitching
# ---------------------------------------------------------------------------


def test_stitch_joins_a_headerless_continuation_page():
    from sarmine.segment.stitch import stitch

    pages = [
        _page(186, _grid([0, 250, 500, 1000], n_rows=3), TABLE_2_HEADER),
        _page(187, _grid([0, 250, 500, 1000], n_rows=2), PAGE_187_FIRST_ROW),
    ]
    tables = stitch(pages)

    assert len(tables) == 1
    table = tables[0]
    assert table.pages == [186, 187]
    assert table.header == TABLE_2_HEADER
    # The header row is not a data row: 3 rows on page 186 minus the header.
    assert [row.page_no for row in table.rows] == [186, 186, 187, 187]
    assert table.stitch_uncertain is False


def test_stitch_rows_carry_their_cell_geometry():
    """Provenance survives the stitch: every row keeps its page and its bboxes."""
    from sarmine.segment.stitch import stitch

    grid = _grid([0, 250, 500, 1000], n_rows=3)
    tables = stitch([_page(186, grid, TABLE_2_HEADER)])

    rows = tables[0].rows
    assert len(rows) == 2
    assert all(len(row.cells) == 3 for row in rows)
    # Row 0 of the logical table is row 1 of the page, the header being row 0.
    assert rows[0].cells[0].bbox == grid.cell(1, 0).bbox
    assert rows[0].texts == []


def test_stitch_keeps_every_row_of_a_headerless_single_page():
    from sarmine.segment.stitch import stitch

    tables = stitch([_page(9, _grid([0, 250, 500, 1000], n_rows=2), ["1", "A", "D"])])

    assert len(tables) == 1
    assert not tables[0].header
    assert len(tables[0].rows) == 2
    assert tables[0].rows[0].texts == ["1", "A", "D"]


def test_stitch_starts_a_new_table_when_the_columns_do_not_match():
    from sarmine.segment.stitch import stitch

    pages = [
        _page(1, _grid([0, 250, 500, 1000]), TABLE_2_HEADER),
        _page(2, _grid([0, 700, 800, 1000]), PAGE_187_FIRST_ROW),
    ]

    assert len(stitch(pages)) == 2


def test_stitch_flags_a_marginal_geometry_match_as_uncertain():
    """R8.5 — a stitch that only just clears tolerance goes to the review queue."""
    from sarmine.segment.stitch import stitch

    pages = [
        _page(1, _grid([0, 250, 500, 1000]), TABLE_2_HEADER),
        _page(2, _grid([0, 290, 540, 1000]), PAGE_187_FIRST_ROW),
    ]
    tables = stitch(pages)

    assert len(tables) == 1
    assert tables[0].stitch_uncertain is True
    assert [a.kind for a in tables[0].anomalies] == ["table_stitch_uncertain"]


def test_stitch_of_no_pages_returns_no_tables():
    from sarmine.segment.stitch import stitch

    assert stitch([]) == []


# ---------------------------------------------------------------------------
# AC-2.4 — the real pages
# ---------------------------------------------------------------------------


def test_ac_2_4_real_pages_186_and_187_stitch_into_one_table():
    """AC-2.4 with grids detected from the committed 300 dpi page rasters."""
    from sarmine.segment.rulings import detect_grid
    from sarmine.segment.stitch import TablePage, is_continuation, stitch

    grid_186 = detect_grid(PAGES / "p-186-000.png")
    grid_187 = detect_grid(PAGES / "p-187-000.png")

    assert (grid_186.n_rows, grid_186.n_cols) == (32, 4)  # header + compounds 1-31
    assert (grid_187.n_rows, grid_187.n_cols) == (23, 4)  # compounds 32-54

    pages = [
        TablePage(page_no=186, grid=grid_186, first_row_text=TABLE_2_HEADER),
        TablePage(page_no=187, grid=grid_187, first_row_text=PAGE_187_FIRST_ROW),
    ]
    assert is_continuation(pages[0], pages[1]) is True

    tables = stitch(pages)

    assert len(tables) == 1
    table = tables[0]
    assert table.pages == [186, 187]
    assert table.header == TABLE_2_HEADER
    assert len(table.rows) == 54
    assert sum(1 for row in table.rows if row.page_no == 186) == 31
    assert sum(1 for row in table.rows if row.page_no == 187) == 23
    assert table.stitch_uncertain is False


@pytest.mark.skipif(
    not (PAGES / "p-187-000.png").is_file(), reason="page fixtures are not present"
)
def test_ac_2_4_page_187s_own_first_row_really_is_not_a_header():
    """The stitch decision must survive real OCR of page 187's first row.

    Tesseract returns near-garbage for these single-letter bin cells, which is
    exactly the input `looks_like_header` has to reject.
    """
    from sarmine.segment.stitch import looks_like_header

    for ocr_output in (["", "PR", "DO", ""], ["32", "A", "D", "H"], ["", "pe", "", ""]):
        assert looks_like_header(ocr_output) is False
