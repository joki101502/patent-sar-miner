"""Multi-page table stitching (PRD R8.5, EC-3, AC-2.4, Plan 4.5).

**This is our own code — no surveyed tool does it.** Azure returns multiple
`boundingRegions`, Textract returns independent per-page `TABLE` blocks; neither
emits a continuation flag.

The reference patent requires it: Table 2 spans pages 186-187 and **page 187 has
no header row** — it opens directly at compound 32. Stitching is decided on
column-geometry similarity plus header-signature continuity: if page N+1's
column x-boundaries match page N's within tolerance and its first row does not
parse as a header, it is a continuation.

Provenance survives the stitch: every row carries the page it came from and the
cell boxes it was read from, so a 54-row logical table still answers "which page
did row 40 come from?".

Import-safe: pure Python, no models.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from sarmine.artifacts.schema import DocumentAnomaly
from sarmine.segment.rulings import Cell, Grid

COLUMN_TOL_FRAC = 0.05
# A geometry match this close to the tolerance is a stitch we are not confident
# about; beyond `RELAXED_TOL_MULTIPLIER × tol` the pages are treated as separate
# tables but the near-miss is still surfaced.
MARGINAL_FRACTION = 0.5
RELAXED_TOL_MULTIPLIER = 2.0

# Words that mark a row as a column header rather than data. Deliberately broad:
# a false "this is a header" only splits a table, which is visible; a false
# "this is data" silently swallows a header into the row set.
HEADER_WORDS = {
    "compound", "cmpd", "cpd", "example", "structure", "name", "number", "no",
    "id", "entry", "assay", "activity", "target", "induction", "inhibition",
    "level", "bin", "result", "data", "ec50", "ic50", "dc50", "ki", "kd",
    "dmax", "potency", "conc", "concentration", "units", "nm", "um", "µm",
    "hbf", "wiz", "zbtb7a", "hibit", "procedure", "synthesis", "coupling",
}

_WORDY = re.compile(r"[A-Za-z]{3,}|[A-Za-z]{2,}\d")
_BARE = re.compile(r"^[^0-9A-Za-z]*(?:\d{1,4}|[A-Za-z])[^0-9A-Za-z]*$")
_TOKEN = re.compile(r"[a-zµ0-9]+")


class TableRow:
    """One data row of a logical table, tied to the page it came from.

    Rows compare and unpack as `(page_no, texts)` pairs so a caller that only
    cares about the values can ignore the geometry, while `cells` keeps the
    boxes needed to re-crop or to build provenance.
    """

    __slots__ = ("page_no", "cells", "texts")

    def __init__(
        self,
        page_no: int,
        cells: Sequence[Cell] = (),
        texts: Sequence[str] = (),
    ) -> None:
        self.page_no = page_no
        self.cells = list(cells)
        self.texts = list(texts)

    def __iter__(self):
        return iter((self.page_no, self.texts))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TableRow):
            return (self.page_no, self.texts, self.cells) == (
                other.page_no,
                other.texts,
                other.cells,
            )
        if isinstance(other, tuple) and len(other) == 2:
            return (self.page_no, self.texts) == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"TableRow(page_no={self.page_no}, texts={self.texts!r}, cells={len(self.cells)})"


@dataclass
class TablePage:
    """One physical page's table, with as much of its text as the caller has.

    `first_row_text` is all the stitch decision needs; `rows` carries the full
    per-cell text when the caller has already OCR'd the page.
    """

    page_no: int
    grid: Grid
    first_row_text: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    header_texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.first_row_text:
            self.first_row_text = list(self.header_texts or (self.rows[0] if self.rows else []))
        if not self.header_texts:
            self.header_texts = list(self.first_row_text)


@dataclass
class LogicalTable:
    """One table, possibly assembled from several physical pages."""

    page_nos: list[int] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)
    stitch_uncertain: bool = False
    anomalies: list[DocumentAnomaly] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return self.page_nos


def looks_like_header(texts: Sequence[str]) -> bool:
    """Does this row of cell texts read as a column header?

    `["Compound No.", "HbF Induction (%)", ...]` is a header;
    `["32", "A", "D", "H"]` — the first row of page 187 — is not.
    """
    cells = [t.strip() for t in texts if t and t.strip()]
    if not cells:
        return False

    # A row of bare small integers or single letters is data, whatever else it
    # happens to contain. This is the page-187 case.
    bare = sum(1 for cell in cells if _BARE.match(cell))
    if bare * 2 >= len(cells):
        return False

    wordy = sum(1 for cell in cells if _WORDY.search(cell))
    if wordy * 2 < len(cells):
        return False

    tokens = {t for cell in cells for t in _TOKEN.findall(cell.lower())}
    return bool(tokens & HEADER_WORDS)


def column_deviation(a: Grid, b: Grid) -> float:
    """Largest width-normalized difference between two grids' column edges.

    `inf` when the grids cannot be compared at all (different column counts, or
    a zero-width page).
    """
    if len(a.x_rulings) != len(b.x_rulings) or not a.x_rulings:
        return math.inf
    if a.width <= 0 or b.width <= 0:
        return math.inf
    return max(abs(xa / a.width - xb / b.width) for xa, xb in zip(a.x_rulings, b.x_rulings))


def columns_match(a: Grid, b: Grid, *, tol_frac: float = COLUMN_TOL_FRAC) -> bool:
    """Do two pages share a column layout, scale-normalized by image width?"""
    return column_deviation(a, b) <= tol_frac


def is_continuation(
    prev: TablePage, nxt: TablePage, *, tol_frac: float = COLUMN_TOL_FRAC
) -> bool:
    """Does `nxt` continue `prev`'s table (PRD R8.5)?

    Three conditions: the pages are adjacent, their column geometry matches, and
    `nxt` does not open with a header of its own.
    """
    if nxt.page_no != prev.page_no + 1:
        return False
    if looks_like_header(nxt.first_row_text):
        return False
    return columns_match(prev.grid, nxt.grid, tol_frac=tol_frac)


def _page_rows(page: TablePage, skip_first: bool) -> list[TableRow]:
    """Every data row of one page, with its cells and any text the caller has."""
    start = 1 if skip_first else 0
    n_rows = max(page.grid.n_rows, len(page.rows))
    rows: list[TableRow] = []
    for index in range(start, n_rows):
        texts = page.rows[index] if index < len(page.rows) else []
        if not texts and index == 0:
            texts = page.first_row_text
        rows.append(TableRow(page.page_no, page.grid.row_cells(index), texts))
    return rows


def stitch_tables(
    pages: list[TablePage], *, tol_frac: float = COLUMN_TOL_FRAC
) -> list[LogicalTable]:
    """Group per-page tables into logical tables (PRD R8.5).

    A page continues the previous one when it is the next page, its columns line
    up and its first row is not a header. Marginal decisions set
    `stitch_uncertain` and emit a `table_stitch_uncertain` anomaly rather than
    being resolved silently.
    """
    tables: list[LogicalTable] = []
    current: LogicalTable | None = None
    previous: TablePage | None = None

    for page in sorted(pages, key=lambda p: p.page_no):
        has_header = looks_like_header(page.first_row_text)

        continuation = False
        note: str | None = None

        if current is not None and previous is not None:
            deviation = column_deviation(previous.grid, page.grid)
            adjacent = page.page_no == previous.page_no + 1
            if is_continuation(previous, page, tol_frac=tol_frac):
                continuation = True
                if deviation > MARGINAL_FRACTION * tol_frac:
                    note = (
                        f"Page {page.page_no} was stitched onto page {previous.page_no} on a "
                        f"marginal column match (deviation {deviation:.3f} against a tolerance "
                        f"of {tol_frac:.3f})."
                    )
            elif adjacent and not has_header and deviation <= RELAXED_TOL_MULTIPLIER * tol_frac:
                note = (
                    f"Page {page.page_no} was started as a new table, but its columns nearly "
                    f"match page {previous.page_no} (deviation {deviation:.3f} against a "
                    f"tolerance of {tol_frac:.3f}) and it has no header row."
                )

        if continuation and current is not None:
            current.page_nos.append(page.page_no)
            current.rows.extend(_page_rows(page, skip_first=False))
            target = current
        else:
            current = LogicalTable(
                page_nos=[page.page_no],
                header=list(page.first_row_text) if has_header else [],
                rows=_page_rows(page, skip_first=has_header),
            )
            tables.append(current)
            target = current

        if note:
            target.stitch_uncertain = True
            target.anomalies.append(
                DocumentAnomaly(
                    kind="table_stitch_uncertain",
                    severity="warning",
                    message=note + " (PRD R8.5 / EC-3)",
                )
            )

        previous = page

    return tables


stitch = stitch_tables
