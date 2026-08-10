"""Column role assignment for the compound table (Plan 4.2, PRD §8.2).

Routing depends on knowing which column is which: the name cell goes to
Tesseract and OPSIN, the structure cell goes to OCSR, and neither may ever see
the other's pixels (PRD R8.1). Column order varies by filer, so roles are
derived from geometry and content — never from a hard-coded index.
"""

from __future__ import annotations

import re
from typing import Literal

from sarmine.segment.rulings import Grid

ColumnRole = Literal["number", "structure", "name", "unknown"]

MIN_NAME_ALPHA = 3
# A lone leftover column this narrow relative to the table is a number column,
# not a squeezed structure drawing.
NARROW_COLUMN_FRACTION = 0.25


def _alpha_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


# A chemical name is long lowercase runs (`dioxopiperidin`, `phenoxypyridin`);
# OCR of a structure drawing is short uppercase fragments (`N`, `O`, `HN`, `OS`).
_WORD = re.compile(r"[A-Za-z]{4,}")


def _name_likeness(text: str) -> int:
    """Weight of evidence that a column holds a chemical NAME, not a drawing.

    Raw alphabetic count does NOT work here and this is the bug it caused: OCR
    of a structure drawing emits plenty of letters, and on real pages 65, 67, 71,
    74 and 76 of the reference patent it emits MORE than the IUPAC name beside
    it. Those five pages were dropped from the run entirely — 10 compounds lost —
    because the name column was mistaken for the structure column. Scoring on
    long lowercase words separates them cleanly.
    """
    words = _WORD.findall(text or "")
    lowercase_words = [w for w in words if w.islower()]
    return sum(len(w) for w in lowercase_words)


def _looks_numeric(text: str) -> bool:
    digits = sum(1 for ch in text if ch.isdigit())
    letters = _alpha_count(text)
    return digits > 0 and digits >= letters


def assign_column_roles(
    grid: Grid,
    column_text: dict[int, str],
    column_ink: dict[int, float] | None = None,
) -> dict[int, ColumnRole]:
    """Label each column `number` / `structure` / `name` / `unknown`.

    The structure column is settled first, from geometry: the widest column, or
    the densest when per-column ink coverage is supplied. A drawing's atom
    labels can OCR to more letters than the name beside it, so letters alone
    would sometimes route the drawing to Tesseract and the name to OCSR — the
    swap PRD R8.1 forbids. Of what remains, the name column is the one whose
    OCR yields the most alphabetic characters and the compound-number column is
    the narrowest. Anything left over is `unknown` rather than guessed.
    """
    widths = grid.column_widths()
    if not widths:
        return {}

    columns = list(range(len(widths)))
    roles: dict[int, ColumnRole] = {column: "unknown" for column in columns}
    remaining = list(columns)

    # The NAME column is settled first, because it is the only one with a
    # trustworthy signal. Neither width nor ink identifies the structure column:
    # on source page 65 the name column is the WIDEST, and on page 63 the dense
    # name text carries the HIGHEST ink coverage. Long lowercase words, by
    # contrast, separate a IUPAC name from a drawing's atom labels every time.
    likeness = {column: _name_likeness(column_text.get(column, "")) for column in remaining}
    name_col = max(remaining, key=lambda c: (likeness[c], -c))
    if likeness[name_col] >= MIN_NAME_ALPHA:
        roles[name_col] = "name"
        remaining.remove(name_col)

    # A two-column table is `name | number`; it has no structure column to find.
    if len(columns) >= 3 and remaining:
        if column_ink:
            structure_col = max(remaining, key=lambda c: (column_ink.get(c, 0.0), widths[c], -c))
        else:
            structure_col = max(remaining, key=lambda c: (widths[c], -c))
        roles[structure_col] = "structure"
        remaining.remove(structure_col)

    if len(remaining) == 1 and len(columns) < 3:
        only = remaining[0]
        is_narrow = widths[only] < NARROW_COLUMN_FRACTION * sum(widths)
        if is_narrow or _looks_numeric(column_text.get(only, "")):
            roles[only] = "number"
        else:
            roles[only] = "structure"
    elif remaining:
        roles[min(remaining, key=lambda c: (widths[c], c))] = "number"

    return roles
