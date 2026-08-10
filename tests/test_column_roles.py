"""Column role assignment on real compound-table pages (Plan 4.2, PRD R8.1).

Getting this wrong is safety-critical, not cosmetic: PRD R8.1 forbids ever
sending a structure drawing to Tesseract or a name cell to OCSR, and the role
assignment is what enforces it. It is also what decides whether a page is
recognised as a compound table at all — five of the reference patent's 28
compound-table pages were being dropped entirely because the name column was
misidentified.

The trap is that raw alphabetic count does NOT identify the name column. OCR of
a structure drawing emits a lot of letters — atom labels and bond-line noise
like `OS ONO ON FAT A in A O N O NH` — and on several real pages it emits MORE
letters than the IUPAC name beside it. What separates them is the shape of the
words: names are long lowercase runs, drawings are short uppercase fragments.
"""

from __future__ import annotations

from sarmine.segment.roles import assign_column_roles
from sarmine.segment.rulings import Grid, cells_from_rulings

# Verbatim OCR captured from the de-rotated source page 65 of WO2024097932A1.
STRUCTURE_OCR_P65 = "“OS: _ oS: ONO ON\nFAT 1\nA in A\nO\nN O\nNH\n\nO O\n\nNoy\ny\nO O HN\nl"
NAME_OCR_P65 = (
    "4-((6-(2-(dimethylamino)ethoxy)-4-\nphenoxypyridin-3-yl)amino)-2-(2,6-"
    "dioxopiperidin-3-yl)isoindoline-1,3-dione"
)
NUMBER_OCR_P65 = "(\n\nee\n\n10\nO:\n"


def grid_from(widths: list[int], n_rows: int = 3) -> Grid:
    xs = [0]
    for width in widths:
        xs.append(xs[-1] + width)
    ys = [i * 500 for i in range(n_rows + 1)]
    return Grid(
        n_rows=n_rows,
        n_cols=len(widths),
        y_rulings=ys,
        x_rulings=xs,
        cells=cells_from_rulings(ys, xs),
        width=xs[-1],
        height=ys[-1],
    )


def test_structure_drawing_ocr_has_more_letters_than_the_name_beside_it():
    """The premise of the bug, pinned so the fix cannot be reasoned away."""
    letters = lambda s: sum(1 for ch in s if ch.isalpha())  # noqa: E731
    assert letters(STRUCTURE_OCR_P65) >= letters(NAME_OCR_P65) * 0.4


def test_name_column_is_identified_on_real_page_65_geometry():
    """Source page 65: the name column is NARROWER than the structure column,
    so neither width nor letter count identifies it. This page and four others
    were silently dropped from the run."""
    grid = grid_from([258, 1034, 1349])
    roles = assign_column_roles(
        grid, {0: NUMBER_OCR_P65, 1: STRUCTURE_OCR_P65, 2: NAME_OCR_P65}
    )

    assert roles[2] == "name"
    assert roles[1] == "structure"
    assert roles[0] == "number"


def test_name_column_is_identified_on_real_page_63_geometry():
    """Source page 63, where the structure column IS the widest — the fix must
    not break the case that already worked."""
    structure = "OQ\nO N\nHN\nsoy yy,\nNo\n\nsone\n\\\n/"
    name = (
        "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-\n4-methoxyphenyl)-1-methyl-1H-\n"
        "benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
    )
    grid = grid_from([285, 1574, 923])
    roles = assign_column_roles(grid, {0: "5", 1: structure, 2: name})

    assert roles[2] == "name"
    assert roles[1] == "structure"
    assert roles[0] == "number"


def test_column_order_is_not_assumed():
    """PRD Plan 4.2 — column order varies by filer, so nothing is indexed by
    position. Same three columns, name first."""
    grid = grid_from([1349, 258, 1034])
    roles = assign_column_roles(
        grid, {0: NAME_OCR_P65, 1: NUMBER_OCR_P65, 2: STRUCTURE_OCR_P65}
    )

    assert roles[0] == "name"
    assert roles[2] == "structure"
    assert roles[1] == "number"


def test_a_column_of_pure_drawing_noise_is_never_called_a_name():
    """PRD R8.1 — a structure drawing must never be routed to the name channel."""
    grid = grid_from([258, 1034, 1349])
    roles = assign_column_roles(
        grid, {0: "12", 1: STRUCTURE_OCR_P65, 2: "O N HN NH O O N"}
    )

    assert roles[1] != "name"
    assert roles[2] != "name"


def test_ink_density_still_wins_for_the_structure_column_when_supplied():
    grid = grid_from([258, 1034, 1349])
    roles = assign_column_roles(
        grid,
        {0: NUMBER_OCR_P65, 1: STRUCTURE_OCR_P65, 2: NAME_OCR_P65},
        {0: 0.023, 1: 0.043, 2: 0.027},
    )

    assert roles[1] == "structure"
    assert roles[2] == "name"
