"""Compound rows and their source cells must stay index-aligned.

`_run_image_channel` pairs `compounds[i]` with `cells[i]` to attach each OCSR
result to the right compound, so any filtering that shortens one list without
the other silently mis-attributes structures — or, as it did in a real run,
raises `IndexError: list index out of range` and loses the whole run after four
minutes of work.

This is the kind of coupling that deserves a test rather than a comment: the two
lists are built in different functions, and nothing in either signature says
they have to line up.
"""

from __future__ import annotations

from sarmine.pipeline import CompoundCell, _build_compounds

IUPAC_NAME = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-1H-"
    "benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
)
DRAWING_NOISE = "“OS: _ oS: ONO ON\nFAT 1\nA in A\nO\nN O\nNH"


def cell(number: int | None, name: str, page_no: int = 63) -> CompoundCell:
    return CompoundCell(
        page_no=page_no,
        compound_number=number,
        name_raw=name,
        structure_crop=None,
        provenance=[],
    )


def test_build_compounds_returns_one_compound_per_input_cell():
    """The caller pairs the two lists by index, so this must hold even when some
    cells are junk — filtering belongs to the caller, not to this function."""
    cells = [
        cell(5, IUPAC_NAME),
        cell(None, DRAWING_NOISE),
        cell(6, IUPAC_NAME),
        cell(None, ""),
    ]

    compounds, _ = _build_compounds(cells, "WO2024097932A1", "structured")

    assert len(compounds) == len(cells)


def test_compound_order_follows_cell_order():
    cells = [cell(7, IUPAC_NAME), cell(8, IUPAC_NAME), cell(9, IUPAC_NAME)]

    compounds, _ = _build_compounds(cells, "WO2024097932A1", "structured")

    assert [c.compound_number for c in compounds] == [7, 8, 9]
