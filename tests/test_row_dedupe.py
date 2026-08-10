"""One detected band per compound row (PRD §8.1, R8.3, EC-4, EC-24, EC-26).

Reconciling the two detectors takes the union of their row boundaries, which
lifts compound-number recovery but also splits single rows into several bands.
A measured run on the 54-compound reference patent emitted 111 rows, with
compound 1 appearing three times and compound 7 twice.

The compound table's own numbering is the row identity: two bands carrying the
same compound number are one row seen twice, not two compounds. That is a
segmentation artifact and is distinct from EC-24, which concerns two genuinely
different rows resolving to the same structure.
"""

from __future__ import annotations

from sarmine.pipeline import CompoundCell, dedupe_compound_rows

NAME_5 = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-1H-"
    "benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
)
NAME_6 = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((3-methyl-6-(2-methylpyridin-4-yl)"
    "benzo[d]isoxazol-5-yl)amino)isoindoline-1,3-dione"
)


def cell(number, name, crop=None, page_no=63) -> CompoundCell:
    return CompoundCell(
        page_no=page_no,
        compound_number=number,
        name_raw=name,
        structure_crop=crop,
        provenance=[],
    )


def test_bands_sharing_a_compound_number_collapse_to_one_row():
    cells = [cell(1, NAME_5), cell(1, NAME_5[:40]), cell(1, "")]

    kept = dedupe_compound_rows(cells)

    assert [c.compound_number for c in kept] == [1]


def test_the_band_with_the_fullest_name_wins():
    """The fragment bands are slivers; the real row carries the whole name."""
    cells = [cell(7, NAME_5[:30]), cell(7, NAME_5)]

    kept = dedupe_compound_rows(cells)

    assert kept[0].name_raw == NAME_5


def test_a_structure_crop_is_preserved_when_the_winning_band_lacks_one(tmp_path):
    """Losing the drawing would silently drop the whole image channel for that row."""
    crop = tmp_path / "s.png"
    crop.write_bytes(b"")
    cells = [cell(9, NAME_5[:20], crop=crop), cell(9, NAME_5)]

    kept = dedupe_compound_rows(cells)

    assert len(kept) == 1
    assert kept[0].name_raw == NAME_5
    assert kept[0].structure_crop == crop


def test_distinct_compounds_are_never_merged():
    cells = [cell(5, NAME_5), cell(6, NAME_6)]

    kept = dedupe_compound_rows(cells)

    assert [c.compound_number for c in kept] == [5, 6]


def test_numberless_rows_survive_when_their_name_is_distinct():
    """R11.5 — a real compound whose number failed to OCR must not be dropped."""
    cells = [cell(5, NAME_5), cell(None, NAME_6)]

    kept = dedupe_compound_rows(cells)

    assert len(kept) == 2


def test_a_numberless_fragment_of_a_kept_row_is_dropped():
    """The sliver bands the union produces repeat the neighbouring row's name."""
    cells = [cell(5, NAME_5), cell(None, NAME_5[:60])]

    kept = dedupe_compound_rows(cells)

    assert [c.compound_number for c in kept] == [5]


def test_rows_on_different_pages_are_never_merged():
    """A split band is always within one page, so cross-page merging is unsafe.

    Measured: page 66's compound 11 read as `1` when a digit was dropped, and
    merging it with the real compound 1 from page 61 destroyed compound 11
    outright — a misread number silently deleting a compound is far worse than
    a duplicate number reaching the review queue (EC-4, EC-24).
    """
    cells = [cell(1, NAME_5, page_no=61), cell(1, NAME_6, page_no=66)]

    kept = dedupe_compound_rows(cells)

    assert len(kept) == 2
    assert {c.page_no for c in kept} == {61, 66}


def test_bands_on_the_same_page_still_collapse():
    cells = [cell(1, NAME_5, page_no=61), cell(1, NAME_5[:30], page_no=61)]

    assert len(dedupe_compound_rows(cells)) == 1


def test_order_is_preserved():
    cells = [cell(3, NAME_5), cell(4, NAME_6), cell(3, NAME_5)]

    kept = dedupe_compound_rows(cells)

    assert [c.compound_number for c in kept] == [3, 4]
