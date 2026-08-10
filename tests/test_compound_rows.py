"""Which detected row bands are actually compound rows (PRD R8.4, EC-4).

Reconciling the two table detectors takes the union of their row boundaries,
which is what lifts compound-number recovery from 35/54 to 51/54 and satisfies
AC-2.2. A union necessarily also produces bands that are not compound rows —
header strips, captions, whitespace between drawings — and a real run emitted
111 `Compound` rows for a patent that contains 54.

The filter has to be asymmetric, because the two failure modes cost differently:

* A band with neither a number nor a name is noise. Dropping it costs nothing.
* A band with a name but no readable number IS a real compound whose number
  failed to OCR. It must be KEPT — PRD R11.5/EC-4 say flag the gap and never
  invent the number, and PRD R11.4 says a compound with no activity data still
  belongs in the table.
"""

from __future__ import annotations

from sarmine.pipeline import is_compound_row

IUPAC_NAME = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-1H-"
    "benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
)
# Verbatim OCR of a structure drawing from the reference patent's page 65.
DRAWING_NOISE = "“OS: _ oS: ONO ON\nFAT 1\nA in A\nO\nN O\nNH\n\nO O\n\nNoy\ny\nO O HN\nl"


def test_a_row_with_a_number_and_a_name_is_a_compound_row():
    assert is_compound_row(5, IUPAC_NAME) is True


def test_a_row_with_a_name_but_no_readable_number_is_kept():
    """PRD R11.5 / EC-4 — the number is flagged, never invented, and PRD R11.4
    keeps the compound in the table regardless."""
    assert is_compound_row(None, IUPAC_NAME) is True


def test_a_row_with_a_number_but_no_name_is_kept():
    """The structure drawing may still be readable by the image channel."""
    assert is_compound_row(12, "") is True


def test_a_band_with_neither_a_number_nor_a_name_is_dropped():
    assert is_compound_row(None, "") is False


def test_a_band_holding_only_structure_drawing_noise_is_dropped():
    """This is the band the union of row boundaries actually produces."""
    assert is_compound_row(None, DRAWING_NOISE) is False


def test_a_band_of_stray_punctuation_is_dropped():
    assert is_compound_row(None, "( \n ee \n O: \n ,,, \n |") is False


def test_a_header_strip_is_dropped():
    assert is_compound_row(None, "Compound No. Structure Name") is False


def test_a_truncated_but_real_name_fragment_is_kept():
    """OCR sometimes clips a name; a genuine chemical fragment is still a row."""
    assert is_compound_row(None, "4-((6-(2-(dimethylamino)ethoxy)-4-phenoxypyridin") is True
