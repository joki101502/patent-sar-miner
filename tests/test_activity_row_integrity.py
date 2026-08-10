"""Two defects found by scoring a real run against the gold set.

1. Tesseract reads the single letter `C` as `Cc` in every one of Table 2's five
   `C` cells — its usual case-duplication on an isolated capital. Those cells
   were being dropped, losing five real values.
2. A row on page 187 mis-reads its compound number as `4`, which already exists
   on page 186. Silently accepting it overwrote compound 4's real values with
   another compound's. PRD R11.5 / EC-4: flag, never interpolate, never guess.
"""

from __future__ import annotations

from sarmine.artifacts.schema import BinDefinition
from sarmine.assay.legend import decode_bin
from sarmine.pipeline import accept_compound_number

LEGENDS = {
    "HbF Induction (%)": [
        BinDefinition(label="A", assay="HbF Induction (%)", lower=66.0, upper=100.0, units="%", score=3),
        BinDefinition(label="B", assay="HbF Induction (%)", lower=33.0, upper=66.0, units="%", score=2),
        BinDefinition(label="C", assay="HbF Induction (%)", upper=33.0, units="%", score=1),
    ],
    "WIZ EC50 (uM)": [
        BinDefinition(label="D", assay="WIZ EC50 (uM)", upper=0.01, units="uM", score=3),
        BinDefinition(label="E", assay="WIZ EC50 (uM)", lower=0.01, upper=0.1, units="uM", score=2),
        BinDefinition(label="F", assay="WIZ EC50 (uM)", lower=0.1, units="uM", score=1),
    ],
}


def test_case_duplicated_letter_decodes_to_its_bin():
    """`C` OCR'd as `Cc` is still a C — measured on all five C cells in Table 2."""
    assert decode_bin("Cc", LEGENDS) is not None
    assert decode_bin("Cc", LEGENDS).label == "C"


def test_other_case_duplications_decode_too():
    for raw in ("Aa", "aA", "bB", "dD"):
        decoded = decode_bin(raw, LEGENDS)
        assert decoded is not None, raw
        assert decoded.label == raw[0].upper()


def test_two_different_letters_are_not_collapsed():
    """`DO` is not a `D`: only a repeat of the SAME letter is a case duplication."""
    assert decode_bin("DO", LEGENDS) is None
    assert decode_bin("EF", LEGENDS) is None


def test_an_unknown_letter_is_still_refused():
    assert decode_bin("Z", LEGENDS) is None
    assert decode_bin("", LEGENDS) is None


def test_a_compound_number_is_accepted_when_it_advances():
    seen: set[int] = set()
    assert accept_compound_number(1, seen, last=None) is True
    assert accept_compound_number(2, {1}, last=1) is True
    assert accept_compound_number(32, {1, 2}, last=2) is True


def test_a_repeated_compound_number_is_refused():
    """EC-4 — the second `4` in a stitched table must not overwrite the first."""
    assert accept_compound_number(4, {1, 2, 3, 4, 32, 33}, last=33) is False


def test_a_number_that_goes_backwards_is_refused():
    """Page 187 opens at 32; a `4` after that is a misread, not a compound."""
    assert accept_compound_number(4, {32, 33}, last=33) is False


def test_a_missing_number_is_refused_rather_than_invented():
    assert accept_compound_number(None, {1}, last=1) is False
