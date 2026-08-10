"""Letter-bin cells are constrained by their own legend (PRD R10.5, R10.6, §13.2).

Measured against the gold set: compounds 26-30 all read `Cc` where the value is
`C` — one stray mark beside the letter, five identical errors, the single
largest error class in the run. The assay's legend already enumerates the legal
labels for that column, so it is the natural alphabet to read the cell against.

This never invents a value. A cell that does not collapse to exactly one legend
label is left verbatim, which sends it to the review queue as "bin letter
outside the resolved legend" rather than guessing between two candidates.
"""

from __future__ import annotations

from sarmine.pipeline import normalize_bin_cell

HBF = ["A", "B", "C"]
WIZ = ["D", "E", "F"]


def test_a_clean_letter_is_unchanged():
    assert normalize_bin_cell("A", HBF) == "A"


def test_the_measured_error_class_is_repaired():
    """`Cc` is one letter plus a smudge, not a two-letter value."""
    assert normalize_bin_cell("Cc", HBF) == "C"


def test_a_leading_ruling_line_artifact_is_stripped():
    """The ruling bleeds in as `p` on this document."""
    assert normalize_bin_cell("pA", HBF) == "A"


def test_case_is_normalized():
    assert normalize_bin_cell("c", HBF) == "C"


def test_whitespace_is_ignored():
    assert normalize_bin_cell("  E \n", WIZ) == "E"


def test_two_different_legend_labels_are_left_alone_for_review():
    """Ambiguity must reach a human, not be resolved by picking the first."""
    assert normalize_bin_cell("AB", HBF) == "AB"


def test_a_letter_outside_the_legend_is_left_alone_for_review():
    assert normalize_bin_cell("Z", HBF) == "Z"


def test_an_empty_cell_stays_empty():
    """EC-7 — a blank is a gap, never a value."""
    assert normalize_bin_cell("", HBF) == ""
    assert normalize_bin_cell("   ", HBF) == ""


def test_numeric_columns_are_untouched():
    """A patent with real numbers has no letter legend for that column."""
    assert normalize_bin_cell("7.6", []) == "7.6"
    assert normalize_bin_cell(">10,000", []) == ">10,000"


def test_a_numeric_value_is_not_mangled_by_a_letter_legend():
    assert normalize_bin_cell("7.6", HBF) == "7.6"
