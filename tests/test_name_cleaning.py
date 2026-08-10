"""OCR name cleaning before OPSIN (PRD §9.2, R9.4, EC-8).

A name cell is OCR'd as several physical lines. IUPAC names carry meaningful
hyphens, and a table cell wraps them mid-name, so joining the lines with a space
inserts a space that OPSIN rejects — measured on compound 5 of the reference
patent, whose name cell OCRs across four lines.
"""

from __future__ import annotations

from sarmine.ocr.homoglyph import clean_ocr_name
from sarmine.structure.opsin import parse_names

# Verbatim tesseract output for the name cell of compound 5, PDF page 63.
COMPOUND_5_OCR = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-\n"
    "4-methoxypheny])-1-methyl-1H-\n"
    "benzo[d]imidazol-6-yl)amino)isoindoline-\n"
    "1,3-dione\n"
)

COMPOUND_5_INCHIKEY = "WZPDSZGYLXZFEK-UHFFFAOYSA-N"


def test_a_hyphen_at_a_line_break_is_part_of_the_name():
    assert clean_ocr_name("2-(3-fluoro-\n4-methoxyphenyl)") == "2-(3-fluoro-4-methoxyphenyl)"


def test_a_line_break_inside_a_locant_closes_up():
    assert clean_ocr_name("isoindoline-\n1,3-dione") == "isoindoline-1,3-dione"


def test_words_separated_by_a_break_keep_their_space():
    assert clean_ocr_name("benzoic\nacid") == "benzoic acid"


def test_surrounding_whitespace_and_dashes_are_normalized():
    assert clean_ocr_name("  4\u2013amino\u2014benzene  ") == "4-amino-benzene"


def test_cleaning_compound_5_leaves_only_the_homoglyph_error():
    cleaned = clean_ocr_name(COMPOUND_5_OCR)
    assert " " not in cleaned
    assert cleaned == (
        "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxypheny])-1-methyl-"
        "1H-benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
    )


def test_raw_ocr_name_fails_opsin_but_the_repaired_one_parses():
    """The whole point: cleaning is necessary, and repair finishes the job."""
    from sarmine.ocr.homoglyph import repair_batch

    raw_result = parse_names([COMPOUND_5_OCR])[0]
    assert raw_result.smiles is None

    cleaned = clean_ocr_name(COMPOUND_5_OCR)
    repaired = repair_batch([cleaned], lambda batch: [r.smiles for r in parse_names(batch)])[0]
    assert repaired.smiles is not None, "the ']'->'l' confusion should be repaired (R9.4)"

    from sarmine.structure.opsin import smiles_to_inchikey

    assert smiles_to_inchikey(repaired.smiles) == COMPOUND_5_INCHIKEY
