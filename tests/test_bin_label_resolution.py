"""`published_value` stays verbatim; `bin_label_raw` carries the resolved bin.

PRD R10.1 makes the published value immutable — every downstream dispute is only
debuggable if the raw OCR string survives, so a cell tesseract read as `Cc` must
still say `Cc`. But the bin it denotes is `C`, and that is what ranking, export
and evaluation compare on. The two therefore cannot be the same field.
"""

from __future__ import annotations

from sarmine.artifacts.schema import BinDefinition, Provenance
from sarmine.assay.lexicon import HeaderMatch
from sarmine.assay.normalize import build_measurement

LEGENDS = {
    "HbF Induction (%)": [
        BinDefinition(label="A", assay="HbF Induction (%)", lower=66.0, upper=100.0, units="%", score=3),
        BinDefinition(label="C", assay="HbF Induction (%)", upper=33.0, units="%", score=1),
    ]
}

HEADER = HeaderMatch(
    standard_type="Induction",
    published_type="HbF Induction (%)",
    units="%",
    target="HbF",
    bao_endpoint="BAO_0000201",
    is_log_form=False,
    confidence=1.0,
    matched_alias="HbF Induction (%)",
)

PROVENANCE = Provenance(
    page_no=186,
    bbox=(0, 0, 10, 10),
    raster_width=100,
    raster_height=100,
    crop_path="crops/p186_activity_1.png",
    source="pdf_ocr",
    extractor="tesseract",
)


def _build(raw: str):
    return build_measurement(
        compound_id="WO:26",
        header=HEADER,
        raw_cell=raw,
        provenance=PROVENANCE,
        assay_group_key="WO::HbF",
        legends=LEGENDS,
    )


def test_case_duplicated_cell_keeps_its_verbatim_published_value():
    measurement = _build("Cc")
    assert measurement is not None
    assert measurement.published_value == "Cc"


def test_case_duplicated_cell_resolves_to_the_real_bin_label():
    measurement = _build("Cc")
    assert measurement.bin_label_raw == "C"
    assert measurement.bin_score == 1
    assert measurement.bin_upper_nM == 33.0


def test_a_clean_cell_is_unchanged():
    measurement = _build("A")
    assert measurement.published_value == "A"
    assert measurement.bin_label_raw == "A"
    assert measurement.bin_score == 3


def test_no_midpoint_is_ever_imputed():
    """PRD R10.5 — a bin is interval-censored; `standard_value` stays null."""
    assert _build("Cc").standard_value is None
    assert _build("A").standard_value is None


def test_an_unreadable_cell_produces_no_measurement():
    """PRD EC-14 — refuse, queue for review; never guess a bin."""
    assert _build("DO") is None
