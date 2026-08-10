"""Wide SAR table + XLSX/CSV export (PRD §15 preamble, G7, R13.6, AC-7.5).

Storage is LONG — one row per measurement — but display and export are WIDE,
one row per compound. The pivot has to preserve two things the naive version
loses: the verbatim letter next to its decoded interval (PRD C1/R10.5), and the
distinction between "no value reported" and "value is low" (EC-7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sarmine.artifacts.schema import Compound, Measurement, Provenance
from sarmine.export import (
    correction_rows,
    to_csv,
    to_wide_frame,
    to_xlsx,
)


def prov(page_no: int = 186) -> Provenance:
    return Provenance(
        page_no=page_no,
        bbox=(10, 20, 30, 40),
        raster_width=2477,
        raster_height=3505,
        crop_path=f"crops/p{page_no:03d}_activity_0.png",
        source="structured",
        extractor="tesseract@5.5.3",
    )


def compound(num: int, **kw) -> Compound:
    base = dict(
        compound_id=f"WO:{num}",
        compound_local_id=str(num),
        compound_number=num,
        smiles_final="CCO",
        inchikey_full="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        crosscheck_tier="AGREE_FULL",
        rdkit_version="2026.03.5",
        provenance=[prov(63)],
        rank=num,
        potency_score=7,
        selectivity_score=2,
        rank_rationale=["WIZ D (top bin)"],
    )
    base.update(kw)
    return Compound(**base)


def bin_measurement(num: int, assay: str, letter: str | None, **kw) -> Measurement:
    base = dict(
        measurement_id=f"WO:{num}:{assay}",
        compound_id=f"WO:{num}",
        assay_group_key=f"WO2024097932A1::{assay}",
        assay_name_raw=assay,
        published_type=assay,
        published_value=letter or "",
        standard_type="EC50",
        standard_relation="<",
        standard_value=None,
        bin_label_raw=letter,
        bin_lower_nM=None,
        bin_upper_nM=10.0,
        is_censored=True,
        provenance=prov(),
    )
    base.update(kw)
    return Measurement(**base)


def test_wide_frame_has_one_row_per_compound_and_a_column_per_assay():
    compounds = [compound(1), compound(2)]
    measurements = [
        bin_measurement(1, "WIZ EC50 (uM)", "D"),
        bin_measurement(1, "ZBTB7A EC50 (uM)", "I"),
        bin_measurement(2, "WIZ EC50 (uM)", "E"),
        bin_measurement(2, "ZBTB7A EC50 (uM)", "H"),
    ]

    frame = to_wide_frame(compounds, measurements)

    assert len(frame) == 2
    assert "WIZ EC50 (uM)" in frame.columns
    assert "ZBTB7A EC50 (uM)" in frame.columns
    assert frame.loc[frame["compound_number"] == 1, "WIZ EC50 (uM)"].item() == "D"


def test_letter_is_exported_beside_its_decoded_interval():
    """PRD C1/R10.5 — the letter is the extracted value, the interval is the
    normalized one. Exporting only one of them loses the audit trail."""
    frame = to_wide_frame([compound(1)], [bin_measurement(1, "WIZ EC50 (uM)", "D")])

    assert frame.loc[0, "WIZ EC50 (uM)"] == "D"
    assert "< 10" in str(frame.loc[0, "WIZ EC50 (uM) (nM)"])


def test_a_blank_cell_stays_blank_and_never_becomes_a_number():
    """PRD EC-7 / R12.9 — blank is not zero and not a low value."""
    frame = to_wide_frame([compound(33)], [bin_measurement(33, "HbF Induction (%)", None)])

    value = frame.loc[0, "HbF Induction (%)"]
    assert value in ("", None) or (isinstance(value, float) and str(value) == "nan")
    assert str(value) != "0"


def test_provenance_columns_accompany_every_compound():
    """PRD AC-8.1 — every SMILES carries page and crop."""
    frame = to_wide_frame([compound(1)], [bin_measurement(1, "WIZ EC50 (uM)", "D")])

    assert frame.loc[0, "structure_page"] == 63
    assert "crops/" in frame.loc[0, "structure_crop"]


def test_confidence_and_channel_columns_are_exported():
    """The reviewer must be able to see which channels produced a row."""
    frame = to_wide_frame([compound(1, structure_source="name+image")], [])

    for column in ("crosscheck_tier", "structure_source", "inchikey_full", "smiles_final"):
        assert column in frame.columns


def test_csv_round_trips(tmp_path: Path):
    out = to_csv([compound(1)], [bin_measurement(1, "WIZ EC50 (uM)", "D")], tmp_path / "sar.csv")

    assert out.is_file()
    text = out.read_text("utf-8")
    assert "WIZ EC50 (uM)" in text and "D" in text


def test_xlsx_is_written_with_both_sheets(tmp_path: Path):
    """PRD G7 — Excel export. Corrections travel with the data (AC-7.5)."""
    openpyxl = pytest.importorskip("openpyxl")
    out = to_xlsx(
        [compound(1)],
        [bin_measurement(1, "WIZ EC50 (uM)", "D")],
        tmp_path / "sar.xlsx",
        corrections=[
            {
                "target_kind": "compound",
                "target_id": "WO:1",
                "field": "smiles_final",
                "original": "CCO",
                "corrected": "CCC",
                "timestamp": "2026-08-09T00:00:00Z",
                "note": None,
            }
        ],
    )

    book = openpyxl.load_workbook(out)
    assert "SAR table" in book.sheetnames
    assert "Measurements" in book.sheetnames
    assert "Corrections" in book.sheetnames


def test_correction_rows_carry_original_and_corrected_side_by_side():
    """PRD AC-7.5 — the export shows what was changed, not just the new value."""
    rows = correction_rows(
        [
            {
                "target_kind": "compound",
                "target_id": "WO:1",
                "field": "smiles_final",
                "original": "CCO",
                "corrected": "CCC",
                "timestamp": "2026-08-09T00:00:00Z",
                "note": "redrawn from crop",
            }
        ]
    )

    assert rows[0]["original"] == "CCO"
    assert rows[0]["corrected"] == "CCC"


def test_measurements_sheet_keeps_the_long_form():
    """PRD G5 — long internally. The wide view is for humans; the long form is
    what a downstream consumer should read."""
    frame = to_wide_frame([compound(1)], [bin_measurement(1, "WIZ EC50 (uM)", "D")])
    assert "published_value" not in frame.columns  # that lives on the long sheet
