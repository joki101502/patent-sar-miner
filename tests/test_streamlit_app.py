"""Part 11 — the Streamlit surface (PRD §14, R13.4–R13.6, R14.3–R14.5).

Streamlit's own test harness is not a dependency here, so these exercise the
pure functions the screens are built from: the frame the SAR table renders, the
correction replay, and the export. That is where the logic lives; the widget
calls around it are declarative.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from sarmine.artifacts.schema import (
    Bundle,
    Compound,
    Measurement,
    Provenance,
    RunManifest,
)

app = pytest.importorskip("app.streamlit_app", reason="streamlit not installed")


def _provenance() -> Provenance:
    return Provenance(
        page_no=63,
        bbox=(10, 10, 110, 110),
        raster_width=2477,
        raster_height=3505,
        crop_path="crops/p063_structure_0.png",
        source="pdf_ocr",
        extractor="molscribe@1.1.1",
        rotation_applied=90,
    )


def _bundle(tmp_path: Path) -> Bundle:
    compound = Compound(
        compound_id="WO:5",
        compound_local_id="5",
        compound_number=5,
        smiles_final="CCO",
        smiles_from_name="CCO",
        inchikey_full="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        crosscheck_tier="AGREE_FULL",
        rank=1,
        rank_tie_group=1,
        potency_score=8,
        selectivity_score=2,
        rank_rationale=["WIZ D (top bin)"],
        mw=46.07,
        provenance=[_provenance()],
        rdkit_version="2026.03.5",
    )
    measurement = Measurement(
        measurement_id="WO:5:WIZ EC50 (uM)",
        compound_id="WO:5",
        assay_group_key="WO::WIZ",
        assay_name_raw="WIZ EC50 (uM)",
        target_raw="WIZ",
        published_type="WIZ EC50 (uM)",
        published_value="D",
        standard_type="EC50",
        standard_relation="<",
        standard_units="nM",
        bin_label_raw="D",
        bin_upper_nM=10.0,
        bin_score=3,
        is_censored=True,
        provenance=_provenance(),
    )
    manifest = RunManifest(
        pubnum="WO2024097932A1",
        run_id="r1",
        created_at="2026-08-09T00:00:00Z",
        source_mode="pdf_ocr",
        n_compounds=1,
        n_measurements=1,
        target_assay="WIZ",
        off_target_assay="ZBTB7A",
        stage_peak_rss_mb={"ocr": 400.0},
        stage_timings_s={"ocr": 12.0},
    )
    return Bundle(
        root=str(tmp_path),
        manifest=manifest,
        compounds=[compound],
        measurements=[measurement],
        anomalies=[],
    )


def test_the_app_module_imports_without_running_a_pipeline():
    """It must be import-safe: Streamlit re-executes the script on every widget."""
    module = importlib.import_module("app.streamlit_app")
    assert hasattr(module, "main")


def test_sar_frame_shows_the_decoded_interval_beside_the_letter(tmp_path):
    """PRD §14.1 — the letter AND its decoded interval, never one without the other."""
    frame = app._sar_frame(_bundle(tmp_path))
    assert len(frame) == 1
    cell = frame.iloc[0]["WIZ EC50 (uM)"]
    assert cell.startswith("D")
    assert "10" in cell and "nM" in cell


def test_sar_frame_carries_provenance_pages_and_confidence(tmp_path):
    frame = app._sar_frame(_bundle(tmp_path))
    row = frame.iloc[0]
    assert row["confidence"] == "AGREE_FULL"
    assert "63" in row["pages"]
    assert row["InChIKey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_page_ranges_parse():
    assert app._parse_ranges("61-88,182-187") == [(61, 88), (182, 187)]
    assert app._parse_ranges("63") == [(63, 63)]
    assert app._parse_ranges("") is None
    assert app._parse_ranges("garbage") is None


def test_interval_rendering_is_explicit_about_direction(tmp_path):
    bundle = _bundle(tmp_path)
    assert app._interval(bundle.measurements[0]).startswith("<")


def test_corrections_replay_onto_a_freshly_read_bundle(tmp_path, monkeypatch):
    """PRD R13.4/R13.5 — the edit survives a rerun and re-ranking follows it."""
    from sarmine.review.edits import CorrectionStore

    bundle = _bundle(tmp_path)
    store = CorrectionStore()
    store.correct_compound(bundle.compounds[0], "smiles_final", "CCC", note="redrawn")

    monkeypatch.setitem(app.st.session_state, "corrections", store)
    corrected = app._apply_corrections(_bundle(tmp_path))

    assert corrected.compounds[0].smiles_final == "CCC"
    # The audit trail must not grow just because the page re-rendered.
    assert len(store.entries) == 1
    assert store.to_rows()[0]["original"] == "CCO"


def test_percentage_formatting_handles_a_missing_score():
    assert app._pct(None) == "—"
    assert app._pct(0.8532) == "85.3%"
