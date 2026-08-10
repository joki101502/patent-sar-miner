"""Part 13.3 — scoring a run against a committed gold set (PRD §20.1, AC-10.*).

The headline number is END-TO-END TRIPLET accuracy (compound + assay + value all
correct), not per-component accuracy (PRD AC-10.2). BioMiner reports F1 = 0.32 on
full bioactivity triplets while its component tasks look far better; reporting
per-component would be misleading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sarmine.artifacts.schema import Compound, Measurement, Provenance, RunManifest
from sarmine.artifacts.writer import write_bundle
from sarmine.evaluate import calibrate_threshold, evaluate_run, format_report

GOLD_WO = Path(__file__).resolve().parent.parent / "gold" / "WO2024097932A1.gold.json"
GOLD_US = Path(__file__).resolve().parent.parent / "gold" / "US20250368620A1.gold.json"

ASSAYS = ("HbF Induction (%)", "WIZ EC50 (uM)", "ZBTB7A EC50 (uM)")


def prov(page_no: int = 186) -> Provenance:
    return Provenance(
        page_no=page_no,
        bbox=(0, 0, 10, 10),
        raster_width=2477,
        raster_height=3505,
        crop_path=f"crops/p{page_no:03d}_activity_0.png",
        source="structured",
        extractor="tesseract@5.5.3",
    )


def build_bundle(tmp_path: Path, *, activity: dict, structures: dict | None = None,
                 run_id: str = "RUN1") -> Path:
    """Materialize a bundle whose contents we control, so accuracy is predictable."""
    structures = structures or {}
    compounds, measurements = [], []
    for num, row in activity.items():
        cid = f"WO2024097932A1:{num}"
        compounds.append(
            Compound(
                compound_id=cid,
                compound_local_id=str(num),
                compound_number=int(num),
                inchikey_full=structures.get(str(num)),
                smiles_from_name="C" if structures.get(str(num)) else None,
                crosscheck_tier="AGREE_FULL" if structures.get(str(num)) else "NONE",
                rdkit_version="2026.03.5",
                provenance=[prov(63)],
            )
        )
        for assay, letter in row.items():
            if letter is None:
                continue
            measurements.append(
                Measurement(
                    measurement_id=f"{cid}:{assay}",
                    compound_id=cid,
                    assay_group_key=f"WO2024097932A1::{assay}",
                    assay_name_raw=assay,
                    published_type=assay,
                    published_value=letter,
                    standard_type="EC50",
                    standard_relation="<",
                    bin_label_raw=letter,
                    is_censored=True,
                    provenance=prov(),
                )
            )
    manifest = RunManifest(
        pubnum="WO2024097932A1",
        run_id=run_id,
        created_at="2026-08-09T00:00:00Z",
        source_mode="structured",
        n_compounds=len(compounds),
        n_measurements=len(measurements),
        target_assay="WIZ",
        off_target_assay="ZBTB7A",
    )
    return write_bundle(manifest, compounds, measurements, [], tmp_path)


@pytest.fixture
def gold_activity() -> dict:
    return json.loads(GOLD_WO.read_text("utf-8"))["activity"]


def test_gold_sets_are_committed_and_well_formed():
    """PRD §20.1 — both gold sets ship with the repo."""
    wo = json.loads(GOLD_WO.read_text("utf-8"))
    assert wo["counts"]["n_compounds"] == 54
    assert wo["counts"]["n_activity_cells"] == 162
    assert len(wo["counts"]["blank_hbf_compounds"]) == 11
    assert wo["counts"]["max_selectivity_compounds"] == ["10", "16", "20", "52"]
    us = json.loads(GOLD_US.read_text("utf-8"))
    assert us["assays"][0]["standard_type"] == "DC50"
    assert us["split_header"]["expected_reconstruction"][-1] == "HiBiT DC50 (nM)"


def test_perfect_run_scores_one(tmp_path, gold_activity):
    """A bundle that exactly reproduces the gold set scores 100% on every axis."""
    run = build_bundle(tmp_path, activity=gold_activity)
    report = evaluate_run(run, GOLD_WO)

    assert report["activity_cells"]["total"] == 162
    assert report["activity_cells"]["correct"] == 151  # 162 minus the 11 blanks
    assert report["activity_cells"]["accuracy"] == pytest.approx(1.0)
    assert report["triplet"]["accuracy"] == pytest.approx(1.0)


def test_blank_cells_are_not_credited_as_correct_values(tmp_path, gold_activity):
    """PRD EC-7 — a blank is not a value. Emitting one where gold is blank is wrong."""
    corrupted = {k: dict(v) for k, v in gold_activity.items()}
    corrupted["33"]["HbF Induction (%)"] = "C"  # gold has this blank
    run = build_bundle(tmp_path, activity=corrupted)
    report = evaluate_run(run, GOLD_WO)

    assert report["activity_cells"]["spurious"] == 1
    assert report["activity_cells"]["accuracy"] < 1.0


def test_missing_and_wrong_cells_are_scored_separately(tmp_path, gold_activity):
    corrupted = {k: dict(v) for k, v in gold_activity.items()}
    corrupted["1"]["WIZ EC50 (uM)"] = "F"      # gold "E" -> wrong
    corrupted["2"]["WIZ EC50 (uM)"] = None      # gold "D" -> missing
    run = build_bundle(tmp_path, activity=corrupted)
    report = evaluate_run(run, GOLD_WO)

    assert report["activity_cells"]["wrong"] == 1
    assert report["activity_cells"]["missing"] == 1
    assert report["activity_cells"]["correct"] == 149


def test_triplet_accuracy_requires_compound_assay_and_value_together(tmp_path, gold_activity):
    """PRD AC-10.2 — the headline number is the full triplet, not per-component."""
    only_two = {k: gold_activity[k] for k in ("1", "2")}
    run = build_bundle(tmp_path, activity=only_two)
    report = evaluate_run(run, GOLD_WO)

    assert report["triplet"]["correct"] == 6
    assert report["triplet"]["gold_total"] == 151
    assert report["triplet"]["recall"] == pytest.approx(6 / 151)
    assert report["triplet"]["accuracy"] < 0.05


def test_structure_accuracy_is_exact_inchikey_against_gold(tmp_path, gold_activity):
    """PRD AC-10.1/AC-10.3 — structure accuracy is exact InChIKey match."""
    run = build_bundle(
        tmp_path,
        activity=gold_activity,
        structures={"5": "WZPDSZGYLXZFEK-UHFFFAOYSA-N"},
    )
    report = evaluate_run(run, GOLD_WO)

    assert report["structures"]["scored"] == 1
    assert report["structures"]["correct"] == 1
    assert report["structures"]["accuracy"] == pytest.approx(1.0)


def test_structure_accuracy_reports_its_own_gold_provenance(tmp_path, gold_activity):
    """The structure gold is thin and partly derived, not hand-drawn. Saying so is
    part of the deliverable (PRD R14.5) — the report must not imply otherwise."""
    run = build_bundle(tmp_path, activity=gold_activity, structures={"5": "WRONGKEYAAAAAA-UHFFFAOYSA-N"})
    report = evaluate_run(run, GOLD_WO)

    assert report["structures"]["correct"] == 0
    assert "hand_checked" in report["structures"]["gold_verification"]


def test_skeleton_match_is_reported_separately_from_exact_match(tmp_path, gold_activity):
    """PRD R9.22/EC-19 — stereo-only disagreement is a different failure from a
    wrong skeleton, because this chemotype's glutarimide centre is drawn flat
    about half the time."""
    run = build_bundle(
        tmp_path, activity=gold_activity, structures={"5": "WZPDSZGYLXZFEK-QFIPXVFZSA-N"}
    )
    report = evaluate_run(run, GOLD_WO)

    assert report["structures"]["correct"] == 0
    assert report["structures"]["skeleton_correct"] == 1


def test_join_accuracy_is_reported(tmp_path, gold_activity):
    """PRD AC-10.1 — evaluate reports structure, activity-cell AND join accuracy."""
    run = build_bundle(tmp_path, activity=gold_activity)
    report = evaluate_run(run, GOLD_WO)

    assert "join" in report
    assert report["join"]["compounds_matched"] == 54
    assert report["join"]["accuracy"] == pytest.approx(1.0)


def test_format_report_leads_with_the_triplet_number(tmp_path, gold_activity):
    run = build_bundle(tmp_path, activity=gold_activity)
    text = format_report(evaluate_run(run, GOLD_WO))

    assert "triplet" in text.lower()
    headline = text.strip().splitlines()
    assert any("end-to-end" in line.lower() for line in headline[:6])


def test_calibrate_sweeps_the_threshold_and_returns_the_chosen_value(tmp_path, gold_activity):
    """PRD R13.2 — tau must be CALIBRATED against the gold set, never guessed."""
    run = build_bundle(
        tmp_path, activity=gold_activity, structures={"5": "WZPDSZGYLXZFEK-UHFFFAOYSA-N"}
    )
    result = calibrate_threshold(run, GOLD_WO)

    assert "chosen_threshold" in result
    assert 0.0 <= result["chosen_threshold"] <= 1.0
    assert len(result["sweep"]) > 1
    for point in result["sweep"]:
        assert {"threshold", "precision", "recall", "n_flagged"} <= set(point)
