"""Part 1 — data model and artifact bundle (PRD §15, Plan Part 1.6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sarmine.artifacts.schema import (
    Compound,
    DocumentAnomaly,
    Measurement,
    Provenance,
    RunManifest,
)
from sarmine.artifacts.writer import read_bundle, write_bundle


def make_provenance(**overrides) -> Provenance:
    kwargs = dict(
        page_no=63,
        bbox=(350, 211, 1394, 715),
        raster_width=2480,
        raster_height=3508,
        crop_path="crops/p063_structure_0.png",
        source="pdf_ocr",
        extractor="molscribe@1.1.1",
        rotation_applied=90,
    )
    kwargs.update(overrides)
    return Provenance(**kwargs)


def test_provenance_carries_page_bbox_and_crop():
    p = make_provenance()
    assert p.page_no == 63
    assert p.bbox == (350, 211, 1394, 715)
    assert p.crop_path.endswith(".png")
    assert p.rotation_applied == 90


def test_published_value_is_always_a_string():
    """PRD R10.1 — ">10,000", "5.6 ± 0.3", "A", "n.d." are all real cell contents."""
    m = Measurement(
        measurement_id="m1",
        compound_id="WO2024097932A1:5",
        assay_group_key="WO2024097932A1::WIZ EC50 (uM)",
        assay_name_raw="WIZ EC50 (uM)",
        published_type="WIZ EC50 (uM)",
        published_value="A",
        standard_type="EC50",
        standard_relation="=",
        provenance=make_provenance(),
    )
    assert isinstance(m.published_value, str)

    coerced = Measurement(
        measurement_id="m2",
        compound_id="WO2024097932A1:5",
        assay_group_key="k",
        assay_name_raw="h",
        published_type="h",
        published_value=10000,  # must not stay an int
        standard_type="EC50",
        standard_relation="=",
        provenance=make_provenance(),
    )
    assert coerced.published_value == "10000"
    assert isinstance(coerced.published_value, str)


def test_pchembl_and_pdc50_are_mutually_exclusive():
    """PRD R10.4 — pDC50 must never silently pool into pchembl_value."""
    with pytest.raises(ValidationError):
        Measurement(
            measurement_id="m3",
            compound_id="c",
            assay_group_key="k",
            assay_name_raw="h",
            published_type="h",
            published_value="1.0",
            standard_type="DC50",
            standard_relation="=",
            pchembl_value=8.0,
            pdc50_value=8.0,
            provenance=make_provenance(),
        )


def test_dmax_requires_a_paired_dc50_on_the_same_row():
    """PRD R10.4 — Dmax is a paired attribute of DC50, never standalone."""
    with pytest.raises(ValidationError):
        Measurement(
            measurement_id="m4",
            compound_id="c",
            assay_group_key="k",
            assay_name_raw="Dmax (%)",
            published_type="Dmax (%)",
            published_value="95",
            standard_type="Dmax",
            standard_relation="=",
            dmax_pct=95.0,
            standard_value=None,
            provenance=make_provenance(),
        )

    ok = Measurement(
        measurement_id="m5",
        compound_id="c",
        assay_group_key="k",
        assay_name_raw="HiBiT DC50 (nM)",
        published_type="HiBiT DC50 (nM)",
        published_value="7.6",
        standard_type="DC50",
        standard_relation="=",
        standard_value=7.6,
        standard_units="nM",
        dmax_pct=95.0,
        provenance=make_provenance(),
    )
    assert ok.dmax_pct == 95.0


def test_compound_records_rdkit_version():
    """PRD R9.20 — canonicalization changes between RDKit releases."""
    c = Compound(
        compound_id="WO2024097932A1:5",
        compound_local_id="5",
        compound_number=5,
        crosscheck_tier="AGREE_FULL",
        rdkit_version="2026.03.5",
    )
    assert c.rdkit_version == "2026.03.5"
    assert c.markush_detected is False
    assert c.provenance == []


def test_inchikey_skeleton_is_derived_from_full_key():
    c = Compound(
        compound_id="c",
        compound_local_id="5",
        inchikey_full="WZPDSZGYLXZFEK-UHFFFAOYSA-N",
        crosscheck_tier="AGREE_FULL",
        rdkit_version="2026.03.5",
    )
    assert c.inchikey_skeleton == "WZPDSZGYLXZFEK"


def test_bundle_round_trips_identically(tmp_path):
    """Plan Part 1.6 / PRD §20.3 — written then re-read reproduces identical objects."""
    prov = make_provenance()
    compound = Compound(
        compound_id="WO2024097932A1:5",
        compound_local_id="5",
        compound_number=5,
        smiles_from_name="O=C1NC(CCC1N1C(C2=CC=CC(=C2C1=O)N)=O)=O",
        smiles_from_image="COc1ccc(F)cc1",
        smiles_final="O=C1NC(CCC1N1C(C2=CC=CC(=C2C1=O)N)=O)=O",
        structure_source="name+image",
        inchikey_full="WZPDSZGYLXZFEK-UHFFFAOYSA-N",
        crosscheck_tier="AGREE_FULL",
        opsin_status="SUCCESS",
        rdkit_version="2026.03.5",
        provenance=[prov],
        rank_rationale=["WIZ D (top bin)"],
    )
    measurement = Measurement(
        measurement_id="WO2024097932A1:5:WIZ",
        compound_id=compound.compound_id,
        assay_group_key="WO2024097932A1::WIZ EC50 (uM)",
        assay_name_raw="WIZ EC50 (uM)",
        target_raw="WIZ",
        published_type="WIZ EC50 (uM)",
        published_value="D",
        standard_type="EC50",
        standard_relation="<",
        standard_value=None,
        standard_units="nM",
        is_censored=True,
        censor_direction="upper_bound",
        bin_label_raw="D",
        bin_upper_nM=10.0,
        bin_score=3,
        provenance=prov,
    )
    anomaly = DocumentAnomaly(
        kind="legend_contradiction",
        severity="warning",
        message="HbF level A defined as 66-100% but restated as 67-100%",
    )
    manifest = RunManifest(
        pubnum="WO2024097932A1",
        run_id="20260809T000000",
        created_at="2026-08-09T00:00:00Z",
        source_mode="structured",
        n_pages=223,
        n_compounds=1,
        n_measurements=1,
        target_assay="WIZ",
        off_target_assay="ZBTB7A",
        anomalies=[anomaly],
        stage_timings_s={"resolve": 1.0},
        stage_peak_rss_mb={"resolve": 120.0},
        versions={"rdkit": "2026.03.5"},
    )

    out = write_bundle(manifest, [compound], [measurement], [anomaly], tmp_path)
    bundle = read_bundle(out)

    assert bundle.manifest == manifest
    assert bundle.compounds == [compound]
    assert bundle.measurements == [measurement]
    assert bundle.anomalies == [anomaly]


def test_bundle_layout_matches_prd(tmp_path):
    """PRD §15.5 — artifacts/{pubnum}/{run_id}/..."""
    manifest = RunManifest(
        pubnum="WO2024097932A1",
        run_id="RUN1",
        created_at="2026-08-09T00:00:00Z",
        source_mode="pdf_ocr",
        n_pages=223,
        n_compounds=0,
        n_measurements=0,
        target_assay="WIZ",
    )
    out = write_bundle(manifest, [], [], [], tmp_path)
    assert out == tmp_path / "WO2024097932A1" / "RUN1"
    for name in ("manifest.json", "compounds.json", "measurements.json", "anomalies.json"):
        assert (out / name).is_file()
    for sub in ("crops", "svg", "source"):
        assert (out / sub).is_dir()
