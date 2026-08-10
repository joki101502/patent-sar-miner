"""Part 10 — review queue, overlay rendering and correction audit trail.

Covers PRD §13.2 (every trigger, with its normative priority), R13.1, R13.2,
R13.7, R9.15 and acceptance criteria AC-7.1–AC-7.5, AC-8.1–AC-8.2.
"""

from __future__ import annotations

import pytest
from PIL import Image

from sarmine.artifacts.schema import Compound, DocumentAnomaly, Measurement, Provenance
from sarmine.review.edits import AuditEntry, CorrectionStore, missing_provenance
from sarmine.review.queue import ReviewItem, build_queue, sort_queue
from sarmine.review.render import (
    render_crop_with_bbox,
    render_structure_png,
    render_structure_svg,
)

VALID_SMILES = "O=C1NC(=O)CCC1N1C(=O)c2ccccc2C1=O"
INVALID_SMILES = "C1CC(=O"


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


def make_compound(**overrides) -> Compound:
    """A clean compound: nothing about it should reach the review queue."""
    kwargs = dict(
        compound_id="WO2024097932A1:5",
        compound_local_id="5",
        compound_number=5,
        smiles_from_name=VALID_SMILES,
        smiles_from_image=VALID_SMILES,
        smiles_final=VALID_SMILES,
        structure_source="name+image",
        inchikey_full="WZPDSZGYLXZFEK-UHFFFAOYSA-N",
        inchikey_from_name="WZPDSZGYLXZFEK-UHFFFAOYSA-N",
        inchikey_from_image="WZPDSZGYLXZFEK-UHFFFAOYSA-N",
        crosscheck_tier="AGREE_FULL",
        ocsr_confidence_molecule=0.97,
        ocsr_confidence_min_atom=0.95,
        ocsr_confidence_min_bond=0.94,
        opsin_status="SUCCESS",
        markush_detected=False,
        heavy_atoms=30,
        rdkit_version="2026.03.5",
        provenance=[make_provenance()],
    )
    kwargs.update(overrides)
    return Compound(**kwargs)


def make_measurement(**overrides) -> Measurement:
    """A clean measurement: nothing about it should reach the review queue."""
    kwargs = dict(
        measurement_id="WO2024097932A1:5:WIZ",
        compound_id="WO2024097932A1:5",
        assay_group_key="WO2024097932A1::WIZ EC50 (uM)",
        assay_name_raw="WIZ EC50 (uM)",
        target_raw="WIZ",
        published_type="WIZ EC50 (uM)",
        published_value="D",
        published_units="uM",
        standard_type="EC50",
        standard_relation="<",
        standard_units="nM",
        bin_label_raw="D",
        bin_definition="D: < 0.01 uM",
        bin_upper_nM=10.0,
        bin_score=3,
        is_censored=True,
        censor_direction="upper_bound",
        provenance=make_provenance(),
    )
    kwargs.update(overrides)
    return Measurement(**kwargs)


def triggers(items: list[ReviewItem]) -> set[str]:
    return {item.trigger for item in items}


def find(items: list[ReviewItem], trigger: str) -> ReviewItem:
    matches = [item for item in items if item.trigger == trigger]
    assert matches, f"no review item with trigger {trigger!r}; got {sorted(triggers(items))}"
    return matches[0]


# --- PRD §13.2 triggers -----------------------------------------------------


def test_opsin_parse_failure_is_high_priority():
    """PRD §13.2 / AC-7.1 — OPSIN parse failure after homoglyph repair."""
    compound = make_compound(
        opsin_status="FAILURE",
        smiles_from_name=None,
        homoglyph_repair_applied="Cyrillic \u0430 -> a",
    )
    items = build_queue([compound], [])
    item = find(items, "opsin_parse_failure")
    assert item.priority == "high"
    assert item.compound_id == compound.compound_id
    assert item.reason


def test_crosscheck_conflict_is_high_priority():
    """PRD §13.2 / §9 — skeletons differ; never auto-pick a winner."""
    compound = make_compound(
        crosscheck_tier="CONFLICT",
        inchikey_from_image="ZZZZZZZZZZZZZZ-UHFFFAOYSA-N",
    )
    item = find(build_queue([compound], []), "crosscheck_conflict")
    assert item.priority == "high"
    assert item.crosscheck_tier == "CONFLICT"
    assert item.inchikey_from_name != item.inchikey_from_image


def test_unsanitizable_smiles_in_either_channel_is_high_priority():
    """PRD §13.2 — either channel's SMILES failing RDKit sanitization."""
    from_image = make_compound(smiles_from_image=INVALID_SMILES)
    item = find(build_queue([from_image], []), "smiles_sanitization_failure")
    assert item.priority == "high"
    assert "image" in item.reason

    from_name = make_compound(smiles_from_name=INVALID_SMILES)
    item = find(build_queue([from_name], []), "smiles_sanitization_failure")
    assert item.priority == "high"
    assert "name" in item.reason


def test_unreadable_compound_number_is_high_priority():
    """PRD §13.2 — compound number unreadable."""
    compound = make_compound(compound_number=None)
    item = find(build_queue([compound], []), "compound_number_unreadable")
    assert item.priority == "high"


def test_compound_number_breaking_monotonicity_is_high_priority():
    """PRD §13.2 — a number that goes backwards means a misread or a bad stitch."""
    first = make_compound(compound_id="p:1", compound_local_id="1", compound_number=1)
    second = make_compound(compound_id="p:9", compound_local_id="9", compound_number=9)
    third = make_compound(compound_id="p:3", compound_local_id="3", compound_number=3)

    items = build_queue([first, second, third], [])
    item = find(items, "compound_number_monotonicity")
    assert item.priority == "high"
    assert item.compound_id == "p:3"
    assert [i.compound_id for i in items if i.trigger == "compound_number_monotonicity"] == ["p:3"]


def test_gaps_in_compound_numbering_do_not_break_monotonicity():
    """1, 5, 9 is increasing; only a decrease is a misread."""
    compounds = [
        make_compound(compound_id=f"p:{n}", compound_local_id=str(n), compound_number=n)
        for n in (1, 5, 9)
    ]
    assert "compound_number_monotonicity" not in triggers(build_queue(compounds, []))


def test_blank_activity_cell_is_high_priority():
    """PRD §13.2 — activity cell blank where the table implies a value."""
    measurement = make_measurement(
        published_value="",
        bin_label_raw=None,
        bin_definition=None,
        bin_upper_nM=None,
        bin_score=None,
    )
    item = find(build_queue([make_compound()], [measurement]), "activity_cell_blank")
    assert item.priority == "high"
    assert item.measurement_id == measurement.measurement_id
    assert item.page_no == measurement.provenance.page_no


def test_missing_assay_column_is_a_blank_activity_cell():
    """PRD §13.2 — the table implies a value per assay; a missing one is blank."""
    compound = make_compound()
    items = build_queue(
        [compound], [make_measurement()], expected_assays_per_compound=2
    )
    item = find(items, "activity_cell_blank")
    assert item.priority == "high"
    assert item.compound_id == compound.compound_id


def test_bin_letter_outside_the_legend_is_high_priority():
    """PRD §13.2 — a letter the resolved legend does not define."""
    measurement = make_measurement(
        published_value="Z",
        bin_label_raw="Z",
        bin_definition=None,
        bin_upper_nM=None,
        bin_score=None,
    )
    item = find(build_queue([make_compound()], [measurement]), "bin_letter_outside_legend")
    assert item.priority == "high"
    assert "Z" in item.reason


def test_unrecognized_units_are_high_priority():
    """PRD §13.2 — units implicit or unrecognized."""
    measurement = make_measurement(
        published_value="5.6",
        published_units=None,
        standard_value=5.6,
        standard_units=None,
        bin_label_raw=None,
        bin_definition=None,
        bin_upper_nM=None,
        bin_score=None,
    )
    items = build_queue([make_compound()], [measurement])
    item = find(items, "units_unrecognized")
    assert item.priority == "high"


def test_units_implicit_in_the_header_are_not_queued_once_standardized():
    """A resolved unit is not a review trigger, however it was resolved."""
    measurement = make_measurement(
        published_value="5.6",
        published_units=None,
        standard_value=5600.0,
        standard_units="nM",
        bin_label_raw=None,
        bin_definition=None,
        bin_upper_nM=None,
        bin_score=None,
    )
    assert "units_unrecognized" not in triggers(build_queue([make_compound()], [measurement]))


def test_agree_skeleton_is_medium_priority():
    """PRD §13.2 — stereochemistry disagreement between the two channels."""
    compound = make_compound(
        crosscheck_tier="AGREE_SKELETON",
        inchikey_from_image="WZPDSZGYLXZFEK-ZZZZZZZZSA-N",
    )
    item = find(build_queue([compound], []), "crosscheck_agree_skeleton")
    assert item.priority == "medium"


def test_low_minimum_atom_confidence_is_queued_despite_a_high_molecule_mean():
    """PRD R13.1 — one wrong atom ruins the structure while barely moving the mean."""
    compound = make_compound(
        ocsr_confidence_molecule=0.95,
        ocsr_confidence_min_atom=0.60,
        ocsr_confidence_min_bond=0.93,
    )
    item = find(build_queue([compound], []), "ocsr_low_confidence")
    assert item.priority == "medium"
    assert "0.6" in item.reason


def test_low_minimum_bond_confidence_is_queued():
    """PRD R13.1 — the gate is on minimum atom *or* bond confidence."""
    compound = make_compound(
        ocsr_confidence_molecule=0.99,
        ocsr_confidence_min_atom=0.97,
        ocsr_confidence_min_bond=0.42,
    )
    item = find(build_queue([compound], []), "ocsr_low_confidence")
    assert item.priority == "medium"


def test_ocsr_threshold_defaults_to_config_and_is_overridable():
    """PRD R13.2 — τ comes from config, pending gold-set calibration."""
    from sarmine.config import get_config

    tau = get_config().ocsr_conf_threshold
    borderline = make_compound(
        ocsr_confidence_min_atom=tau - 0.01, ocsr_confidence_min_bond=0.99
    )
    assert "ocsr_low_confidence" in triggers(build_queue([borderline], []))
    assert "ocsr_low_confidence" not in triggers(
        build_queue([borderline], [], ocsr_conf_threshold=0.10)
    )


def test_detector_disagreement_anomaly_is_medium_priority():
    """PRD §13.2 — morphology detector and TATR disagree on the cell grid."""
    anomaly = DocumentAnomaly(
        kind="detector_disagreement",
        severity="warning",
        message="Morphology found 9 columns, TATR found 7 on page 63",
        provenance=make_provenance(),
    )
    item = find(build_queue([], [], [anomaly]), "detector_disagreement")
    assert item.priority == "medium"
    assert item.page_no == 63
    assert anomaly.message in item.reason


def test_rotation_uncertain_anomaly_is_medium_priority():
    """PRD §13.2 — page rotation applied at a low verification score."""
    anomaly = DocumentAnomaly(
        kind="rotation_uncertain",
        severity="warning",
        message="Rotated page 63 by 90 deg at verification score 0.11",
    )
    item = find(build_queue([], [], [anomaly]), "rotation_uncertain")
    assert item.priority == "medium"


def test_markush_is_informational_and_never_auto_resolved():
    """PRD §13.2 — MolClassifier says Markush: informational, never auto-resolved."""
    compound = make_compound(markush_detected=True)
    item = find(build_queue([compound], []), "markush_structure")
    assert item.priority == "info"


def test_more_than_seventy_heavy_atoms_is_medium_priority():
    """PRD §13.2 / EC-13 — big molecules are where OCSR degrades."""
    compound = make_compound(heavy_atoms=71)
    item = find(build_queue([compound], []), "heavy_atom_count")
    assert item.priority == "medium"

    assert "heavy_atom_count" not in triggers(build_queue([make_compound(heavy_atoms=70)], []))
    assert "heavy_atom_count" in triggers(
        build_queue([make_compound(heavy_atoms=40)], [], heavy_atom_threshold=30)
    )


# --- what must NOT enter the queue -----------------------------------------


def test_single_source_with_good_confidence_produces_no_review_item():
    """PRD R9.15 — absence of a cross-check is NEUTRAL, not suspicious.

    Treating single-source rows as low-confidence floods the queue with the
    majority case (PatCID: only 31.2% of compounds appear in >1 source).
    """
    compound = make_compound(
        crosscheck_tier="SINGLE_SOURCE",
        smiles_from_image=None,
        inchikey_from_image=None,
        structure_source="name",
    )
    assert build_queue([compound], [make_measurement()]) == []


def test_a_clean_extraction_produces_an_empty_queue():
    assert build_queue([make_compound()], [make_measurement()], []) == []


# --- prioritization ---------------------------------------------------------


def test_sort_queue_orders_high_then_medium_then_info_and_is_stable():
    """PRD §13.2 — the queue is worked top-down."""
    compounds = [
        make_compound(compound_id="p:1", compound_local_id="1", compound_number=1,
                      markush_detected=True),
        make_compound(compound_id="p:2", compound_local_id="2", compound_number=2,
                      heavy_atoms=90),
        make_compound(compound_id="p:3", compound_local_id="3", compound_number=3,
                      crosscheck_tier="CONFLICT"),
        make_compound(compound_id="p:4", compound_local_id="4", compound_number=4,
                      heavy_atoms=95),
    ]
    ordered = sort_queue(build_queue(compounds, []))
    assert [i.priority for i in ordered] == ["high", "medium", "medium", "info"]
    assert [i.compound_id for i in ordered] == ["p:3", "p:2", "p:4", "p:1"]


# --- overlay rendering (PRD R13.3, R13.7, AC-7.2, AC-8.2) -------------------

RED = (214, 39, 40)


def write_page(path, size=(400, 300)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def is_red(pixel) -> bool:
    return all(abs(a - b) <= 12 for a, b in zip(pixel, RED))


def any_red(image: Image.Image) -> bool:
    return any(is_red(px) for px in image.convert("RGB").getdata())


def test_bbox_is_drawn_server_side_on_a_full_page(tmp_path):
    """PRD R13.7 / AC-7.2 — PIL draws the overlay; no JS canvas is required."""
    page = write_page(tmp_path / "p-063.png")
    out = render_crop_with_bbox(page, (100, 80, 300, 220), tmp_path / "out" / "overlay.png")

    assert out.is_file()
    image = Image.open(out).convert("RGB")
    assert image.size == (400, 300)
    assert is_red(image.getpixel((100, 80)))
    assert is_red(image.getpixel((300, 220)))
    assert image.getpixel((200, 150)) == (255, 255, 255)


def test_bbox_is_translated_into_crop_coordinates(tmp_path):
    """PRD §15.3 — the bbox lives in page pixel space; the crop is a sub-region."""
    crop = write_page(tmp_path / "crop.png", size=(200, 150))
    out = render_crop_with_bbox(
        crop,
        (350, 211, 450, 261),
        tmp_path / "overlay.png",
        crop_origin=(300, 200),
    )

    image = Image.open(out).convert("RGB")
    assert is_red(image.getpixel((50, 11)))
    assert is_red(image.getpixel((150, 61)))
    assert image.getpixel((100, 100)) == (255, 255, 255)


def test_out_of_bounds_bbox_is_clamped_not_raised(tmp_path):
    page = write_page(tmp_path / "p.png", size=(120, 90))
    out = render_crop_with_bbox(page, (-500, -500, 9999, 9999), tmp_path / "clamped.png")

    image = Image.open(out).convert("RGB")
    assert image.size == (120, 90)
    assert any_red(image)


def test_overlay_is_labelled_with_the_page_number(tmp_path):
    """AC-8.2 — the crop opens with its bbox drawn and the page labelled."""
    page = write_page(tmp_path / "p.png")
    plain = render_crop_with_bbox(page, (100, 80, 300, 220), tmp_path / "plain.png")
    labelled = render_crop_with_bbox(
        page, (100, 80, 300, 220), tmp_path / "labelled.png", label="Page 63"
    )

    plain_ink = sum(1 for px in Image.open(plain).convert("RGB").getdata() if px != (255, 255, 255))
    labelled_ink = sum(
        1 for px in Image.open(labelled).convert("RGB").getdata() if px != (255, 255, 255)
    )
    assert labelled_ink > plain_ink


def test_a_missing_bbox_still_renders_the_crop(tmp_path):
    page = write_page(tmp_path / "p.png", size=(64, 48))
    out = render_crop_with_bbox(page, None, tmp_path / "nobox.png")

    image = Image.open(out).convert("RGB")
    assert image.size == (64, 48)
    assert not any_red(image)


def test_render_structure_svg_and_png(tmp_path):
    """PRD R13.3 — the RDKit-rendered structure sits next to the source crop."""
    svg = render_structure_svg(VALID_SMILES, tmp_path / "svg" / "c5.svg")
    assert svg is not None and svg.is_file()
    text = svg.read_text("utf-8")
    assert "<svg" in text and "</svg>" in text

    png = render_structure_png(VALID_SMILES, tmp_path / "png" / "c5.png", size=(200, 180))
    assert png is not None and png.is_file()
    assert Image.open(png).size == (200, 180)


def test_render_structure_returns_none_for_missing_or_invalid_smiles(tmp_path):
    """A failed extraction must render as 'nothing', never as an exception."""
    assert render_structure_svg(None, tmp_path / "none.svg") is None
    assert render_structure_svg(INVALID_SMILES, tmp_path / "bad.svg") is None
    assert render_structure_png(None, tmp_path / "none.png") is None
    assert render_structure_png(INVALID_SMILES, tmp_path / "bad.png") is None
    assert not (tmp_path / "bad.svg").exists()
    assert not (tmp_path / "bad.png").exists()


# --- corrections and the audit trail (PRD R13.4–R13.6, AC-7.3–AC-7.5) -------

ETHANOL = "CCO"
ETHANOL_KEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_correcting_a_smiles_returns_a_copy_and_retains_the_original():
    """AC-7.3 — the edit updates the compound, retains the original, audits it."""
    store = CorrectionStore()
    compound = make_compound()

    corrected = store.correct_compound(compound, "smiles_final", ETHANOL)

    assert corrected is not compound
    assert corrected.smiles_final == ETHANOL
    assert compound.smiles_final == VALID_SMILES  # the caller's object is untouched
    assert store.original("compound", compound.compound_id, "smiles_final") == VALID_SMILES

    (entry,) = store.entries
    assert isinstance(entry, AuditEntry)
    assert (entry.target_kind, entry.target_id, entry.field) == (
        "compound",
        compound.compound_id,
        "smiles_final",
    )
    assert entry.original == VALID_SMILES
    assert entry.corrected == ETHANOL
    assert entry.timestamp


def test_correcting_the_same_field_twice_keeps_the_first_original():
    """PRD R13.4 — the ORIGINAL extraction is always retained, not the last edit."""
    store = CorrectionStore()
    compound = make_compound()

    once = store.correct_compound(compound, "smiles_final", ETHANOL)
    twice = store.correct_compound(once, "smiles_final", "CCC")

    assert twice.smiles_final == "CCC"
    assert store.original("compound", compound.compound_id, "smiles_final") == VALID_SMILES
    assert len(store.entries) == 2


def test_corrected_smiles_recomputes_the_identity_keys():
    """The join (R9.16) and ranking must not run on stale InChIKeys."""
    store = CorrectionStore()
    corrected = store.correct_compound(make_compound(), "smiles_final", ETHANOL)

    assert corrected.inchikey_full == ETHANOL_KEY
    assert corrected.inchikey_skeleton == ETHANOL_KEY[:14]


def test_correcting_a_channel_smiles_updates_that_channel_key_and_the_final():
    store = CorrectionStore()
    compound = make_compound()

    corrected = store.correct_compound(compound, "smiles_from_name", ETHANOL)

    assert corrected.inchikey_from_name == ETHANOL_KEY
    assert corrected.smiles_final == ETHANOL
    assert corrected.inchikey_full == ETHANOL_KEY
    assert corrected.smiles_from_image == compound.smiles_from_image


def test_an_invalid_corrected_smiles_clears_the_identity_keys():
    """A bad paste must re-enter the review queue, not keep stale keys."""
    store = CorrectionStore()
    corrected = store.correct_compound(make_compound(), "smiles_final", INVALID_SMILES)

    assert corrected.inchikey_full is None
    assert corrected.inchikey_skeleton is None
    assert "smiles_sanitization_failure" in triggers(build_queue([corrected], []))


def test_correcting_a_compound_number_is_audited():
    store = CorrectionStore()
    corrected = store.correct_compound(
        make_compound(compound_number=None), "compound_number", 5, note="read from the crop"
    )

    assert corrected.compound_number == 5
    (entry,) = store.entries
    assert entry.original is None
    assert entry.corrected == "5"
    assert entry.note == "read from the crop"


def test_correcting_a_measurement_returns_a_copy_and_audits_it():
    """PRD R13.4 — 'correct an activity value'."""
    store = CorrectionStore()
    measurement = make_measurement()

    corrected = store.correct_measurement(measurement, "published_value", "C")

    assert corrected is not measurement
    assert corrected.published_value == "C"
    assert measurement.published_value == "D"
    (entry,) = store.entries
    assert entry.target_kind == "measurement"
    assert entry.target_id == measurement.measurement_id
    assert entry.original == "D"
    assert store.original("measurement", measurement.measurement_id, "published_value") == "D"


def test_unknown_fields_are_rejected():
    store = CorrectionStore()
    with pytest.raises(ValueError):
        store.correct_compound(make_compound(), "not_a_field", 1)
    with pytest.raises(ValueError):
        store.correct_measurement(make_measurement(), "not_a_field", 1)


def test_original_returns_none_for_an_uncorrected_field():
    assert CorrectionStore().original("compound", "p:1", "smiles_final") is None


def test_to_rows_exports_original_and_corrected_side_by_side():
    """AC-7.5 — corrections export with both original and corrected values."""
    store = CorrectionStore()
    compound = make_compound()
    once = store.correct_compound(compound, "smiles_final", ETHANOL)
    store.correct_compound(once, "smiles_final", "CCC")
    store.correct_measurement(make_measurement(), "published_value", "C")

    rows = store.to_rows()
    assert len(rows) == 3
    for row in rows:
        assert {"timestamp", "target_kind", "target_id", "field", "original", "corrected"} <= set(
            row
        )
    # both rows for the same field carry the TRUE original, not the intermediate value
    assert [r["original"] for r in rows[:2]] == [VALID_SMILES, VALID_SMILES]
    assert [r["corrected"] for r in rows[:2]] == [ETHANOL, "CCC"]
    assert rows[2]["original"] == "D"


def test_correction_store_round_trips_through_a_dump():
    """PRD R13.6 — session state today, a database drop-in later."""
    store = CorrectionStore()
    once = store.correct_compound(make_compound(), "smiles_final", ETHANOL)
    store.correct_compound(once, "smiles_final", "CCC")
    store.correct_measurement(make_measurement(), "published_value", "C", note="OCR misread")

    restored = CorrectionStore.from_dump(store.model_dump())

    assert restored.entries == store.entries
    assert restored.model_dump() == store.model_dump()
    assert restored.to_rows() == store.to_rows()
    assert restored.original("compound", "WO2024097932A1:5", "smiles_final") == VALID_SMILES


def test_model_dump_is_json_serializable():
    import json

    store = CorrectionStore()
    store.correct_compound(make_compound(), "compound_number", 6)
    assert json.loads(json.dumps(store.model_dump())) == store.model_dump()


# --- provenance completeness (AC-8.1) ---------------------------------------


def test_fully_provenanced_extraction_reports_nothing():
    assert missing_provenance([make_compound()], [make_measurement()]) == []


def test_a_smiles_without_provenance_is_reported():
    """AC-8.1 — every SMILES carries page_no, bbox and a readable crop_path."""
    compound = make_compound(provenance=[])
    reported = missing_provenance([compound], [])

    assert reported
    assert any(compound.compound_id in line and "SMILES" in line for line in reported)
    assert any("compound number" in line.lower() for line in reported)


def test_an_activity_value_without_a_readable_crop_is_reported():
    measurement = make_measurement(provenance=make_provenance(crop_path=""))
    reported = missing_provenance([make_compound()], [measurement])

    assert any(measurement.measurement_id in line for line in reported)


def test_a_compound_with_no_structure_and_no_number_is_not_reported():
    """Nothing was extracted, so there is nothing that needs provenance."""
    empty = make_compound(
        compound_number=None,
        smiles_from_name=None,
        smiles_from_image=None,
        smiles_final=None,
        structure_source="none",
        inchikey_full=None,
        provenance=[],
    )
    assert missing_provenance([empty], []) == []
