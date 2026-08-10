"""Tests for `sarmine.assay.qc` — PRD R10.14, R10.15, R10.16, R10.17, EC-16."""

from __future__ import annotations

import pytest

from sarmine.artifacts.schema import Measurement, Provenance
from sarmine.assay.qc import (
    assay_group_key,
    detect_transcription_errors,
    is_meaningful_difference,
    outside_typical_range,
)
from sarmine.config import get_config

PROVENANCE = Provenance(
    page_no=186,
    bbox=(0, 0, 1, 1),
    raster_width=100,
    raster_height=100,
    crop_path="crops/x.png",
    source="structured",
    extractor="test",
)


def measurement(
    value: float | None,
    *,
    compound_id: str = "c1",
    group: str = "WO2024097932A1::wiz-ec50-um",
    standard_type: str = "EC50",
    units: str | None = "nM",
    relation: str = "=",
    measurement_id: str = "m",
) -> Measurement:
    return Measurement(
        measurement_id=measurement_id,
        compound_id=compound_id,
        assay_group_key=group,
        assay_name_raw="WIZ EC50 (uM)",
        published_type="WIZ EC50 (uM)",
        published_value="" if value is None else str(value),
        standard_type=standard_type,
        standard_relation=relation,  # type: ignore[arg-type]
        standard_value=value,
        standard_units=units,
        provenance=PROVENANCE,
    )


class TestAssayGroupKey:
    """PRD R10.17 — ChEMBL's TOID analogue; the publication number is part of it."""

    def test_the_key_is_deterministic(self) -> None:
        assert assay_group_key("WO2024097932A1", "WIZ EC50 (uM)") == assay_group_key(
            "WO2024097932A1", "WIZ EC50 (uM)"
        )

    def test_the_publication_number_is_part_of_the_key(self) -> None:
        assert "WO2024097932A1" in assay_group_key("WO2024097932A1", "WIZ EC50 (uM)")

    def test_the_same_target_in_a_different_patent_is_a_different_assay(self) -> None:
        # Cross-patent potency comparison without this is how patent SAR mining
        # produces garbage.
        assert assay_group_key("WO2024097932A1", "WIZ EC50 (uM)") != assay_group_key(
            "US20250368620A1", "WIZ EC50 (uM)"
        )

    def test_two_columns_of_the_same_table_are_different_assays(self) -> None:
        assert assay_group_key("WO2024097932A1", "WIZ EC50 (uM)") != assay_group_key(
            "WO2024097932A1", "ZBTB7A EC50 (uM)"
        )

    def test_cosmetic_header_differences_do_not_split_a_column(self) -> None:
        # Page 187 of the reference patent carries no header, so a re-OCR of the
        # same column must land in the same group.
        assert assay_group_key("WO2024097932A1", "WIZ EC50 (uM)") == assay_group_key(
            "WO2024097932A1", "  wiz   EC50 (uM) "
        )

    def test_the_key_is_readable(self) -> None:
        key = assay_group_key("WO2024097932A1", "HbF Induction (%)")
        assert key.startswith("WO2024097932A1")
        assert " " not in key


class TestTranscriptionErrors:
    """PRD R10.14 / EC-16 — exactly 3 or 6 orders of magnitude is a unit mixup."""

    def test_three_orders_of_magnitude_is_flagged(self) -> None:
        anomalies = detect_transcription_errors(
            [measurement(5.0, measurement_id="a"), measurement(5000.0, measurement_id="b")]
        )
        assert len(anomalies) == 1
        assert anomalies[0].kind == "transcription_error"
        assert anomalies[0].severity in {"info", "warning", "error"}
        assert "c1" in anomalies[0].message

    def test_six_orders_of_magnitude_is_flagged(self) -> None:
        anomalies = detect_transcription_errors(
            [measurement(5.0, measurement_id="a"), measurement(5_000_000.0, measurement_id="b")]
        )
        assert len(anomalies) == 1
        assert "6" in anomalies[0].message

    def test_two_orders_of_magnitude_is_not_flagged(self) -> None:
        assert (
            detect_transcription_errors(
                [measurement(5.0, measurement_id="a"), measurement(500.0, measurement_id="b")]
            )
            == []
        )

    @pytest.mark.parametrize("other", [5.0, 50.0, 500.0, 15_000.0, 50_000.0])
    def test_only_exact_powers_of_a_thousand_are_flagged(self, other: float) -> None:
        assert (
            detect_transcription_errors(
                [measurement(5.0, measurement_id="a"), measurement(other, measurement_id="b")]
            )
            == []
        )

    def test_measurements_of_different_compounds_are_not_compared(self) -> None:
        assert (
            detect_transcription_errors(
                [
                    measurement(5.0, compound_id="c1", measurement_id="a"),
                    measurement(5000.0, compound_id="c2", measurement_id="b"),
                ]
            )
            == []
        )

    def test_measurements_of_different_assays_are_not_compared(self) -> None:
        # PRD R10.17 — different assay groups are not "otherwise identical".
        assert (
            detect_transcription_errors(
                [
                    measurement(5.0, group="P::wiz", measurement_id="a"),
                    measurement(5000.0, group="P::zbtb7a", measurement_id="b"),
                ]
            )
            == []
        )

    def test_bins_and_missing_values_are_skipped(self) -> None:
        assert (
            detect_transcription_errors(
                [measurement(None, measurement_id="a"), measurement(None, measurement_id="b")]
            )
            == []
        )

    def test_every_offending_pair_is_reported(self) -> None:
        anomalies = detect_transcription_errors(
            [
                measurement(5.0, measurement_id="a"),
                measurement(5_000.0, measurement_id="b"),
                measurement(5_000_000.0, measurement_id="c"),
            ]
        )
        assert len(anomalies) == 3

    def test_an_empty_input_is_safe(self) -> None:
        assert detect_transcription_errors([]) == []


class TestOutsideTypicalRange:
    """PRD R10.15 — roughly 0.01 nM to 100 uM for IC50/Ki/EC50."""

    @pytest.mark.parametrize("standard_type", ["IC50", "Ki", "EC50"])
    def test_values_far_above_the_range(self, standard_type: str) -> None:
        assert outside_typical_range(measurement(500_000.0, standard_type=standard_type)) is True

    @pytest.mark.parametrize("standard_type", ["IC50", "Ki", "EC50"])
    def test_values_far_below_the_range(self, standard_type: str) -> None:
        assert outside_typical_range(measurement(0.001, standard_type=standard_type)) is True

    @pytest.mark.parametrize("value", [0.01, 1.0, 50.0, 100_000.0])
    def test_values_inside_the_range_including_the_boundaries(self, value: float) -> None:
        assert outside_typical_range(measurement(value)) is False

    def test_endpoints_outside_the_rules_scope_are_not_judged(self) -> None:
        assert outside_typical_range(measurement(1e9, standard_type="DC50")) is False
        assert outside_typical_range(measurement(1e9, standard_type="Inhibition")) is False

    def test_a_bin_has_no_value_to_judge(self) -> None:
        assert outside_typical_range(measurement(None)) is False

    def test_a_non_nanomolar_value_is_not_judged(self) -> None:
        assert outside_typical_range(measurement(500_000.0, units="%")) is False
        assert outside_typical_range(measurement(500_000.0, units=None)) is False


class TestNoiseFloor:
    """PRD R10.16 — 0.3 log units is the experimental noise floor."""

    def test_the_default_floor_comes_from_config(self) -> None:
        assert get_config().noise_floor_log_units == 0.3

    def test_a_full_log_unit_is_meaningful(self) -> None:
        assert is_meaningful_difference(10.0, 100.0) is True

    def test_a_difference_below_the_floor_is_not_meaningful(self) -> None:
        assert is_meaningful_difference(10.0, 15.0) is False
        assert is_meaningful_difference(10.0, 19.9) is False

    def test_a_difference_just_above_the_floor_is_meaningful(self) -> None:
        assert is_meaningful_difference(10.0, 20.1) is True

    def test_the_comparison_is_symmetric(self) -> None:
        assert is_meaningful_difference(100.0, 10.0) is True
        assert is_meaningful_difference(15.0, 10.0) is False

    def test_identical_values_are_never_meaningful(self) -> None:
        assert is_meaningful_difference(42.0, 42.0) is False

    def test_the_floor_is_configurable(self) -> None:
        assert is_meaningful_difference(10.0, 100.0, floor_log_units=1.5) is False
        assert is_meaningful_difference(10.0, 15.0, floor_log_units=0.1) is True

    def test_non_positive_values_cannot_be_compared_on_a_log_scale(self) -> None:
        assert is_meaningful_difference(0.0, 100.0) is False
        assert is_meaningful_difference(-1.0, 100.0) is False
