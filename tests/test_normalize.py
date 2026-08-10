"""Tests for `sarmine.assay.normalize` — PRD R10.1–R10.5, R10.11–R10.13."""

from __future__ import annotations

import functools
import math
from pathlib import Path

import pytest
from lxml import html as lxml_html

from sarmine.artifacts.schema import BinDefinition, Measurement, Provenance
from sarmine.assay.legend import parse_legends
from sarmine.assay.lexicon import HeaderMatch, load_lexicon, reconstruct_header
from sarmine.assay.normalize import (
    build_measurement,
    normalize_units,
    parse_cell,
    pchembl_value,
    pdc50_value,
    to_nM,
)

FIXTURE = Path(__file__).parent / "fixtures" / "source" / "WO2024097932A1.html"

PROVENANCE = Provenance(
    page_no=186,
    bbox=(10, 20, 30, 40),
    raster_width=2550,
    raster_height=3300,
    crop_path="crops/p-186-cell.png",
    source="pdf_ocr",
    extractor="tesseract",
)


@functools.lru_cache(maxsize=1)
def reference_legends() -> dict[str, list[BinDefinition]]:
    """The legends of WO2024097932A1, decoded from its real description text."""
    sections = lxml_html.parse(str(FIXTURE)).xpath('//section[@itemprop="description"]')
    legends, _ = parse_legends(str(sections[0].text_content()))
    return legends


def header_for(raw: str) -> HeaderMatch:
    match = load_lexicon().match(raw)
    assert match is not None, raw
    return match


def _all_reference_cells() -> list[tuple[str, str]]:
    """Every (column, letter) pair of WO2024097932A1's Table 2 (PRD §3.4)."""
    return [
        (header, label)
        for header, labels in (
            ("HbF Induction (%)", "ABC"),
            ("WIZ EC50 (uM)", "DEF"),
            ("ZBTB7A EC50 (uM)", "GHI"),
        )
        for label in labels
    ]


def measure(
    raw_cell: str,
    header: str,
    *,
    legends: dict[str, list[BinDefinition]] | None = None,
    assay_group_key: str = "WO2024097932A1::col",
    is_off_target: bool = False,
    cell_line: str | None = None,
    timepoint_h: float | None = None,
) -> Measurement | None:
    return build_measurement(
        compound_id="WO2024097932A1-c5",
        header=header_for(header),
        raw_cell=raw_cell,
        provenance=PROVENANCE,
        assay_group_key=assay_group_key,
        legends=legends,
        is_off_target=is_off_target,
        cell_line=cell_line,
        timepoint_h=timepoint_h,
    )


class TestNormalizeUnits:
    """PRD R10.12 / EC-15 — homoglyph repair before unit parsing."""

    def test_canonical_concentration_units_pass_through(self) -> None:
        assert normalize_units("nM") == "nM"
        assert normalize_units("uM") == "uM"
        assert normalize_units("mM") == "mM"
        assert normalize_units("M") == "M"

    def test_micro_sign_and_greek_mu_both_normalize_to_u(self) -> None:
        assert normalize_units("\u00b5M") == "uM"  # MICRO SIGN
        assert normalize_units("\u03bcM") == "uM"  # GREEK SMALL LETTER MU

    def test_p_adjacent_to_M_is_the_ocr_corruption_of_micro(self) -> None:
        # PRD R10.12 — this source renders every `µM` as `pM`.
        assert normalize_units("pM") == "uM"
        assert normalize_units("pM.") == "uM"

    def test_picomolar_is_not_representable_in_v1(self) -> None:
        # The `pM` symbol is reserved for the corruption above, so a spelled-out
        # picomolar must be refused rather than silently mapped (PRD R10.11).
        assert normalize_units("picomolar") is None

    def test_nanometre_is_a_different_quantity_and_is_not_promoted_to_nanomolar(self) -> None:
        # PRD R10.12 — `nM` vs `nm`.
        assert normalize_units("nm") == "nm"
        assert normalize_units("nM") == "nM"

    def test_spelled_out_molarities(self) -> None:
        assert normalize_units("nanomolar") == "nM"
        assert normalize_units("micromolar") == "uM"
        assert normalize_units("millimolar") == "mM"

    def test_percent_forms(self) -> None:
        assert normalize_units("%") == "%"
        assert normalize_units("percent") == "%"

    def test_surrounding_parentheses_and_padding_are_stripped(self) -> None:
        assert normalize_units("(uM)") == "uM"
        assert normalize_units("  [ nM ] ") == "nM"

    def test_absent_or_unrecognized_units_return_none(self) -> None:
        # PRD R10.11 / EC-14 — refuse; never infer.
        assert normalize_units(None) is None
        assert normalize_units("") is None
        assert normalize_units("   ") is None
        assert normalize_units("fortnights") is None
        assert normalize_units("Nm") is None


class TestToNanomolar:
    """PRD R10.2 / R10.13 — standardize concentrations to nM using `pint`."""

    def test_micromolar_to_nanomolar(self) -> None:
        assert to_nM(1.0, "uM") == pytest.approx(1000.0)
        assert to_nM(0.01, "uM") == pytest.approx(10.0)
        assert to_nM(0.03, "uM") == pytest.approx(30.0)
        assert to_nM(0.1, "uM") == pytest.approx(100.0)

    def test_nanomolar_is_the_identity(self) -> None:
        assert to_nM(7.6, "nM") == pytest.approx(7.6)

    def test_millimolar_and_molar(self) -> None:
        assert to_nM(1.0, "mM") == pytest.approx(1_000_000.0)
        assert to_nM(0.01, "M") == pytest.approx(1e7)

    def test_corrupted_micromolar_symbol_is_normalized_before_conversion(self) -> None:
        # PRD R10.12 / AC-4.4 — `pM` in this source means µM.
        assert to_nM(0.01, "pM") == pytest.approx(10.0)

    def test_dimensionless_and_length_units_are_refused(self) -> None:
        # PRD R10.11 — a wrong unit must fail loudly, never be coerced.
        with pytest.raises(ValueError):
            to_nM(50.0, "%")
        with pytest.raises(ValueError):
            to_nM(50.0, "nm")
        with pytest.raises(ValueError):
            to_nM(50.0, "fortnights")

    def test_conversion_is_exact_enough_for_log_math(self) -> None:
        assert math.log10(to_nM(0.001, "uM")) == pytest.approx(0.0)


class TestParseCell:
    """PRD R10.1 — real patent cell contents, kept verbatim and also parsed."""

    def test_a_plain_number(self) -> None:
        cell = parse_cell("7.6")
        assert (cell.raw, cell.relation, cell.value) == ("7.6", "=", 7.6)
        assert cell.text is None
        assert cell.is_blank is False

    def test_censored_greater_than_with_a_thousands_separator(self) -> None:
        cell = parse_cell(">10,000")
        assert cell.relation == ">"
        assert cell.value == pytest.approx(10_000.0)
        assert cell.raw == ">10,000"

    def test_censored_less_than(self) -> None:
        cell = parse_cell("< 0.01")
        assert (cell.relation, cell.value) == ("<", pytest.approx(0.01))

    def test_inclusive_relations(self) -> None:
        assert parse_cell(">= 5").relation == ">="
        assert parse_cell("\u2265 5").relation == ">="
        assert parse_cell("<= 5").relation == "<="
        assert parse_cell("\u2264 5").relation == "<="

    def test_a_value_with_a_standard_deviation_keeps_only_the_central_value(self) -> None:
        cell = parse_cell("5.6 \u00b1 0.3")
        assert cell.value == pytest.approx(5.6)
        assert cell.relation == "="
        assert cell.raw == "5.6 \u00b1 0.3"  # the spread survives verbatim (R10.1)

    def test_a_letter_bin_is_text_not_a_number(self) -> None:
        cell = parse_cell("A")
        assert cell.value is None
        assert cell.text == "A"
        assert cell.is_blank is False

    def test_not_determined_is_text_not_blank(self) -> None:
        cell = parse_cell("n.d.")
        assert cell.value is None
        assert cell.text == "n.d."
        assert cell.is_blank is False

    @pytest.mark.parametrize("raw", ["", "   ", "-", "\u2013", "\u2014"])
    def test_a_blank_cell_never_becomes_a_number(self, raw: str) -> None:
        # PRD EC-7 — compounds 33-38, 49-51, 53, 54 have no HbF value at all.
        cell = parse_cell(raw)
        assert cell.is_blank is True
        assert cell.value is None
        assert cell.value != 0.0


class TestPchemblGate:
    """PRD R10.3 — five conditions, all of which must hold."""

    def test_the_canonical_computation(self) -> None:
        assert pchembl_value("IC50", "=", 10.0, "nM") == pytest.approx(8.0)
        assert pchembl_value("EC50", "=", 1.0, "nM") == pytest.approx(9.0)
        assert pchembl_value("Ki", "=", 100.0, "nM") == pytest.approx(7.0)

    @pytest.mark.parametrize(
        "standard_type", ["IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50"]
    )
    def test_every_permitted_type(self, standard_type: str) -> None:
        assert pchembl_value(standard_type, "=", 10.0, "nM") == pytest.approx(8.0)

    @pytest.mark.parametrize("standard_type", ["DC50", "Dmax", "Inhibition", "Induction", "GI50"])
    def test_types_outside_the_permitted_set_get_none(self, standard_type: str) -> None:
        # PRD R10.4 — ChEMBL computes no pChEMBL for DC50, and neither do we.
        assert pchembl_value(standard_type, "=", 10.0, "nM") is None

    @pytest.mark.parametrize("relation", [">", "<", ">=", "<="])
    def test_censored_values_get_no_pchembl(self, relation: str) -> None:
        # PRD EC-17 — censoring is a third of the corpus, not an edge case.
        assert pchembl_value("IC50", relation, 10.0, "nM") is None

    def test_units_must_already_be_nanomolar(self) -> None:
        assert pchembl_value("IC50", "=", 10.0, "uM") is None
        assert pchembl_value("IC50", "=", 10.0, "%") is None
        assert pchembl_value("IC50", "=", 10.0, None) is None

    def test_value_must_be_present_and_positive(self) -> None:
        assert pchembl_value("IC50", "=", None, "nM") is None
        assert pchembl_value("IC50", "=", 0.0, "nM") is None
        assert pchembl_value("IC50", "=", -1.0, "nM") is None

    def test_a_blocking_validity_comment_suppresses_it(self) -> None:
        assert pchembl_value("IC50", "=", 10.0, "nM", validity_comment="Outside typical range") is None

    def test_manual_validation_is_not_a_blocking_comment(self) -> None:
        assert pchembl_value(
            "IC50", "=", 10.0, "nM", validity_comment="Manually validated"
        ) == pytest.approx(8.0)


class TestPdc50:
    """PRD R10.4 — degrader potency lives in its own column."""

    def test_the_canonical_computation(self) -> None:
        assert pdc50_value(7.6, "=") == pytest.approx(9.0 - math.log10(7.6))
        assert pdc50_value(1.0, "=") == pytest.approx(9.0)

    @pytest.mark.parametrize("relation", [">", "<", ">=", "<="])
    def test_censored_dc50_gets_no_pdc50(self, relation: str) -> None:
        assert pdc50_value(10.0, relation) is None

    def test_missing_or_non_positive_values(self) -> None:
        assert pdc50_value(None, "=") is None
        assert pdc50_value(0.0, "=") is None
        assert pdc50_value(-5.0, "=") is None


class TestBuildMeasurementLetterBins:
    """PRD R10.5 / AC-4.3 / EC-17 — a letter bin is interval-censored data."""

    def test_a_bin_never_gets_an_imputed_midpoint(self) -> None:
        measurement = measure("D", "WIZ EC50 (uM)", legends=reference_legends())
        assert measurement is not None
        # PRD R10.5 — a midpoint would fabricate precision the document lacks.
        assert measurement.standard_value is None

    def test_the_verbatim_letter_survives_in_both_published_columns(self) -> None:
        measurement = measure("D", "WIZ EC50 (uM)", legends=reference_legends())
        assert measurement is not None
        assert measurement.published_value == "D"
        assert isinstance(measurement.published_value, str)  # PRD R10.1
        assert measurement.published_text_value == "D"
        assert measurement.bin_label_raw == "D"

    def test_the_decoded_interval_is_stored_in_nanomolar(self) -> None:
        measurement = measure("D", "WIZ EC50 (uM)", legends=reference_legends())
        assert measurement is not None
        assert measurement.bin_lower_nM is None
        assert measurement.bin_upper_nM == pytest.approx(10.0)
        assert measurement.standard_units == "nM"
        assert measurement.is_censored is True
        assert measurement.censor_direction == "upper_bound"
        assert measurement.standard_relation == "<"

    def test_a_lower_bounded_bin(self) -> None:
        measurement = measure("F", "WIZ EC50 (uM)", legends=reference_legends())
        assert measurement is not None
        assert measurement.bin_lower_nM == pytest.approx(100.0)
        assert measurement.bin_upper_nM is None
        assert measurement.censor_direction == "lower_bound"
        assert measurement.standard_relation == ">"

    def test_a_two_sided_bin(self) -> None:
        measurement = measure("E", "WIZ EC50 (uM)", legends=reference_legends())
        assert measurement is not None
        assert measurement.bin_lower_nM == pytest.approx(10.0)
        assert measurement.bin_upper_nM == pytest.approx(100.0)
        assert measurement.is_censored is True

    def test_a_bin_gets_no_pchembl_and_no_pdc50(self) -> None:
        # PRD EC-17 — censored data is ranked in its own bucket, not scored.
        for label in ("D", "E", "F"):
            measurement = measure(label, "WIZ EC50 (uM)", legends=reference_legends())
            assert measurement is not None
            assert measurement.pchembl_value is None
            assert measurement.pdc50_value is None

    def test_a_contradicted_bin_is_marked_reduced_confidence(self) -> None:
        # PRD R10.6 / EC-5 — level F's summary restatement contradicts.
        contradicted = measure("F", "WIZ EC50 (uM)", legends=reference_legends())
        agreeing = measure("D", "WIZ EC50 (uM)", legends=reference_legends())
        assert contradicted is not None and agreeing is not None
        assert contradicted.reduced_confidence is True
        assert agreeing.reduced_confidence is False

    def test_the_bin_definition_and_score_travel_with_the_measurement(self) -> None:
        measurement = measure("D", "WIZ EC50 (uM)", legends=reference_legends())
        assert measurement is not None
        assert "level D" in (measurement.bin_definition or "")
        assert measurement.bin_score == 3

    def test_a_percentage_bin_keeps_its_own_units(self) -> None:
        measurement = measure("A", "HbF Induction (%)", legends=reference_legends())
        assert measurement is not None
        assert measurement.standard_value is None
        assert measurement.standard_units == "%"
        assert measurement.bin_lower_nM == pytest.approx(66.0)
        assert measurement.bin_upper_nM == pytest.approx(100.0)
        assert measurement.reduced_confidence is True  # level A is contradicted

    def test_no_measurement_ever_carries_the_corrupted_pM_unit(self) -> None:
        # PRD AC-4.4 across every activity column of the reference patent.
        for header, label in _all_reference_cells():
            measurement = measure(label, header, legends=reference_legends())
            assert measurement is not None, (header, label)
            assert measurement.standard_units != "pM"
            assert measurement.published_units != "pM"

    def test_every_reference_bin_has_an_interval_and_no_standard_value(self) -> None:
        # PRD AC-4.3 over all nine bins of the reference patent. One-sided bins
        # (C, D, F, G, I) legitimately carry a single bound.
        for header, label in _all_reference_cells():
            measurement = measure(label, header, legends=reference_legends())
            assert measurement is not None, (header, label)
            assert measurement.standard_value is None, (header, label)
            bounds = (measurement.bin_lower_nM, measurement.bin_upper_nM)
            assert any(bound is not None for bound in bounds), (header, label)
            assert measurement.bin_label_raw == label
            assert measurement.bin_score in {1, 2, 3}

    def test_off_target_and_assay_context_are_carried_through(self) -> None:
        measurement = measure(
            "I",
            "ZBTB7A EC50 (uM)",
            legends=reference_legends(),
            is_off_target=True,
            cell_line="HUDEP-2",
            timepoint_h=24.0,
        )
        assert measurement is not None
        assert measurement.is_off_target is True
        assert measurement.cell_line == "HUDEP-2"
        assert measurement.timepoint_h == 24.0
        assert measurement.target_raw == "ZBTB7A"

    def test_an_undecodable_letter_is_refused(self) -> None:
        assert measure("Z", "WIZ EC50 (uM)", legends=reference_legends()) is None

    def test_a_bin_needs_no_units_in_the_header_because_the_legend_supplies_them(self) -> None:
        measurement = measure("D", "WIZ EC50", legends=reference_legends())
        assert measurement is not None
        assert measurement.published_units is None
        assert measurement.bin_upper_nM == pytest.approx(10.0)


class TestBuildMeasurementNumeric:
    """PRD R10.1–R10.4 — the standardized columns sit beside the published ones."""

    def test_a_micromolar_value_standardizes_to_nanomolar(self) -> None:
        measurement = measure("0.05", "WIZ EC50 (uM)")
        assert measurement is not None
        assert measurement.published_value == "0.05"
        assert measurement.published_units == "uM"
        assert measurement.standard_value == pytest.approx(50.0)
        assert measurement.standard_units == "nM"
        assert measurement.pchembl_value == pytest.approx(9.0 - math.log10(50.0))

    def test_the_second_reference_patents_hibit_dc50(self) -> None:
        # PRD AC-4.5 — header split across four rows, numeric nM value.
        header = header_for(reconstruct_header(["HiBiT", "DC50", "(nM)"]))
        measurement = build_measurement(
            compound_id="US20250368620A1-c1",
            header=header,
            raw_cell="7.6",
            provenance=PROVENANCE,
            assay_group_key="US20250368620A1::hibit-dc50-nm",
        )
        assert measurement is not None
        assert measurement.standard_type == "DC50"
        assert measurement.standard_value == pytest.approx(7.6)
        assert measurement.standard_units == "nM"
        assert measurement.pdc50_value == pytest.approx(9.0 - math.log10(7.6))
        assert measurement.pchembl_value is None  # PRD R10.4

    def test_a_censored_numeric_value(self) -> None:
        measurement = measure(">10,000", "IC50 (nM)")
        assert measurement is not None
        assert measurement.published_value == ">10,000"
        assert measurement.standard_relation == ">"
        assert measurement.standard_value == pytest.approx(10_000.0)
        assert measurement.is_censored is True
        assert measurement.censor_direction == "lower_bound"
        assert measurement.pchembl_value is None  # PRD EC-17

    def test_a_log_form_is_unwound_to_linear_nanomolar(self) -> None:
        measurement = measure("8.0", "pIC50")
        assert measurement is not None
        assert measurement.standard_type == "IC50"
        assert measurement.published_value == "8.0"
        assert measurement.published_units is None
        assert measurement.standard_value == pytest.approx(10.0)
        assert measurement.standard_units == "nM"
        assert measurement.pchembl_value == pytest.approx(8.0)

    def test_unwinding_a_log_form_inverts_the_relation(self) -> None:
        # A pIC50 > 8 is an IC50 < 10 nM; keeping `>` would invert the potency.
        measurement = measure("> 8.0", "pIC50")
        assert measurement is not None
        assert measurement.standard_relation == "<"
        assert measurement.censor_direction == "upper_bound"

    def test_a_percent_endpoint_stays_in_percent(self) -> None:
        measurement = measure("45", "% Inhibition @ 10 uM")
        assert measurement is not None
        assert measurement.standard_type == "Inhibition"
        assert measurement.standard_value == pytest.approx(45.0)
        assert measurement.standard_units == "%"
        assert measurement.pchembl_value is None

    def test_a_dmax_column_does_not_fabricate_a_pairing(self) -> None:
        # PRD R10.4 — Dmax is a paired attribute of a DC50 on the same row, so a
        # lone Dmax cell must not populate `dmax_pct`.
        measurement = measure("95", "Dmax (%)")
        assert measurement is not None
        assert measurement.standard_type == "Dmax"
        assert measurement.standard_value == pytest.approx(95.0)
        assert measurement.dmax_pct is None

    def test_ontology_identifiers_are_carried_for_provenance(self) -> None:
        # PRD R10.13 — identifiers only; the arithmetic is pint's job.
        measurement = measure("0.05", "WIZ EC50 (uM)")
        assert measurement is not None
        assert measurement.bao_endpoint == "BAO_0000188"
        assert measurement.uo_units == "UO_0000065"
        percent = measure("45", "% Inhibition @ 10 uM")
        assert percent is not None and percent.uo_units == "UO_0000187"

    def test_measurement_ids_are_deterministic_and_distinct_per_column(self) -> None:
        first = measure("0.05", "WIZ EC50 (uM)", assay_group_key="P::wiz")
        again = measure("0.05", "WIZ EC50 (uM)", assay_group_key="P::wiz")
        other = measure("0.05", "ZBTB7A EC50 (uM)", assay_group_key="P::zbtb7a")
        assert first is not None and again is not None and other is not None
        assert first.measurement_id == again.measurement_id
        assert first.measurement_id != other.measurement_id


class TestBuildMeasurementRefusals:
    """PRD R10.11 / EC-14 / EC-7 — refusing beats guessing."""

    def test_a_numeric_value_with_no_units_in_the_header_is_refused(self) -> None:
        # A silent nM/uM confusion is a 1000x error in a potency ranking.
        assert header_for("WIZ EC50").units is None
        assert measure("0.05", "WIZ EC50") is None

    def test_units_are_never_inferred_from_magnitude(self) -> None:
        # 0.05 "looks like" µM and 50 "looks like" nM. Both must still be refused.
        for raw in ("0.05", "50", "10000"):
            assert measure(raw, "WIZ EC50") is None, raw

    def test_a_nanometre_header_is_refused_rather_than_read_as_nanomolar(self) -> None:
        # PRD R10.12 — `nm` is a length; it is not a sloppy `nM`.
        header = header_for("IC50 (nm)")
        assert header.units == "nm"
        assert (
            build_measurement(
                compound_id="c1",
                header=header,
                raw_cell="10",
                provenance=PROVENANCE,
                assay_group_key="P::ic50-nm",
            )
            is None
        )

    def test_a_blank_cell_produces_no_measurement_at_all(self) -> None:
        # PRD EC-7 — a blank must never surface as a zero or a low value.
        for raw in ("", "   ", "-"):
            assert measure(raw, "HbF Induction (%)", legends=reference_legends()) is None, raw

    def test_unparseable_text_is_refused(self) -> None:
        assert measure("n.d.", "WIZ EC50 (uM)", legends=reference_legends()) is None

    def test_a_letter_cell_with_no_legends_is_refused(self) -> None:
        assert measure("D", "WIZ EC50 (uM)") is None
        assert measure("D", "WIZ EC50 (uM)", legends={}) is None
