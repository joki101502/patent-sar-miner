"""Tests for `sarmine.assay.legend` — PRD R10.6, R10.7, EC-5, EC-6, AC-4.1, AC-4.2.

Every legend assertion runs against the REAL cached Google Patents description
text for WO2024097932A1, not a paraphrase: the contradictions and the `µ`->`p`
OCR corruption only exist in the genuine prose.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest
from lxml import html as lxml_html

from sarmine.artifacts.schema import BinDefinition
from sarmine.assay.legend import decode_bin, detect_cross_reference_error, parse_legends
from sarmine.assay.normalize import to_nM

FIXTURE = Path(__file__).parent / "fixtures" / "source" / "WO2024097932A1.html"

HBF = "HbF induction"
WIZ = "WIZ EC50"
ZBTB7A = "ZBTB7A EC50"


@functools.lru_cache(maxsize=1)
def description_text() -> str:
    """The verbatim description text, read with lxml so this test owns no source deps."""
    sections = lxml_html.parse(str(FIXTURE)).xpath('//section[@itemprop="description"]')
    assert len(sections) == 1
    return str(sections[0].text_content())


@functools.lru_cache(maxsize=1)
def real_legends() -> dict[str, list[BinDefinition]]:
    legends, _ = parse_legends(description_text())
    return legends


@functools.lru_cache(maxsize=1)
def real_anomalies() -> tuple:
    _, anomalies = parse_legends(description_text())
    return tuple(anomalies)


def bin_for(assay: str, label: str) -> BinDefinition:
    matches = [b for b in real_legends()[assay] if b.label == label]
    assert len(matches) == 1, f"{assay}/{label}: {matches}"
    return matches[0]


class TestFixture:
    def test_the_cached_page_contains_the_legend_paragraphs(self) -> None:
        text = description_text()
        assert len(text) > 240_000  # PRD AC-1.2
        for para in ("[00508]", "[00511]", "[00512]", "[00513]"):
            assert para in text
        assert "pM" in text  # PRD R10.12 — the corruption is present in the source


class TestLegendRecovery:
    """PRD AC-4.1 — all three legends recovered with the R10.7 intervals."""

    def test_exactly_the_three_reference_patent_assays_are_found(self) -> None:
        assert set(real_legends()) == {HBF, WIZ, ZBTB7A}

    def test_each_assay_has_exactly_its_three_bins(self) -> None:
        assert [b.label for b in real_legends()[HBF]] == ["A", "B", "C"]
        assert [b.label for b in real_legends()[WIZ]] == ["D", "E", "F"]
        assert [b.label for b in real_legends()[ZBTB7A]] == ["G", "H", "I"]

    @pytest.mark.parametrize(
        ("label", "lower", "upper", "lower_inclusive", "upper_inclusive"),
        [
            ("A", 66.0, 100.0, True, True),
            ("B", 33.0, 66.0, True, True),
            ("C", None, 33.0, True, False),
        ],
    )
    def test_hbf_percent_bins(
        self,
        label: str,
        lower: float | None,
        upper: float | None,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> None:
        binned = bin_for(HBF, label)
        assert binned.units == "%"
        assert binned.lower == lower
        assert binned.upper == upper
        assert binned.lower_inclusive is lower_inclusive
        assert binned.upper_inclusive is upper_inclusive

    @pytest.mark.parametrize(
        ("assay", "label", "lower", "upper", "lower_inclusive"),
        [
            (WIZ, "D", None, 0.01, True),
            (WIZ, "E", 0.01, 0.1, False),
            (WIZ, "F", 0.1, None, False),
            (ZBTB7A, "G", None, 0.03, True),
            (ZBTB7A, "H", 0.03, 0.1, False),
            (ZBTB7A, "I", 0.1, None, False),
        ],
    )
    def test_ec50_micromolar_bins(
        self,
        assay: str,
        label: str,
        lower: float | None,
        upper: float | None,
        lower_inclusive: bool,
    ) -> None:
        binned = bin_for(assay, label)
        assert binned.assay == assay
        assert binned.units == "uM"
        if lower is None:
            assert binned.lower is None
        else:
            assert binned.lower == pytest.approx(lower)
        if upper is None:
            assert binned.upper is None
        else:
            assert binned.upper == pytest.approx(upper)
        assert binned.lower_inclusive is lower_inclusive

    @pytest.mark.parametrize(
        ("assay", "label", "lower_nM", "upper_nM"),
        [
            (WIZ, "D", None, 10.0),
            (WIZ, "E", 10.0, 100.0),
            (WIZ, "F", 100.0, None),
            (ZBTB7A, "G", None, 30.0),
            (ZBTB7A, "H", 30.0, 100.0),
            (ZBTB7A, "I", 100.0, None),
        ],
    )
    def test_the_nanomolar_equivalents_in_the_r10_7_table(
        self, assay: str, label: str, lower_nM: float | None, upper_nM: float | None
    ) -> None:
        binned = bin_for(assay, label)
        if lower_nM is None:
            assert binned.lower is None
        else:
            assert to_nM(binned.lower, binned.units) == pytest.approx(lower_nM)
        if upper_nM is None:
            assert binned.upper is None
        else:
            assert to_nM(binned.upper, binned.units) == pytest.approx(upper_nM)

    def test_no_bin_carries_the_corrupted_pM_unit(self) -> None:
        # PRD AC-4.4 / EC-15 — `pM` in this source means µM.
        for bins in real_legends().values():
            for binned in bins:
                assert binned.units != "pM"

    def test_every_bin_keeps_its_definitional_sentence(self) -> None:
        for bins in real_legends().values():
            for binned in bins:
                assert "level" in binned.definition_text.lower()

    def test_bin_scores_order_bins_by_desirability(self) -> None:
        # Supports PRD AC-6.3: bin(WIZ D) - bin(ZBTB7A I) == +2.
        assert [b.score for b in real_legends()[HBF]] == [3, 2, 1]
        assert [b.score for b in real_legends()[WIZ]] == [3, 2, 1]
        assert [b.score for b in real_legends()[ZBTB7A]] == [3, 2, 1]
        assert bin_for(WIZ, "D").score - bin_for(ZBTB7A, "I").score == 2


class TestContradictions:
    """PRD R10.6 / EC-5 / AC-4.2 — the definitional sentence wins, loudly."""

    def test_all_anomalies_are_legend_contradictions(self) -> None:
        assert real_anomalies()
        for anomaly in real_anomalies():
            assert anomaly.kind == "legend_contradiction"
            assert anomaly.severity in {"info", "warning", "error"}

    def test_the_contradicted_bins_are_exactly_A_F_G_H_I(self) -> None:
        labels = sorted(_labels_in(anomaly.message) for anomaly in real_anomalies())
        assert labels == ["A", "F", "G", "H", "I"]

    def test_all_three_legends_report_a_contradiction(self) -> None:
        messages = " | ".join(a.message for a in real_anomalies())
        for assay in ("HbF", "WIZ", "ZBTB7A"):
            assert assay in messages

    def test_messages_state_both_the_definitional_and_the_summary_reading(self) -> None:
        for anomaly in real_anomalies():
            assert "definitional" in anomaly.message.lower()
            assert "summary" in anomaly.message.lower()

    def test_hbf_level_A_keeps_the_definitional_66_not_the_summary_67(self) -> None:
        assert bin_for(HBF, "A").lower == 66.0

    def test_wiz_level_F_keeps_the_definitional_lower_bound_not_the_summary_upper(self) -> None:
        # The summary restates F as `< .01 µM`, identical to level D.
        wiz_f = bin_for(WIZ, "F")
        assert wiz_f.lower == pytest.approx(0.1)
        assert wiz_f.upper is None
        assert bin_for(WIZ, "D").upper == pytest.approx(0.01)

    def test_zbtb7a_keeps_the_definitional_thresholds_and_units(self) -> None:
        assert bin_for(ZBTB7A, "G").upper == pytest.approx(0.03)
        assert bin_for(ZBTB7A, "G").units == "uM"  # summary corrupts this to bare `M`
        assert bin_for(ZBTB7A, "H").lower == pytest.approx(0.03)
        assert bin_for(ZBTB7A, "I").lower == pytest.approx(0.1)

    def test_agreeing_bins_produce_no_anomaly(self) -> None:
        contradicted = {_labels_in(a.message) for a in real_anomalies()}
        assert contradicted.isdisjoint({"B", "C", "D", "E"})

    def test_a_percentage_lower_bound_of_zero_is_not_a_contradiction(self) -> None:
        # The summary states C as `0-33%` where the definition says `less than 33%`;
        # on a non-negative quantity those are the same statement.
        assert bin_for(HBF, "C").lower is None


class TestDecodeBin:
    def test_decodes_a_letter_from_any_assay(self) -> None:
        legends = real_legends()
        assert decode_bin("A", legends) is bin_for(HBF, "A")
        assert decode_bin("I", legends) is bin_for(ZBTB7A, "I")

    def test_tolerates_padding_and_case(self) -> None:
        decoded = decode_bin(" d ", real_legends())
        assert decoded is not None and decoded.label == "D"

    def test_unknown_label_returns_none(self) -> None:
        assert decode_bin("Z", real_legends()) is None
        assert decode_bin("", real_legends()) is None

    def test_ambiguous_label_is_refused_rather_than_guessed(self) -> None:
        legends = {
            "Assay One": [BinDefinition(label="A", assay="Assay One", upper=1.0, units="nM")],
            "Assay Two": [BinDefinition(label="A", assay="Assay Two", upper=2.0, units="nM")],
        }
        assert decode_bin("A", legends) is None


class TestCrossReferenceError:
    """PRD EC-6 — [00508] says Table 1; the results are in Table 2."""

    def test_the_real_patent_error_is_detected(self) -> None:
        anomalies = detect_cross_reference_error(description_text())
        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.kind == "cross_reference_error"
        assert anomaly.severity in {"info", "warning"}  # non-blocking
        assert "Table 1" in anomaly.message
        assert "Table 2" in anomaly.message
        assert "00508" in anomaly.message

    def test_a_consistent_paragraph_is_not_flagged(self) -> None:
        text = "[00042] The results are shown in Table 2 below. Table 2 below lists them."
        assert detect_cross_reference_error(text) == []

    def test_empty_text_is_not_flagged(self) -> None:
        assert detect_cross_reference_error("") == []


class TestNoLegends:
    def test_text_without_legends_yields_nothing(self) -> None:
        legends, anomalies = parse_legends("This patent discloses no activity levels at all.")
        assert legends == {}
        assert anomalies == []

    def test_empty_text_is_safe(self) -> None:
        assert parse_legends("") == ({}, [])


def _labels_in(message: str) -> str:
    """The single bin label an anomaly message is about, e.g. `level F`."""
    found = re.findall(r"\blevel ([A-Z])\b", message)
    assert len(set(found)) == 1, message
    return found[0]
