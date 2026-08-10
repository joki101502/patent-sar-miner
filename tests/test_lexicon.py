"""Tests for `sarmine.assay.lexicon` — PRD R10.8, R10.9, R10.10, R10.13, EC-27."""

from __future__ import annotations

from pathlib import Path

from sarmine.assay.lexicon import HeaderMatch, Lexicon, load_lexicon, reconstruct_header
from sarmine.config import get_config


class TestLexiconLoading:
    def test_default_path_is_the_configured_data_file(self) -> None:
        # PRD R10.8 — the lexicon is a data file, not code.
        assert get_config().assay_lexicon.is_file()
        lex = load_lexicon()
        assert isinstance(lex, Lexicon)

    def test_lexicon_is_versioned(self) -> None:
        lex = load_lexicon()
        assert isinstance(lex.version, str) and lex.version

    def test_a_new_endpoint_can_be_added_without_touching_code(self, tmp_path: Path) -> None:
        # PRD R10.8 — extensibility is the whole point of the data file.
        custom = tmp_path / "custom_lexicon.yaml"
        custom.write_text(
            "version: test-1\n"
            "endpoints:\n"
            "  - standard_type: GI50\n"
            "    bao_endpoint: BAO_0002993\n"
            "    aliases: [GI50]\n",
            "utf-8",
        )
        lex = load_lexicon(custom)
        assert lex.version == "test-1"
        match = lex.match("HCT116 GI50 (nM)")
        assert match is not None
        assert match.standard_type == "GI50"
        assert match.units == "nM"
        assert match.target == "HCT116"
        assert match.bao_endpoint == "BAO_0002993"


class TestReferencePatentHeaders:
    """The real headers of both reference patents (PRD §3.4, §3.8)."""

    def test_wiz_ec50_header(self) -> None:
        match = load_lexicon().match("WIZ EC50 (uM)")
        assert match is not None
        assert match.standard_type == "EC50"
        assert match.published_type == "WIZ EC50 (uM)"
        assert match.units == "uM"
        assert match.target == "WIZ"
        assert match.bao_endpoint == "BAO_0000188"  # PRD R10.13
        assert match.is_log_form is False
        assert match.confidence == 1.0
        assert match.matched_alias is not None

    def test_zbtb7a_ec50_header(self) -> None:
        match = load_lexicon().match("ZBTB7A EC50 (uM)")
        assert match is not None
        assert match.standard_type == "EC50"
        assert match.target == "ZBTB7A"
        assert match.units == "uM"

    def test_hbf_induction_header_carries_percent_units(self) -> None:
        match = load_lexicon().match("HbF Induction (%)")
        assert match is not None
        assert match.units == "%"
        assert match.target == "HbF"

    def test_hibit_dc50_header_of_the_second_reference_patent(self) -> None:
        # PRD AC-4.5 — units are explicit in this patent's header.
        match = load_lexicon().match("HiBiT DC50 (nM)")
        assert match is not None
        assert match.standard_type == "DC50"
        assert match.units == "nM"
        assert match.is_log_form is False

    def test_compound_number_columns_of_both_patents(self) -> None:
        lex = load_lexicon()
        for header in ("Compound No.", "Cmpd. No."):
            match = lex.match(header)
            assert match is not None, header
            assert match.standard_type == "CompoundNumber"
            assert match.bao_endpoint is None

    def test_procedure_reference_columns_are_not_endpoints(self) -> None:
        lex = load_lexicon()
        for header in ("Isoindoline synthesis", "Coupling procedure"):
            match = lex.match(header)
            assert match is not None, header
            assert match.standard_type == "ProcedureReference"
            assert match.bao_endpoint is None


class TestGenericEndpoints:
    """PRD R10.9 — the v1 endpoint set."""

    def test_bao_seed_mapping(self) -> None:
        lex = load_lexicon()
        expected = {
            "IC50 (nM)": ("IC50", "BAO_0000190"),
            "EC50 (nM)": ("EC50", "BAO_0000188"),
            "Ki (nM)": ("Ki", "BAO_0000192"),
            "Kd (nM)": ("Kd", "BAO_0000034"),
        }
        for header, (standard_type, bao) in expected.items():
            match = lex.match(header)
            assert match is not None, header
            assert (match.standard_type, match.bao_endpoint) == (standard_type, bao)

    def test_log_forms_standardize_to_their_linear_type(self) -> None:
        # PRD R10.2 — log forms are unwound, so the standardized type is linear.
        lex = load_lexicon()
        for header, standard_type in (("pIC50", "IC50"), ("pEC50", "EC50"), ("pKi", "Ki")):
            match = lex.match(header)
            assert match is not None, header
            assert match.standard_type == standard_type
            assert match.is_log_form is True
            assert match.units is None  # a log value is unitless

    def test_percent_inhibition_at_a_fixed_dose(self) -> None:
        match = load_lexicon().match("% Inhibition @ 10 uM")
        assert match is not None
        assert match.standard_type == "Inhibition"
        assert match.units == "%"

    def test_dmax_percent(self) -> None:
        match = load_lexicon().match("Dmax (%)")
        assert match is not None
        assert match.standard_type == "Dmax"
        assert match.units == "%"


class TestMatchingOrder:
    """PRD R10.8 — exact/alias, then high-threshold fuzzy, then give up."""

    def test_exact_alias_match_scores_full_confidence(self) -> None:
        match = load_lexicon().match("WIZ EC50 (uM)")
        assert match is not None and match.confidence == 1.0

    def test_ocr_damaged_header_recovers_by_fuzzy_match_at_reduced_confidence(self) -> None:
        match = load_lexicon().match("WIZ EC5O (uM)")  # capital O for zero
        assert match is not None
        assert match.standard_type == "EC50"
        assert get_config().header_fuzzy_threshold <= match.confidence < 1.0

    def test_nonsense_header_returns_none_rather_than_a_bad_guess(self) -> None:
        lex = load_lexicon()
        for header in ("Zorblax quibbling (frobnitz)", "Melting point (C)", ""):
            assert lex.match(header) is None, header

    def test_match_returns_a_headermatch_dataclass(self) -> None:
        match = load_lexicon().match("HiBiT DC50 (nM)")
        assert isinstance(match, HeaderMatch)


class TestReconstructHeader:
    """PRD R10.10 / EC-27 — the second reference patent's four-row split header."""

    def test_isoindoline_synthesis(self) -> None:
        assert reconstruct_header(["Iso-", "indoline", "syn-", "thesis"]) == "Isoindoline synthesis"

    def test_coupling_procedure(self) -> None:
        assert reconstruct_header(["Coupl-", "ing", "proce-", "dure"]) == "Coupling procedure"

    def test_hibit_dc50_nm(self) -> None:
        assert reconstruct_header(["HiBiT", "DC50", "(nM)"]) == "HiBiT DC50 (nM)"

    def test_reconstructed_header_then_matches_the_lexicon(self) -> None:
        # PRD AC-4.5 — reconstruct first, then match.
        header = reconstruct_header(["HiBiT", "DC50", "(nM)"])
        match = load_lexicon().match(header)
        assert match is not None
        assert (match.standard_type, match.units) == ("DC50", "nM")

    def test_blank_and_whitespace_rows_are_ignored(self) -> None:
        assert reconstruct_header(["Cmpd. No.", "", "   ", None or ""]) == "Cmpd. No."

    def test_single_row_is_unchanged(self) -> None:
        assert reconstruct_header(["HbF Induction (%)"]) == "HbF Induction (%)"

    def test_hyphen_followed_by_whitespace_is_still_a_word_wrap(self) -> None:
        assert reconstruct_header(["Iso- ", "indoline"]) == "Isoindoline"

    def test_empty_input_is_an_empty_header(self) -> None:
        assert reconstruct_header([]) == ""
