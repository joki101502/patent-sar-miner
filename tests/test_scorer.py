"""Part 9 — ranking, selectivity, computed properties, investment signal.

Covers PRD R12.1–R12.10, EC-7, EC-17, AC-6.1–AC-6.6 and the Appendix B.1
ground-truth activity table.
"""

from __future__ import annotations

from collections import Counter

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from sarmine.artifacts.schema import Compound, Measurement, Provenance
from sarmine.rank.scorer import (
    BIN_SCORES,
    HBA_DEFINITION,
    NO_ACTIVITY_DATA,
    TPSA_INCLUDE_S_AND_P,
    apply_investment_signal,
    compute_properties,
    detect_in_vivo,
    detect_in_vivo_terms,
    efficiency_metrics,
    rank_compounds,
    score_potency,
    score_selectivity,
    shortlist,
)

# Verbatim from the reference patent (tests/fixtures/source/WO2024097932A1.html).
# decisions.md F9 — these are the document's only two hits, and both are false
# positives for the signal the chemist wants.
IMAGING_BOILERPLATE = (
    "useful as therapeutic agents, e.g., cancer and inflammation therapeutic agents, "
    "research reagents, e.g., binding assay reagents, and diagnostic agents, e.g., "
    "in vivo imaging agents. All isotopic variations of the compounds of formula (I)"
)
ANTIBODY_REAGENT = (
    "During permeabilization step, cells were stained with PE-labelled Mouse Anti "
    "-Human Fetal Hemoglobin (1 : 10, clone 2D 12; BD Biosciences, Cat# BDB560041) "
    "incubated at room temperature for 20 minutes and protected from light."
)

# PRD Appendix B.2 — compound 5, MolScribe channel.
SMILES_5 = "COc1ccc(-c2cc3ncn(C)c3cc2Nc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)cc1F"
# A thioether: the molecule that exposes the CalcTPSA(includeSandP) gotcha.
SMILES_THIOETHER = "CSc1ccccc1CCN"

PROPERTY_FIELDS = {
    "mw",
    "clogp",
    "tpsa",
    "qed",
    "hbd_lipinski",
    "hba_lipinski",
    "rotb_strict",
    "heavy_atoms",
    "fsp3",
    "n_aromatic_rings",
}


# --- R12.1 — the bin table --------------------------------------------------


def test_bin_scores_match_the_prd_table():
    assert BIN_SCORES == {"A": 3, "B": 2, "C": 1, "D": 3, "E": 2, "F": 1, "G": 3, "H": 2, "I": 1}


# --- R12.6 — computed properties -------------------------------------------


def test_compute_properties_returns_exactly_the_compound_property_fields():
    props = compute_properties(SMILES_5)

    assert set(props) == PROPERTY_FIELDS
    assert props["mw"] == pytest.approx(527.51, abs=0.01)
    assert props["clogp"] == pytest.approx(3.533, abs=0.001)
    assert props["tpsa"] == pytest.approx(122.63, abs=0.01)
    assert props["qed"] == pytest.approx(0.3809, abs=0.0001)
    assert props["hbd_lipinski"] == 2
    assert props["hba_lipinski"] == 10
    assert props["rotb_strict"] == 5
    assert props["heavy_atoms"] == 39
    assert props["fsp3"] == pytest.approx(0.1786, abs=0.0001)
    assert props["n_aromatic_rings"] == 4


def test_compute_properties_is_null_for_missing_or_unparseable_smiles():
    for bad in (None, "", "not a molecule"):
        props = compute_properties(bad)
        assert set(props) == PROPERTY_FIELDS
        assert all(v is None for v in props.values())


def test_tpsa_gotcha_the_sulfur_flag_changes_the_answer():
    """PRD R12.6 — `CalcTPSA` defaults `includeSandP=False`; record which we used."""
    mol = Chem.MolFromSmiles(SMILES_THIOETHER)
    without_s = rdMolDescriptors.CalcTPSA(mol)
    with_s = rdMolDescriptors.CalcTPSA(mol, includeSandP=True)

    assert without_s == pytest.approx(26.02, abs=0.01)
    assert with_s == pytest.approx(51.32, abs=0.01)
    assert without_s != with_s

    assert TPSA_INCLUDE_S_AND_P is False
    assert compute_properties(SMILES_THIOETHER)["tpsa"] == pytest.approx(without_s)


def test_hba_gotcha_lipinski_differs_from_the_pharmacophore_count():
    """PRD R12.6 — `CalcNumLipinskiHBA` (all N+O) ≠ `CalcNumHBA`."""
    mol = Chem.MolFromSmiles(SMILES_5)
    lipinski = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    pharmacophore = rdMolDescriptors.CalcNumHBA(mol)

    assert (lipinski, pharmacophore) == (10, 7)
    assert lipinski != pharmacophore

    assert HBA_DEFINITION == "lipinski"
    assert compute_properties(SMILES_5)["hba_lipinski"] == lipinski


# --- R12.8 / EC-17 — efficiency metrics ------------------------------------


def test_efficiency_metrics_use_the_chembl_formulas():
    metrics = efficiency_metrics(8.0, mw=400.0, tpsa=80.0, heavy_atoms=30, clogp=3.0)

    assert metrics["LE"] == pytest.approx(1.37 * 8.0 / 30)
    assert metrics["BEI"] == pytest.approx(20.0)
    assert metrics["SEI"] == pytest.approx(10.0)
    assert metrics["LLE"] == pytest.approx(5.0)


def test_efficiency_metrics_are_undefined_without_a_pchembl():
    """PRD R12.8 / EC-17 — censored data yields no pChEMBL, so no LE or LLE."""
    metrics = efficiency_metrics(None, mw=400.0, tpsa=80.0, heavy_atoms=30, clogp=3.0)

    assert set(metrics) == {"LE", "BEI", "SEI", "LLE"}
    assert all(v is None for v in metrics.values())


def test_efficiency_metrics_tolerate_missing_or_zero_denominators():
    metrics = efficiency_metrics(8.0, mw=None, tpsa=0.0, heavy_atoms=0, clogp=None)

    assert metrics == {"LE": None, "BEI": None, "SEI": None, "LLE": None}


# --- R12.9 / AC-6.6 — in vivo detection and the investment signal ----------


def test_the_patents_only_two_hits_are_not_confirmed_in_vivo_findings():
    """AC-6.6 — an antibody reagent and imaging boilerplate are not in vivo work."""
    assert detect_in_vivo(IMAGING_BOILERPLATE) == []
    assert detect_in_vivo(ANTIBODY_REAGENT) == []
    assert detect_in_vivo(IMAGING_BOILERPLATE + " " + ANTIBODY_REAGENT) == []


def test_raw_term_hits_are_reported_separately_from_confirmed_findings():
    """The vocabulary does fire; the API keeps raw hits distinct from a finding."""
    assert detect_in_vivo_terms(IMAGING_BOILERPLATE) == ["in vivo"]
    assert detect_in_vivo_terms(ANTIBODY_REAGENT) == ["mouse"]


def test_genuine_in_vivo_efficacy_prose_is_detected():
    text = (
        "Compound 12 was dosed orally at 10 mg/kg in male CD-1 mice; the observed "
        "AUC was 1240 ng*h/mL with a Cmax of 310 ng/mL and oral bioavailability of 42%."
    )
    confirmed = detect_in_vivo(text)

    assert "mouse" in confirmed
    assert "AUC" in confirmed
    assert "Cmax" in confirmed
    assert "oral bioavailability" in confirmed


def test_vocabulary_does_not_fire_inside_unrelated_words():
    text = "The degradation rate was generated at ambient temperature for the pKa panel."

    assert detect_in_vivo_terms(text) == []


def test_investment_signal_records_a_reason_for_every_flag():
    compound = Compound(compound_id="c1", compound_local_id="12")

    apply_investment_signal(compound, in_examples=True, in_claims=True, in_prose=False)

    assert (compound.in_examples, compound.in_claims, compound.in_prose) == (True, True, False)
    assert compound.has_in_vivo is False
    reasons = " | ".join(compound.investment_reasons)
    assert "Examples" in reasons and "claims" in reasons
    assert "prose" not in reasons
    assert len(compound.investment_reasons) == 2


def test_investment_signal_names_the_in_vivo_terms_that_fired():
    compound = Compound(compound_id="c1", compound_local_id="12")

    apply_investment_signal(
        compound,
        in_examples=True,
        in_claims=False,
        in_prose=True,
        in_vivo_terms=["mouse", "AUC"],
    )

    assert compound.has_in_vivo is True
    assert any("mouse" in r and "AUC" in r for r in compound.investment_reasons)


def test_investment_signal_is_a_badge_of_reasons_not_an_opaque_number():
    """PRD R12.9 — display as a badge with reasons, never a single number."""
    compound = Compound(compound_id="c1", compound_local_id="12")

    apply_investment_signal(compound, in_examples=False, in_claims=False, in_prose=False)

    assert compound.investment_reasons == []
    assert compound.has_in_vivo is False


# --- PRD Appendix B.1 — the verified activity table, transcribed -----------

PUBNUM = "WO2024097932A1"

#: (compound number, HbF, WIZ EC50, ZBTB7A EC50); None is a blank source cell.
APPENDIX_B1: tuple[tuple[int, str | None, str, str], ...] = (
    (1, "A", "E", "G"),
    (2, "A", "D", "G"),
    (3, "A", "E", "G"),
    (4, "A", "E", "H"),
    (5, "A", "E", "H"),
    (6, "A", "D", "H"),
    (7, "A", "E", "H"),
    (8, "A", "E", "H"),
    (9, "A", "D", "H"),
    (10, "A", "D", "I"),
    (11, "A", "F", "I"),
    (12, "A", "D", "H"),
    (13, "A", "E", "H"),
    (14, "A", "F", "I"),
    (15, "A", "D", "H"),
    (16, "A", "D", "I"),
    (17, "A", "E", "H"),
    (18, "A", "D", "H"),
    (19, "A", "F", "I"),
    (20, "B", "D", "I"),
    (21, "B", "D", "H"),
    (22, "B", "E", "I"),
    (23, "B", "F", "I"),
    (24, "B", "F", "I"),
    (25, "B", "E", "I"),
    (26, "C", "E", "I"),
    (27, "C", "E", "I"),
    (28, "C", "F", "I"),
    (29, "C", "E", "I"),
    (30, "C", "E", "H"),
    (31, "B", "E", "I"),
    (32, "A", "D", "H"),
    (33, None, "E", "H"),
    (34, None, "E", "G"),
    (35, None, "E", "H"),
    (36, None, "E", "G"),
    (37, None, "D", "G"),
    (38, None, "E", "G"),
    (39, "B", "E", "H"),
    (40, "A", "D", "G"),
    (41, "A", "D", "G"),
    (42, "B", "F", "H"),
    (43, "A", "D", "G"),
    (44, "A", "E", "H"),
    (45, "A", "D", "G"),
    (46, "A", "D", "G"),
    (47, "A", "D", "G"),
    (48, "A", "D", "G"),
    (49, None, "D", "G"),
    (50, None, "E", "G"),
    (51, None, "E", "H"),
    (52, "A", "D", "I"),
    (53, None, "F", "I"),
    (54, None, "F", "I"),
)

ASSAYS: dict[str, tuple[str, str, bool, str]] = {
    # key: (assay_name_raw, target_raw, is_off_target, standard_type)
    "HbF": ("HbF Induction (%)", "HbF", False, "Induction"),
    "WIZ": ("WIZ EC50 (uM)", "WIZ", False, "EC50"),
    # PRD §3.6 / decisions.md J2 — ZBTB7A is the off-target.
    "ZBTB7A": ("ZBTB7A EC50 (uM)", "ZBTB7A", True, "EC50"),
}


def make_provenance(page_no: int = 186) -> Provenance:
    return Provenance(
        page_no=page_no,
        bbox=(0, 0, 100, 100),
        raster_width=2480,
        raster_height=3508,
        crop_path=f"crops/p{page_no:03d}_cell.png",
        source="pdf_ocr",
        extractor="tesseract@5.5.3",
    )


def make_measurement(compound_id: str, assay: str, letter: str) -> Measurement:
    name, target, is_off_target, standard_type = ASSAYS[assay]
    tail = compound_id.rsplit(":", 1)[-1]
    page_no = 186 if not tail.isdigit() or int(tail) < 32 else 187
    return Measurement(
        measurement_id=f"{compound_id}:{assay}",
        compound_id=compound_id,
        assay_group_key=f"{PUBNUM}::{name}",
        assay_name_raw=name,
        target_raw=target,
        is_off_target=is_off_target,
        published_type=name,
        published_value=letter,
        standard_type=standard_type,
        standard_relation="<",
        standard_value=None,  # PRD R10.5 — no midpoint imputation
        standard_units="nM",
        # PRD EC-17 — a letter bin is censored, so no pChEMBL and no LE/LLE.
        is_censored=True,
        censor_direction="upper_bound",
        bin_label_raw=letter,
        bin_score=BIN_SCORES[letter],
        provenance=make_provenance(page_no),
    )


def appendix_b1_compounds() -> list[Compound]:
    return [
        Compound(compound_id=f"{PUBNUM}:{n}", compound_local_id=str(n), compound_number=n)
        for n, _, _, _ in APPENDIX_B1
    ]


def appendix_b1_measurements() -> list[Measurement]:
    out: list[Measurement] = []
    for n, hbf, wiz, zbtb7a in APPENDIX_B1:
        for assay, letter in (("HbF", hbf), ("WIZ", wiz), ("ZBTB7A", zbtb7a)):
            if letter is None:
                continue  # EC-7 — a blank cell produces no measurement at all
            out.append(make_measurement(f"{PUBNUM}:{n}", assay, letter))
    return out


def test_appendix_b1_has_54_compounds_and_162_cells():
    assert len(APPENDIX_B1) == 54
    assert sum(len(row) - 1 for row in APPENDIX_B1) == 162
    assert len(appendix_b1_measurements()) == 151  # 162 cells − 11 blanks


def test_appendix_b1_blank_hbf_cells_are_the_eleven_verified_ones():
    """PRD Appendix B.1 / EC-7 / decisions.md F11."""
    blanks = {n for n, hbf, _, _ in APPENDIX_B1 if hbf is None}

    assert blanks == {33, 34, 35, 36, 37, 38, 49, 50, 51, 53, 54}
    assert len(blanks) == 11


def test_appendix_b1_wiz_and_zbtb7a_columns_are_complete():
    assert all(wiz and zbtb7a for _, _, wiz, zbtb7a in APPENDIX_B1)


def test_appendix_b1_bin_distributions_match_the_verified_counts():
    hbf = Counter(row[1] for row in APPENDIX_B1)
    wiz = Counter(row[2] for row in APPENDIX_B1)
    zbtb7a = Counter(row[3] for row in APPENDIX_B1)

    assert (hbf["A"], hbf["B"], hbf["C"], hbf[None]) == (29, 9, 5, 11)
    assert (wiz["D"], wiz["E"], wiz["F"]) == (21, 24, 9)
    assert (zbtb7a["G"], zbtb7a["H"], zbtb7a["I"]) == (16, 20, 18)


# --- R12.1 — potency ------------------------------------------------------


def test_potency_score_sums_the_available_bins():
    measurements = [
        make_measurement("c1", "HbF", "A"),
        make_measurement("c1", "WIZ", "D"),
        make_measurement("c1", "ZBTB7A", "H"),
    ]

    assert score_potency(measurements) == 3 + 3 + 2


def test_potency_score_is_none_when_nothing_was_measured():
    assert score_potency([]) is None


def test_a_blank_cell_is_never_scored_as_a_low_value():
    """PRD R12.9 / EC-7 / AC-6.2 — rank on available assays, never impute."""
    blank_hbf = [make_measurement("c1", "WIZ", "D"), make_measurement("c1", "ZBTB7A", "H")]
    lowest_hbf = [*blank_hbf, make_measurement("c1", "HbF", "C")]

    assert score_potency(blank_hbf) == 5
    assert score_potency(lowest_hbf) == 6
    # the blank compound is scored on 2 assays, not given HbF = C's single point
    assert score_potency(blank_hbf) == score_potency(lowest_hbf) - BIN_SCORES["C"]


def test_a_cell_with_no_decoded_bin_is_skipped_not_zeroed():
    unreadable = make_measurement("c1", "HbF", "A")
    unreadable.bin_score = None

    assert score_potency([unreadable, make_measurement("c1", "WIZ", "D")]) == 3


# --- R12.4 / R12.5 — selectivity is a separate, configurable axis ---------


def test_selectivity_is_target_minus_off_target():
    """PRD R12.4 / AC-6.3 — a D/I compound is maximally selective at +2."""
    measurements = [make_measurement("c1", "WIZ", "D"), make_measurement("c1", "ZBTB7A", "I")]

    assert score_selectivity(measurements, target="WIZ", off_target="ZBTB7A") == 2


def test_swapping_target_and_off_target_flips_the_sign():
    """PRD R12.5 — the assignment is an input; it inverts the ranking."""
    measurements = [make_measurement("c1", "WIZ", "D"), make_measurement("c1", "ZBTB7A", "I")]

    assert score_selectivity(measurements, target="ZBTB7A", off_target="WIZ") == -2


def test_selectivity_is_none_without_an_off_target_or_a_missing_arm():
    measurements = [make_measurement("c1", "WIZ", "D"), make_measurement("c1", "ZBTB7A", "I")]

    assert score_selectivity(measurements, target="WIZ", off_target=None) is None
    assert score_selectivity(measurements[:1], target="WIZ", off_target="ZBTB7A") is None


def test_potency_and_selectivity_are_separate_axes():
    """PRD R12.1 — potent ZBTB7A activity raises potency but is undesirable."""
    potent_off_target = [make_measurement("c", "WIZ", "D"), make_measurement("c", "ZBTB7A", "G")]
    selective = [make_measurement("c", "WIZ", "D"), make_measurement("c", "ZBTB7A", "I")]

    assert score_potency(potent_off_target) > score_potency(selective)
    assert score_selectivity(
        potent_off_target, target="WIZ", off_target="ZBTB7A"
    ) < score_selectivity(selective, target="WIZ", off_target="ZBTB7A")


# --- AC-6.1 – AC-6.5 — ranking over the ground-truth table -----------------


def test_only_compounds_10_16_20_52_reach_plus_two_selectivity():
    """PRD Appendix B.1 / AC-6.3 / Plan 9.4 — the sharpest correctness check."""
    compounds = appendix_b1_compounds()

    rank_compounds(compounds, appendix_b1_measurements())

    most_selective = {c.compound_number for c in compounds if c.selectivity_score == 2}
    assert most_selective == {10, 16, 20, 52}
    assert max(c.selectivity_score for c in compounds) == 2


def test_every_compound_with_activity_gets_a_potency_score_and_a_rank():
    """AC-6.1."""
    compounds = appendix_b1_compounds()

    rank_compounds(compounds, appendix_b1_measurements())

    assert all(c.potency_score is not None for c in compounds)
    assert all(c.rank is not None for c in compounds)
    assert sorted(c.rank for c in compounds) == list(range(1, 55))


def test_blank_hbf_compounds_are_ranked_on_their_remaining_assays():
    """AC-6.2 — and the rationale says so, rather than hiding the gap."""
    compounds = appendix_b1_compounds()

    rank_compounds(compounds, appendix_b1_measurements())

    by_number = {c.compound_number: c for c in compounds}
    blank = by_number[49]  # HbF blank, WIZ D, ZBTB7A G
    assert blank.potency_score == BIN_SCORES["D"] + BIN_SCORES["G"]
    assert any("2 of 3" in r for r in blank.rank_rationale)
    assert blank.rank is not None


def test_every_ranked_compound_carries_a_readable_rationale():
    """AC-6.5."""
    compounds = appendix_b1_compounds()

    rank_compounds(compounds, appendix_b1_measurements())

    for compound in compounds:
        assert compound.rank_rationale
        assert all(isinstance(r, str) and " " in r for r in compound.rank_rationale)
        assert any("potency" in r for r in compound.rank_rationale)


def test_tie_groups_are_exposed_rather_than_faked_into_precision():
    """PRD R12.3 / AC-6.4 — three bins over 54 compounds is tie-heavy."""
    compounds = appendix_b1_compounds()

    rank_compounds(compounds, appendix_b1_measurements())

    groups = Counter(c.rank_tie_group for c in compounds)
    assert len(groups) < 54
    assert max(groups.values()) > 1


def test_rank_compounds_mutates_in_place_and_returns_nothing():
    compounds = appendix_b1_compounds()

    assert rank_compounds(compounds, appendix_b1_measurements()) is None
    assert compounds[0].rank is not None


def test_a_compound_with_no_measurements_is_marked_no_activity_data():
    """PRD R11.4 / EC-23 — the unjoined Example still shows up, unranked."""
    orphan_row = Compound(compound_id="x", compound_local_id="Example 77")
    compounds = [*appendix_b1_compounds(), orphan_row]

    rank_compounds(compounds, appendix_b1_measurements())

    orphan = compounds[-1]
    assert orphan.potency_score is None
    assert orphan.rank is None
    assert NO_ACTIVITY_DATA in orphan.rank_rationale


# --- R12.2 — the tie-break ladder -----------------------------------------


def tie_break_case() -> list[Compound]:
    def compound(local_id: str, qed: float, mw: float) -> Compound:
        return Compound(
            compound_id=local_id, compound_local_id=local_id, qed=qed, mw=mw
        )

    return [
        compound("X", 0.10, 500.0),
        compound("Y", 0.90, 500.0),
        compound("Z", 0.50, 400.0),
        compound("W", 0.50, 300.0),
    ]


def tie_break_measurements() -> list[Measurement]:
    out: list[Measurement] = []
    for local_id, letters in (
        ("X", ("A", "F", "H")),  # potency 6, one assay at the top bin
        ("Y", ("B", "E", "H")),  # potency 6, none at the top bin
        ("Z", ("B", "E", "H")),
        ("W", ("B", "E", "H")),
    ):
        for assay, letter in zip(("HbF", "WIZ", "ZBTB7A"), letters):
            out.append(make_measurement(local_id, assay, letter))
    return out


def test_tie_breaks_run_top_bin_count_then_qed_then_mw():
    """PRD R12.2."""
    compounds = tie_break_case()

    rank_compounds(compounds, tie_break_measurements())

    ranks = {c.compound_local_id: c.rank for c in compounds}
    assert all(c.potency_score == 6 for c in compounds)
    assert ranks == {"X": 1, "Y": 2, "W": 3, "Z": 4}


def test_the_rationale_names_the_signal_that_drove_each_rank():
    """PRD R12.2 — always display which signal drove the rank."""
    compounds = tie_break_case()

    rank_compounds(compounds, tie_break_measurements())

    reasons = {c.compound_local_id: " | ".join(c.rank_rationale) for c in compounds}
    assert "top-bin count" in reasons["Y"]
    assert "QED" in reasons["W"]
    assert "MW" in reasons["Z"]


def test_tie_group_boundary_is_where_the_assay_data_stops_discriminating():
    compounds = tie_break_case()

    rank_compounds(compounds, tie_break_measurements())

    groups = {c.compound_local_id: c.rank_tie_group for c in compounds}
    assert groups["X"] != groups["Y"]
    assert groups["Y"] == groups["Z"] == groups["W"]


# --- R12.10 — the shortlist ------------------------------------------------


def test_shortlist_is_the_top_ten_with_tie_groups_visible():
    """PRD R12.10 / AC-6.4."""
    compounds = appendix_b1_compounds()
    rank_compounds(compounds, appendix_b1_measurements())

    top = shortlist(compounds)

    assert len(top) == 10
    assert [c.rank for c in top] == list(range(1, 11))
    assert all(c.rank_tie_group is not None for c in top)
    assert max(Counter(c.rank_tie_group for c in top).values()) > 1


def test_shortlist_excludes_compounds_with_no_activity_data():
    orphan_row = Compound(compound_id="x", compound_local_id="Example 77")
    compounds = [*appendix_b1_compounds(), orphan_row]
    rank_compounds(compounds, appendix_b1_measurements())

    assert all(c.rank is not None for c in shortlist(compounds, n=54))


def test_investment_signal_marks_32_of_the_54_compounds():
    """AC-6.6 — 32 of 54 clear the Examples bar and none has in vivo data."""
    compounds = appendix_b1_compounds()
    in_examples = set(range(1, 33))

    for compound in compounds:
        apply_investment_signal(
            compound,
            in_examples=compound.compound_number in in_examples,
            in_claims=False,
            in_prose=False,
            in_vivo_terms=detect_in_vivo(IMAGING_BOILERPLATE + " " + ANTIBODY_REAGENT),
        )

    assert sum(1 for c in compounds if c.in_examples) == 32
    assert not any(c.has_in_vivo for c in compounds)
