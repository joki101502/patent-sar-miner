"""Part 6.5 — the join (PRD §11, Plan Part 6.5).

Covers R11.1–R11.6, EC-4, EC-23, EC-24 and AC-5.1–AC-5.4.
"""

from __future__ import annotations

from sarmine.artifacts.schema import Compound
from sarmine.join.linker import (
    NAME_SIMILARITY_THRESHOLD,
    NO_ACTIVITY_DATA,
    ActivityRow,
    ExampleEntry,
    join,
    link_activity_rows,
    link_examples,
    normalized_name_similarity,
    validate_monotonic,
)

PUBNUM = "WO2024097932A1"

# PRD Appendix B.2 — the verified compound on page 63.
NAME_5 = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-"
    "1H-benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
)
NAME_5_WITH_STEREO = "(S)-" + NAME_5
# decisions.md F10 — Example 1's name: same chemotype, different compound.
NAME_SIBLING = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((1-methyl-6-phenoxy-1H-indazol-5-yl)amino)"
    "isoindoline-1,3-dione"
)


# --- R11.5 / EC-4 / AC-5.4 — flag, never interpolate -----------------------


def test_clean_sequence_produces_no_anomalies():
    assert validate_monotonic(list(range(1, 55))) == []


def test_gap_is_flagged_and_the_missing_number_is_named():
    anomalies = validate_monotonic([1, 2, 4, 5])
    assert len(anomalies) == 1
    assert anomalies[0].kind == "compound_number_gap"
    assert "3" in anomalies[0].message


def test_unreadable_number_is_flagged_not_guessed():
    anomalies = validate_monotonic([1, None, 3])
    unreadable = [a for a in anomalies if "unreadable" in a.message]
    assert len(unreadable) == 1
    assert unreadable[0].kind == "compound_number_gap"


def test_broken_monotonicity_is_flagged():
    anomalies = validate_monotonic([1, 3, 2])
    assert len(anomalies) == 1
    assert anomalies[0].kind == "compound_number_gap"
    assert "monotonic" in anomalies[0].message


def test_duplicate_number_is_flagged_once():
    anomalies = validate_monotonic([1, 2, 2, 3])
    assert len(anomalies) == 1
    assert "duplicate" in anomalies[0].message


# --- R11.2 / R11.3 — name-similarity fallback ------------------------------


def test_identical_names_score_one():
    assert normalized_name_similarity(NAME_5, NAME_5) == 1.0


def test_similarity_ignores_case_whitespace_and_punctuation_noise():
    noisy = "2-(2,6-Dioxopiperidin-3-YL) isoindoline-1,3-dione"
    clean = "2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione"
    assert normalized_name_similarity(noisy, clean) == 1.0


def test_stereo_prefix_alone_does_not_lower_similarity():
    """PRD R11.3 — the glutarimide centre is drawn flat about half the time."""
    assert normalized_name_similarity(NAME_5, NAME_5_WITH_STEREO) == 1.0


def test_a_different_compound_of_the_same_chemotype_stays_below_threshold():
    score = normalized_name_similarity(NAME_5, NAME_SIBLING)
    assert score < NAME_SIMILARITY_THRESHOLD


def test_empty_name_scores_zero():
    assert normalized_name_similarity("", NAME_5) == 0.0


# --- fixtures ---------------------------------------------------------------


def skeleton(n: int) -> str:
    return f"C{n:04d}".ljust(14, "X")


def inchikey(n: int, stereo: str = "UHFFFAOYSA") -> str:
    return f"{skeleton(n)}-{stereo}-N"


def make_compound(n: int, **overrides) -> Compound:
    kwargs: dict = dict(
        compound_id=f"{PUBNUM}:{n}",
        compound_local_id=str(n),
        compound_number=n,
        inchikey_full=inchikey(n),
        inchikey_from_name=inchikey(n),
        inchikey_from_image=inchikey(n),
        smiles_from_name=f"C{n}",
        structure_source="name+image",
    )
    kwargs.update(overrides)
    return Compound(**kwargs)


def make_activity_row(n: int, **overrides) -> ActivityRow:
    kwargs: dict = dict(
        compound_number=n,
        values={"HbF Induction (%)": "A", "WIZ EC50 (uM)": "D", "ZBTB7A EC50 (uM)": "I"},
        page_no=186,
    )
    kwargs.update(overrides)
    return ActivityRow(**kwargs)


def make_example(n: int, **overrides) -> ExampleEntry:
    kwargs: dict = dict(
        local_id=f"Example {n}",
        name=f"example compound number {n}",
        inchikey=inchikey(n),
        smiles=f"C{n}",
        page_no=100 + n,
        has_nmr=True,
        has_ms=True,
    )
    kwargs.update(overrides)
    return ExampleEntry(**kwargs)


# --- R11.1 / R11.6 — the compound-number join ------------------------------


def test_activity_rows_join_on_compound_number():
    compounds = [make_compound(n) for n in (1, 2, 3)]
    rows = [make_activity_row(n) for n in (1, 2, 3)]

    matched, anomalies = link_activity_rows(compounds, rows)

    assert set(matched) == {1, 2, 3}
    assert matched[2].values["WIZ EC50 (uM)"] == "D"
    assert anomalies == []


def test_joined_row_records_its_channels_and_their_agreement():
    """PRD R11.6 / AC-5.3."""
    result = join([make_compound(1)], [make_activity_row(1)], [])

    (compound,) = result.compounds
    assert compound.join_method == "compound_number"
    assert "compound_number" in compound.join_channels
    assert "activity_number" in compound.join_channels
    assert "name" in compound.join_channels and "image" in compound.join_channels
    assert "channels_agree" in compound.join_channels


def test_stereo_only_channel_disagreement_records_skeleton_agreement():
    """PRD R11.3 — full keys differ only in the stereo layer."""
    compound = make_compound(1, inchikey_from_image=inchikey(1, stereo="ABCDEFGHSA"))
    result = join([compound], [make_activity_row(1)], [])

    assert "channels_agree_skeleton" in result.compounds[0].join_channels


def test_channel_conflict_is_recorded_never_hidden():
    compound = make_compound(1, inchikey_from_image=inchikey(99))
    result = join([compound], [make_activity_row(1)], [])

    assert "channels_conflict" in result.compounds[0].join_channels


def test_gap_is_flagged_and_no_row_is_synthesized_for_the_missing_number():
    """PRD R11.5 / EC-4 / AC-5.4."""
    numbers = [1, 2, 4, 5]
    result = join(
        [make_compound(n) for n in numbers],
        [make_activity_row(n) for n in numbers],
        [],
    )

    assert [c.compound_number for c in result.compounds] == numbers
    gaps = [a for a in result.anomalies if a.kind == "compound_number_gap"]
    assert gaps and any("3" in a.message for a in gaps)


def test_activity_row_with_no_matching_compound_is_flagged():
    rows = [make_activity_row(1), make_activity_row(9)]
    _, anomalies = link_activity_rows([make_compound(1)], rows)

    assert any("9" in a.message for a in anomalies)


# --- R11.2 / R11.3 — Examples ↔ compound-table rows ------------------------


def test_examples_join_by_inchikey_from_name_first():
    compounds = [make_compound(n) for n in (1, 2)]
    examples = [make_example(2)]

    links, _ = link_examples(compounds, examples)

    assert links == {"Example 2": f"{PUBNUM}:2"}


def test_examples_fall_back_to_name_similarity_when_no_inchikey():
    compounds = [make_compound(1)]
    examples = [make_example(1, inchikey=None, name=NAME_5_WITH_STEREO)]

    result = join(compounds, [make_activity_row(1)], examples, names={f"{PUBNUM}:1": NAME_5})

    assert result.n_joined_examples == 1
    assert "name_similarity" in result.compounds[0].join_method


def test_examples_fall_back_to_skeleton_key_when_stereo_differs():
    """PRD R11.3 — a stereo-strict join splits one compound into two."""
    compounds = [make_compound(1)]
    examples = [make_example(1, inchikey=inchikey(1, stereo="ABCDEFGHSA"))]

    result = join(compounds, [make_activity_row(1)], examples)

    assert result.n_joined_examples == 1
    assert "inchikey_skeleton" in result.compounds[0].join_method


def test_agreement_across_join_methods_raises_confidence():
    """PRD R11.2 — agreement across methods is itself evidence."""
    compounds = [make_compound(1)]
    examples = [make_example(1, name=NAME_5)]

    result = join(compounds, [make_activity_row(1)], examples, names={f"{PUBNUM}:1": NAME_5})

    assert "methods_agree" in result.compounds[0].join_channels


def test_two_examples_cannot_claim_the_same_compound_row():
    compounds = [make_compound(1)]
    examples = [make_example(1), make_example(1, local_id="Example 1b")]

    result = join(compounds, [make_activity_row(1)], examples)

    assert result.n_joined_examples == 1
    assert result.unjoined_examples == ["Example 1b"]


def test_link_examples_mutates_nothing():
    compounds = [make_compound(1)]
    before = compounds[0].model_dump()

    link_examples(compounds, [make_example(1)])

    assert compounds[0].model_dump() == before


# --- R11.4 / EC-23 — unjoined Examples still appear ------------------------


def test_unjoined_example_still_produces_a_sar_row():
    compounds = [make_compound(1)]
    examples = [make_example(1), make_example(77)]

    result = join(compounds, [make_activity_row(1)], examples)

    assert result.unjoined_examples == ["Example 77"]
    orphan = next(c for c in result.compounds if c.example_local_id == "Example 77")
    assert orphan.compound_number is None
    assert orphan.in_examples is True
    assert NO_ACTIVITY_DATA in orphan.rank_rationale
    assert orphan.compound_id.startswith(f"{PUBNUM}:")


# --- EC-24 — duplicate structures -----------------------------------------


def test_duplicate_inchikeys_are_flagged_never_dropped():
    compounds = [make_compound(1), make_compound(2, inchikey_full=inchikey(1))]

    result = join(compounds, [make_activity_row(1), make_activity_row(2)], [])

    assert len(result.compounds) == 2
    assert all(c.potential_duplicate for c in result.compounds)
    assert any(a.kind == "duplicate_structure" for a in result.anomalies)


def test_join_does_not_mutate_its_inputs():
    compounds = [make_compound(1)]
    before = compounds[0].model_dump()

    join(compounds, [make_activity_row(1)], [make_example(1)])

    assert compounds[0].model_dump() == before


# --- AC-5.1 / AC-5.2 — full-scale fixtures ---------------------------------


def test_all_54_compound_rows_join_to_activity_rows():
    """AC-5.1."""
    compounds = [make_compound(n) for n in range(1, 55)]
    rows = [make_activity_row(n) for n in range(1, 55)]

    result = join(compounds, rows, [])

    assert result.n_joined_activity == 54
    assert all(c.join_method for c in result.compounds)
    assert [a for a in result.anomalies if a.kind == "compound_number_gap"] == []


def test_at_least_30_of_32_examples_join_to_compound_rows():
    """AC-5.2."""
    compounds = [make_compound(n) for n in range(1, 55)]
    rows = [make_activity_row(n) for n in range(1, 55)]
    examples = [make_example(n) for n in range(1, 33)]

    result = join(compounds, rows, examples)

    assert result.n_joined_examples >= 30
    assert result.n_joined_examples == 32
    assert sum(1 for c in result.compounds if c.in_examples) == 32
    assert len(result.compounds) == 54
