"""Part 8 — the cross-check, the core method (PRD §9.4, Plan Part 8.2).

Covers R9.13 (the five tiers, exercised with synthetic InChIKey pairs per
§20.2), R9.14 (OPSIN wins conflicts), R9.15 (SINGLE_SOURCE is neutral),
R9.16 (compare keys, never SMILES) and EC-9/EC-10/EC-11/EC-24 plus AC-3.6.
"""

from __future__ import annotations

import pytest

from sarmine.artifacts.schema import Compound
from sarmine.structure.crosscheck import CrosscheckResult, crosscheck, find_duplicates

# PRD §9.4 — the verified reference pair. The SMILES are textually completely
# different; only the InChIKey reveals that these are the same molecule.
OPSIN_SMILES = (
    "O=C1NC(CCC1N1C(C2=CC=CC(=C2C1=O)NC=1C(=CC2=C(N(C=N2)C)C1)C1=CC(=C(C=C1)OC)F)=O)=O"
)
MOLSCRIBE_SMILES = "COc1ccc(-c2cc3ncn(C)c3cc2Nc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)cc1F"
REFERENCE_INCHIKEY = "WZPDSZGYLXZFEK-UHFFFAOYSA-N"

# Thalidomide flat vs (S) — same connectivity, different stereo layer (EC-9).
THALIDOMIDE_FLAT_KEY = "UEJJHQNACJXSKW-UHFFFAOYSA-N"
THALIDOMIDE_S_KEY = "UEJJHQNACJXSKW-SECBINFHSA-N"

# A different molecule entirely (EC-10).
OTHER_KEY = "RTHCYVBBDHJXIQ-UHFFFAOYSA-N"


def _compound(local_id: str, inchikey: str | None, **kwargs: object) -> Compound:
    return Compound(
        compound_id=f"WO2024097932A1::{local_id}",
        compound_local_id=local_id,
        inchikey_full=inchikey,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# R9.16 — the point of the whole method
# --------------------------------------------------------------------------


def test_the_reference_pair_agrees_despite_different_smiles() -> None:
    """PRD R9.16/SPIKE S5 — two independent derivations, one InChIKey."""
    assert OPSIN_SMILES != MOLSCRIBE_SMILES  # this is the point

    result = crosscheck(
        inchikey_from_name=REFERENCE_INCHIKEY,
        inchikey_from_image=REFERENCE_INCHIKEY,
        smiles_from_name=OPSIN_SMILES,
        smiles_from_image=MOLSCRIBE_SMILES,
    )

    assert isinstance(result, CrosscheckResult)
    assert result.tier == "AGREE_FULL"
    assert result.structure_source == "name+image"
    assert result.inchikey_full == REFERENCE_INCHIKEY
    assert result.inchikey_skeleton == "WZPDSZGYLXZFEK"


def test_identical_smiles_are_never_required_for_agreement() -> None:
    """A SMILES comparison would have produced a false CONFLICT here."""
    same_molecule_different_strings = crosscheck(
        REFERENCE_INCHIKEY, REFERENCE_INCHIKEY, OPSIN_SMILES, MOLSCRIBE_SMILES
    )
    assert same_molecule_different_strings.tier != "CONFLICT"


# --------------------------------------------------------------------------
# R9.13 — all five tiers
# --------------------------------------------------------------------------


def test_tier_agree_full() -> None:
    result = crosscheck("AAAAAAAAAAAAAA-BBBBBBBBBB-C", "AAAAAAAAAAAAAA-BBBBBBBBBB-C", "C", "C")
    assert result.tier == "AGREE_FULL"
    assert result.structure_source == "name+image"


def test_tier_agree_skeleton() -> None:
    """PRD EC-9 — the channels disagree on stereochemistry only."""
    result = crosscheck(THALIDOMIDE_FLAT_KEY, THALIDOMIDE_S_KEY, "flat", "wedged")

    assert result.tier == "AGREE_SKELETON"
    assert result.structure_source == "name+image"
    assert result.inchikey_skeleton == "UEJJHQNACJXSKW"
    assert "stereo" in result.detail.lower()


def test_tier_conflict() -> None:
    """PRD EC-10 — the channels disagree on connectivity."""
    result = crosscheck(THALIDOMIDE_FLAT_KEY, OTHER_KEY, "from-name", "from-image")

    assert result.tier == "CONFLICT"
    assert result.inchikey_skeleton == "UEJJHQNACJXSKW"


def test_tier_single_source_name_only() -> None:
    result = crosscheck(THALIDOMIDE_FLAT_KEY, None, "from-name", None)

    assert result.tier == "SINGLE_SOURCE"
    assert result.structure_source == "name"
    assert result.smiles_final == "from-name"
    assert result.inchikey_full == THALIDOMIDE_FLAT_KEY


def test_tier_single_source_image_only() -> None:
    result = crosscheck(None, THALIDOMIDE_FLAT_KEY, None, "from-image")

    assert result.tier == "SINGLE_SOURCE"
    assert result.structure_source == "image"
    assert result.smiles_final == "from-image"
    assert result.inchikey_full == THALIDOMIDE_FLAT_KEY


def test_tier_none() -> None:
    result = crosscheck(None, None, None, None)

    assert result.tier == "NONE"
    assert result.structure_source == "none"
    assert result.smiles_final is None
    assert result.inchikey_full is None
    assert result.inchikey_skeleton is None


def test_blank_strings_count_as_absent_channels() -> None:
    """OPSIN emits an empty line for a name it could not parse."""
    result = crosscheck("", "   ", "", None)
    assert result.tier == "NONE"


def test_every_tier_in_the_prd_table_is_reachable() -> None:
    """PRD §20.2 — synthetic InChIKey pairs exercise all five tiers."""
    tiers = {
        crosscheck("AAAAAAAAAAAAAA-BBBBBBBBBB-C", "AAAAAAAAAAAAAA-BBBBBBBBBB-C", "a", "a").tier,
        crosscheck("AAAAAAAAAAAAAA-BBBBBBBBBB-C", "AAAAAAAAAAAAAA-ZZZZZZZZZZ-C", "a", "b").tier,
        crosscheck("AAAAAAAAAAAAAA-BBBBBBBBBB-C", "QQQQQQQQQQQQQQ-BBBBBBBBBB-C", "a", "b").tier,
        crosscheck("AAAAAAAAAAAAAA-BBBBBBBBBB-C", None, "a", None).tier,
        crosscheck(None, None, None, None).tier,
    }
    assert tiers == {"AGREE_FULL", "AGREE_SKELETON", "CONFLICT", "SINGLE_SOURCE", "NONE"}


# --------------------------------------------------------------------------
# R9.14 — OPSIN wins conflicts
# --------------------------------------------------------------------------


def test_opsin_wins_a_conflict() -> None:
    """PRD R9.14 — OPSIN fails loudly; OCSR fails silently and plausibly."""
    result = crosscheck(THALIDOMIDE_FLAT_KEY, OTHER_KEY, "from-name", "from-image")

    assert result.smiles_final == "from-name"
    assert result.inchikey_full == THALIDOMIDE_FLAT_KEY
    assert result.structure_source == "name"


def test_a_conflict_records_both_candidates_for_the_reviewer() -> None:
    """PRD R9.14 — 'do NOT auto-pick a winner in the UI'; show both."""
    result = crosscheck(THALIDOMIDE_FLAT_KEY, OTHER_KEY, "from-name", "from-image")

    assert THALIDOMIDE_FLAT_KEY in result.detail
    assert OTHER_KEY in result.detail
    assert "unresolved" in result.detail.lower()


def test_opsin_also_supplies_the_stored_value_on_a_stereo_disagreement() -> None:
    result = crosscheck(THALIDOMIDE_S_KEY, THALIDOMIDE_FLAT_KEY, "from-name", "from-image")

    assert result.tier == "AGREE_SKELETON"
    assert result.smiles_final == "from-name"
    assert result.inchikey_full == THALIDOMIDE_S_KEY


# --------------------------------------------------------------------------
# R9.15 — SINGLE_SOURCE is neutral, not low confidence
# --------------------------------------------------------------------------


def test_single_source_is_not_treated_as_suspicious() -> None:
    """PRD R9.15/EC-11 — only ~31.2% of SureChEMBL compounds are multi-source.

    Flagging the majority case would flood the review queue.
    """
    result = crosscheck(THALIDOMIDE_FLAT_KEY, None, "from-name", None)

    assert result.tier == "SINGLE_SOURCE"
    assert result.smiles_final == "from-name"  # accepted, not withheld
    assert "neutral" in result.detail.lower()
    assert "conflict" not in result.detail.lower()


@pytest.mark.parametrize(
    ("name_key", "image_key"),
    [(THALIDOMIDE_FLAT_KEY, None), (None, THALIDOMIDE_FLAT_KEY)],
)
def test_single_source_accepts_the_structure_from_either_channel(
    name_key: str | None, image_key: str | None
) -> None:
    result = crosscheck(name_key, image_key, "from-name", "from-image")

    assert result.tier == "SINGLE_SOURCE"
    assert result.inchikey_full == THALIDOMIDE_FLAT_KEY
    assert result.smiles_final is not None


def test_a_channel_with_a_key_but_no_smiles_still_tiers() -> None:
    """Presence is decided by the key, because the key is what we compare (R9.16)."""
    result = crosscheck(REFERENCE_INCHIKEY, REFERENCE_INCHIKEY, None, MOLSCRIBE_SMILES)

    assert result.tier == "AGREE_FULL"
    assert result.smiles_final == MOLSCRIBE_SMILES


# --------------------------------------------------------------------------
# EC-24 / AC-3.6 — duplicates
# --------------------------------------------------------------------------


def test_no_duplicates_among_distinct_compounds() -> None:
    compounds = [_compound("1", THALIDOMIDE_FLAT_KEY), _compound("2", OTHER_KEY)]
    assert find_duplicates(compounds) == []


def test_identical_full_keys_are_reported_as_duplicates() -> None:
    """PRD AC-3.6 — no two distinct compounds may share a full InChIKey unflagged."""
    compounds = [
        _compound("1", THALIDOMIDE_FLAT_KEY),
        _compound("2", THALIDOMIDE_FLAT_KEY),
    ]
    pairs = find_duplicates(compounds)

    assert pairs == [(compounds[0].compound_id, compounds[1].compound_id)]


def test_duplicates_merge_on_the_skeleton_key() -> None:
    """PRD EC-24 — merge on skeleton key; flag; do not silently drop."""
    compounds = [
        _compound("1", THALIDOMIDE_FLAT_KEY),
        _compound("2", THALIDOMIDE_S_KEY),
    ]
    pairs = find_duplicates(compounds)

    assert pairs == [(compounds[0].compound_id, compounds[1].compound_id)]


def test_find_duplicates_does_not_drop_or_mutate_anything() -> None:
    """PRD EC-24 — the caller flags; nothing disappears."""
    compounds = [_compound("1", THALIDOMIDE_FLAT_KEY), _compound("2", THALIDOMIDE_FLAT_KEY)]
    find_duplicates(compounds)

    assert len(compounds) == 2
    assert all(c.potential_duplicate is False for c in compounds)


def test_compounds_without_a_key_are_never_duplicates_of_each_other() -> None:
    compounds = [_compound("1", None), _compound("2", None)]
    assert find_duplicates(compounds) == []


def test_three_way_collision_reports_every_pair() -> None:
    compounds = [_compound(str(i), THALIDOMIDE_FLAT_KEY) for i in (1, 2, 3)]
    pairs = find_duplicates(compounds)

    assert len(pairs) == 3
    assert (compounds[0].compound_id, compounds[2].compound_id) in pairs


def test_ac_3_6_holds_once_reported_duplicates_are_flagged() -> None:
    """AC-3.6 — the acceptance criterion, stated as the invariant it is."""
    compounds = [_compound("1", THALIDOMIDE_FLAT_KEY), _compound("2", THALIDOMIDE_FLAT_KEY)]
    flagged = {cid for pair in find_duplicates(compounds) for cid in pair}
    for compound in compounds:
        if compound.compound_id in flagged:
            compound.potential_duplicate = True

    keys_by_unflagged = [
        c.inchikey_full for c in compounds if c.inchikey_full and not c.potential_duplicate
    ]
    assert len(keys_by_unflagged) == len(set(keys_by_unflagged))
