"""Part 8 — structure standardization (PRD §9.5, Plan Part 8.1).

Covers R9.17 (the ChEMBL pipeline is the primary standardizer), R9.18 (salt
stripping), R9.19 (no tautomer canonicalization for display), R9.20 (record the
RDKit version), R9.21/EC-25 (metals and >7 boron pass through unstandardized),
R9.22/EC-19 (stereochemistry as drawn, undefined centers flagged) and EC-20
(sanitization failure is a review trigger, not a crash).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sarmine.structure.standardize import (
    StandardizedStructure,
    inchikey_from_smiles,
    rdkit_version,
    skeleton,
    standardize_smiles,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# PRD §9.4 — the verified reference compound (compound 5, source page 63).
REFERENCE_SMILES = "COc1ccc(-c2cc3ncn(C)c3cc2Nc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)cc1F"
REFERENCE_INCHIKEY = "WZPDSZGYLXZFEK-UHFFFAOYSA-N"

# The glutarimide C3 stereocentre of this chemotype: patents draw it flat about
# half the time (PRD R9.22). Thalidomide, flat and (S), is the canonical pair.
THALIDOMIDE_FLAT = "c1ccc2c(c1)C(=O)N(C1CCC(=O)NC1=O)C2=O"
THALIDOMIDE_S = "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"

# A real HCl salt and its free base (PRD R9.18, EC-18).
FLUOXETINE_FREE_BASE = "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"
FLUOXETINE_HCL = FLUOXETINE_FREE_BASE + ".Cl"

# Keto-enol tautomers of acetylacetone: different InChIKeys, one canonical form.
ACETYLACETONE_KETO = "O=C(C)CC(=O)C"
ACETYLACETONE_ENOL = "CC(O)=CC(=O)C"

BORON_CAGE_8 = "BBBBBBBB"  # >7 boron atoms — excluded by the ChEMBL pipeline
BORON_CAGE_7 = "BBBBBBB"  # exactly 7 — standardized normally
FERROCENE_LIKE = "[Fe]CCO"  # a metal on the pipeline's hardcoded list


# --------------------------------------------------------------------------
# import safety and version recording
# --------------------------------------------------------------------------


def test_module_import_does_not_pull_in_rdkit() -> None:
    """PRD §17.5 — no heavyweight imports at module scope."""
    code = (
        "import sys; import sarmine.structure.standardize; "
        "print('rdkit' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert proc.stdout.strip() == "False"


def test_rdkit_version_is_recorded_verbatim() -> None:
    """PRD R9.20 — canonicalization has changed between RDKit releases."""
    import rdkit

    assert rdkit_version() == rdkit.__version__
    assert rdkit_version()


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------


def test_inchikey_from_smiles_matches_the_reference_compound() -> None:
    assert inchikey_from_smiles(REFERENCE_SMILES) == REFERENCE_INCHIKEY


def test_inchikey_from_smiles_returns_none_for_junk() -> None:
    """PRD EC-20 — reject the channel; never raise."""
    assert inchikey_from_smiles("C1CC(") is None
    assert inchikey_from_smiles("") is None


def test_skeleton_is_the_first_fourteen_characters() -> None:
    """PRD R9.16/§5 — the skeleton key is the connectivity block."""
    assert skeleton(REFERENCE_INCHIKEY) == "WZPDSZGYLXZFEK"
    assert len(skeleton(REFERENCE_INCHIKEY) or "") == 14
    assert skeleton(None) is None


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_standardize_the_reference_compound() -> None:
    result = standardize_smiles(REFERENCE_SMILES)

    assert isinstance(result, StandardizedStructure)
    assert result.error is None
    assert result.sanitization_failed is False
    assert result.standardization_skipped is False
    assert result.inchikey_full == REFERENCE_INCHIKEY
    assert result.inchikey_skeleton == "WZPDSZGYLXZFEK"
    assert result.smiles
    assert inchikey_from_smiles(result.smiles) == REFERENCE_INCHIKEY


def test_standardize_none_is_an_empty_result_not_a_crash() -> None:
    for empty in (None, "", "   "):
        result = standardize_smiles(empty)
        assert result.inchikey_full is None
        assert result.error
        assert result.sanitization_failed is False


def test_checker_penalty_is_zero_for_a_clean_molecule() -> None:
    """PRD R9.17 — `check_molblock` runs first so high penalties can be quarantined."""
    assert standardize_smiles("O=C1NC(=O)C2(CCCC2)N1").checker_penalty == 0


def test_checker_penalty_is_recorded_for_a_flagged_molecule() -> None:
    """A stray radical is exactly the kind of OCSR artefact worth quarantining."""
    result = standardize_smiles("[CH3]")
    assert result.checker_penalty >= 6


# --------------------------------------------------------------------------
# R9.18 / EC-18 — salt stripping
# --------------------------------------------------------------------------


def test_salt_and_free_base_standardize_to_the_same_key() -> None:
    """PRD R9.18 — otherwise the drawn base and the tabulated salt become two compounds."""
    salt_raw = inchikey_from_smiles(FLUOXETINE_HCL)
    base_raw = inchikey_from_smiles(FLUOXETINE_FREE_BASE)
    assert salt_raw != base_raw  # unstandardized, they are two different compounds

    salt = standardize_smiles(FLUOXETINE_HCL)
    base = standardize_smiles(FLUOXETINE_FREE_BASE)

    assert salt.standardization_skipped is False
    assert salt.inchikey_full == base.inchikey_full
    assert salt.inchikey_full == base_raw
    assert "Cl" not in (salt.smiles or "")


# --------------------------------------------------------------------------
# R9.19 — tautomers: canonical for matching only, never for display
# --------------------------------------------------------------------------


def test_display_smiles_is_not_tautomer_canonicalized() -> None:
    """PRD R9.19 — RDKit's tautomer canonicalization is not idempotent across versions."""
    keto = standardize_smiles(ACETYLACETONE_KETO)
    enol = standardize_smiles(ACETYLACETONE_ENOL)

    assert keto.inchikey_full != enol.inchikey_full  # stored as drawn
    assert keto.smiles != enol.smiles
    assert enol.smiles is not None
    assert inchikey_from_smiles(enol.smiles) == enol.inchikey_full


def test_tautomer_canonical_smiles_is_a_separate_matching_key() -> None:
    keto = standardize_smiles(ACETYLACETONE_KETO)
    enol = standardize_smiles(ACETYLACETONE_ENOL)

    assert keto.smiles_tautomer_canonical
    assert keto.smiles_tautomer_canonical == enol.smiles_tautomer_canonical


# --------------------------------------------------------------------------
# R9.21 / EC-25 — the exclusion rule
# --------------------------------------------------------------------------


def test_metal_containing_molecule_passes_through_unstandardized() -> None:
    """PRD R9.21/EC-25 — check the flag; do not assume standardization happened."""
    result = standardize_smiles(FERROCENE_LIKE)

    assert result.standardization_skipped is True
    assert result.sanitization_failed is False
    assert result.inchikey_full  # still keyed, just not standardized


def test_more_than_seven_borons_passes_through_unstandardized() -> None:
    assert standardize_smiles(BORON_CAGE_8).standardization_skipped is True
    assert standardize_smiles(BORON_CAGE_7).standardization_skipped is False


# --------------------------------------------------------------------------
# R9.22 / EC-19 — stereochemistry exactly as drawn
# --------------------------------------------------------------------------


def test_undefined_stereocentre_is_flagged_not_enumerated() -> None:
    """PRD R9.22 — the glutarimide C3 centre is drawn flat about half the time."""
    flat = standardize_smiles(THALIDOMIDE_FLAT)

    assert flat.has_undefined_stereocenters is True
    assert flat.inchikey_full is not None
    assert flat.inchikey_full.split("-")[1] == "UHFFFAOYSA"  # no stereo layer invented


def test_defined_stereocentre_is_kept_and_not_flagged() -> None:
    defined = standardize_smiles(THALIDOMIDE_S)

    assert defined.has_undefined_stereocenters is False
    assert defined.inchikey_full is not None
    assert defined.inchikey_full.split("-")[1] != "UHFFFAOYSA"


def test_stereoisomers_share_a_skeleton_key_but_not_a_full_key() -> None:
    """PRD §5/R9.16 — this is precisely why the skeleton key exists (EC-19)."""
    flat = standardize_smiles(THALIDOMIDE_FLAT)
    defined = standardize_smiles(THALIDOMIDE_S)

    assert flat.inchikey_full != defined.inchikey_full
    assert flat.inchikey_skeleton == defined.inchikey_skeleton


# --------------------------------------------------------------------------
# EC-20 — sanitization failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["C1CC(", "not_a_smiles", "c1ccccc1C(=O)(=O)(=O)O"])
def test_sanitization_failure_is_reported_not_raised(bad: str) -> None:
    """PRD EC-20 — high-priority review trigger, never an exception."""
    result = standardize_smiles(bad)

    assert result.sanitization_failed is True
    assert result.inchikey_full is None
    assert result.error
    assert result.smiles == bad  # kept verbatim for the reviewer
