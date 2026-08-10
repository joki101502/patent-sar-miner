"""Structure standardization (PRD §9.5, Plan Part 8.1).

Implements R9.17–R9.22 and EC-18/EC-19/EC-20/EC-25. `chembl_structure_pipeline`
is the primary standardizer because it produced the ChEMBL structures any future
join would target; anything else guarantees join failures (R9.17).

Two keys leave this module and they are not interchangeable: `inchikey_full` /
`inchikey_skeleton` identify the compound as drawn or named, while
`smiles_tautomer_canonical` exists only for matching (R9.19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SKELETON_LENGTH = 14  # PRD §5 — the InChIKey connectivity block


@dataclass
class StandardizedStructure:
    """The standardized view of one structure, plus everything that went wrong."""

    smiles: str | None = None
    inchikey_full: str | None = None
    inchikey_skeleton: str | None = None
    smiles_tautomer_canonical: str | None = None
    has_undefined_stereocenters: bool = False
    standardization_skipped: bool = False
    checker_penalty: int = 0
    sanitization_failed: bool = False
    error: str | None = None


def rdkit_version() -> str:
    """PRD R9.20 — recorded on every compound row; canonicalization drifts."""
    import rdkit

    return str(rdkit.__version__)


def skeleton(inchikey: str | None) -> str | None:
    """The first 14 characters — the connectivity-only key (PRD R9.16, EC-19)."""
    if not inchikey:
        return None
    return inchikey[:SKELETON_LENGTH]


def inchikey_from_smiles(smiles: str) -> str | None:
    """PRD R9.16 — the identity of a structure. None when RDKit rejects it (EC-20)."""
    if not smiles or not smiles.strip():
        return None
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        key = MolToInchiKey(mol)
    except Exception:
        return None
    return key or None


def standardize_smiles(smiles: str | None) -> StandardizedStructure:
    """Run the ChEMBL pipeline over one SMILES (PRD R9.17).

    check_molblock -> get_parent_molblock -> standardize_molblock, with the
    exclusion flag surfaced rather than assumed away (R9.21).
    """
    if smiles is None or not smiles.strip():
        return StandardizedStructure(error="no SMILES supplied")

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # PRD EC-20 — a high-priority review trigger. Keep the string verbatim
        # so the reviewer can see what the channel produced.
        return StandardizedStructure(
            smiles=smiles,
            sanitization_failed=True,
            error="RDKit could not sanitize this SMILES",
        )

    from chembl_structure_pipeline import checker, standardizer

    molblock = Chem.MolToMolBlock(mol)
    penalty = _checker_penalty(checker.check_molblock(molblock))

    parent_block, excluded = standardizer.get_parent_molblock(molblock)
    if excluded:
        # PRD R9.21/EC-25 — metals or >7 boron atoms pass through untouched.
        final = mol
    else:
        standardized = standardizer.standardize_molblock(parent_block)
        final = Chem.MolFromMolBlock(standardized)
        if final is None:
            return StandardizedStructure(
                smiles=smiles,
                checker_penalty=penalty,
                sanitization_failed=True,
                error="the standardized molblock did not survive RDKit sanitization",
            )

    result = StandardizedStructure(
        smiles=Chem.MolToSmiles(final),
        checker_penalty=penalty,
        standardization_skipped=bool(excluded),
        has_undefined_stereocenters=_has_undefined_stereocenters(final),
        smiles_tautomer_canonical=_tautomer_canonical_smiles(final),
    )
    result.inchikey_full = inchikey_from_smiles(result.smiles or "")
    result.inchikey_skeleton = skeleton(result.inchikey_full)
    if result.inchikey_full is None:
        result.error = "the standardized structure produced no InChIKey"
    return result


def _checker_penalty(findings: Any) -> int:
    """The worst penalty the ChEMBL checker reported (PRD R9.17)."""
    scores = [int(score) for score, _message in findings or ()]
    return max(scores) if scores else 0


def _has_undefined_stereocenters(mol: Any) -> bool:
    """PRD R9.22/EC-19 — flag, never enumerate."""
    from rdkit.Chem import rdMolDescriptors

    return rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol) > 0


def _tautomer_canonical_smiles(mol: Any) -> str | None:
    """PRD R9.19 — a matching key only; never shown and never stored as display."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    try:
        return Chem.MolToSmiles(rdMolStandardize.TautomerEnumerator().Canonicalize(mol))
    except Exception:
        return None
