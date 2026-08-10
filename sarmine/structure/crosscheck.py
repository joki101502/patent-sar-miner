"""The cross-check — two independent derivations reconciled (PRD §9.4, Plan Part 8.2).

This is the core method of the system. One channel is a deterministic grammar
over OCR'd text (OPSIN), the other a neural model over pixels (MolScribe); they
fail in uncorrelated ways, so their agreement is a real confidence signal.

Implements R9.13 (the five tiers), R9.14 (OPSIN wins conflicts), R9.15
(SINGLE_SOURCE is neutral) and R9.16 (compare InChIKeys, never SMILES), plus
EC-9, EC-10, EC-11 and EC-24/AC-3.6 via `find_duplicates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sarmine.artifacts.schema import Compound, CrosscheckTier, StructureSource
from sarmine.structure.standardize import skeleton


@dataclass
class CrosscheckResult:
    """The tier, the structure that won, and why — all three go into the UI."""

    tier: CrosscheckTier
    smiles_final: str | None
    structure_source: StructureSource
    inchikey_full: str | None
    inchikey_skeleton: str | None
    detail: str


def crosscheck(
    inchikey_from_name: str | None,
    inchikey_from_image: str | None,
    smiles_from_name: str | None,
    smiles_from_image: str | None,
) -> CrosscheckResult:
    """Assign one of the five confidence tiers in PRD R9.13.

    Channel presence is decided by the InChIKey, never by the SMILES string:
    the two channels routinely emit textually different SMILES for the same
    molecule (PRD R9.16, verified in SPIKE S5).
    """
    name_key = _clean(inchikey_from_name)
    image_key = _clean(inchikey_from_image)
    name_smiles = _clean(smiles_from_name)
    image_smiles = _clean(smiles_from_image)

    if name_key is None and image_key is None:
        return CrosscheckResult(
            tier="NONE",
            smiles_final=None,
            structure_source="none",
            inchikey_full=None,
            inchikey_skeleton=None,
            detail="neither channel produced a parseable structure; high-priority review",
        )

    if name_key is None or image_key is None:
        from_name = name_key is not None
        key = name_key or image_key
        # Fall back to the other channel's SMILES only when the winning channel
        # gave a key but no string to display.
        winning_smiles = (
            (name_smiles or image_smiles) if from_name else (image_smiles or name_smiles)
        )
        return CrosscheckResult(
            tier="SINGLE_SOURCE",
            smiles_final=winning_smiles,
            structure_source="name" if from_name else "image",
            inchikey_full=key,
            inchikey_skeleton=skeleton(key),
            # PRD R9.15 — most compounds have only one modality; treating this
            # as suspicious would flood the review queue with the majority case.
            detail=(
                f"only the {'name' if from_name else 'image'} channel produced a structure; "
                "neutral confidence, accepted on that channel's own confidence"
            ),
        )

    if name_key == image_key:
        return CrosscheckResult(
            tier="AGREE_FULL",
            smiles_final=name_smiles or image_smiles,
            structure_source="name+image",
            inchikey_full=name_key,
            inchikey_skeleton=skeleton(name_key),
            detail="both channels produced the same full InChIKey; auto-accepted",
        )

    # PRD R9.14 — OPSIN supplies the stored value from here on: it is
    # deterministic and fails loudly, while OCSR fails silently and plausibly.
    if skeleton(name_key) == skeleton(image_key):
        return CrosscheckResult(
            tier="AGREE_SKELETON",
            smiles_final=name_smiles or image_smiles,
            structure_source="name+image",
            inchikey_full=name_key,
            inchikey_skeleton=skeleton(name_key),
            detail=(
                "connectivity agrees but the stereochemistry layers differ "
                f"(name {name_key}, image {image_key}); "
                "accepted as drawn, stereo disagreement flagged, medium review priority"
            ),
        )

    return CrosscheckResult(
        tier="CONFLICT",
        smiles_final=name_smiles or image_smiles,
        # Not "name+image": the two channels did not corroborate each other, so
        # the stored structure comes from the name channel alone.
        structure_source="name",
        inchikey_full=name_key,
        inchikey_skeleton=skeleton(name_key),
        detail=(
            "the channels disagree on connectivity and the row is unresolved: "
            f"name {name_key} ({name_smiles or 'no SMILES'}) vs "
            f"image {image_key} ({image_smiles or 'no SMILES'}); "
            "the name channel supplies the stored value (PRD R9.14), "
            "high review priority"
        ),
    )


def find_duplicates(compounds: Sequence[Compound]) -> list[tuple[str, str]]:
    """Every pair of distinct compounds that resolve to the same structure.

    Grouping is on the skeleton key, so a stereoisomer pair drawn flat in one
    row and wedged in another is caught too (PRD EC-24, EC-19). Pure reporting:
    the caller sets `potential_duplicate`, and nothing is ever dropped (AC-3.6).
    """
    by_skeleton: dict[str, list[str]] = {}
    for compound in compounds:
        key = skeleton(compound.inchikey_full)
        if key is None:
            continue
        by_skeleton.setdefault(key, []).append(compound.compound_id)

    pairs: list[tuple[str, str]] = []
    for ids in by_skeleton.values():
        distinct = list(dict.fromkeys(ids))
        for i, left in enumerate(distinct):
            for right in distinct[i + 1 :]:
                pairs.append((left, right))
    return pairs


def _clean(value: str | None) -> str | None:
    """OPSIN emits an empty line for a name it could not parse."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
