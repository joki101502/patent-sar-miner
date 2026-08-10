"""Review-queue construction (PRD §13.2, R13.1, R13.2, Plan Part 10.1).

At ~60% end-to-end accuracy the review interface is the product (PRD §13.1),
so the queue must surface exactly the rows a chemist should adjudicate — every
trigger in the normative §13.2 table, at its stated priority, and nothing else.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rdkit import Chem

from sarmine.artifacts.schema import Compound, DocumentAnomaly, Measurement, Provenance
from sarmine.config import get_config
from sarmine.review.edits import missing_provenance  # noqa: F401 — re-exported for the app

Priority = Literal["high", "medium", "info"]

#: Anomalies are document-scoped: they belong to no single compound row.
DOCUMENT_SCOPE = "__document__"

_PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "info": 2}

#: PRD §13.2 — document anomalies that are also per-row review work.
_ANOMALY_TRIGGERS: dict[str, tuple[Priority, str]] = {
    "detector_disagreement": ("medium", "Morphology detector and TATR disagree on the cell grid"),
    "rotation_uncertain": ("medium", "Page rotation applied at a low verification score"),
}


@dataclass
class ReviewItem:
    """One adjudicable finding, carrying everything PRD R13.3 puts on screen."""

    compound_id: str
    priority: Priority
    trigger: str
    reason: str
    page_no: int | None
    bbox: tuple[int, int, int, int] | None
    crop_path: str | None
    smiles_from_name: str | None
    smiles_from_image: str | None
    inchikey_from_name: str | None
    inchikey_from_image: str | None
    crosscheck_tier: str | None
    measurement_id: str | None = None


def _sanitizes(smiles: str | None) -> bool:
    if not smiles:
        return True
    return Chem.MolFromSmiles(smiles) is not None


#: Leading relation, then a number: ">10,000", "5.6 ± 0.3", "-1.2" are all values.
_NUMERIC = re.compile(r"^\s*[<>=~≤≥]*\s*[+-]?\d+(\.\d+)?")


def _looks_numeric(text: str | None) -> bool:
    return bool(text) and _NUMERIC.match(text.replace(",", "")) is not None


def _item(
    compound: Compound | None,
    priority: Priority,
    trigger: str,
    reason: str,
    *,
    provenance: Provenance | None = None,
    compound_id: str | None = None,
    measurement_id: str | None = None,
) -> ReviewItem:
    prov = provenance
    if prov is None and compound is not None and compound.provenance:
        prov = compound.provenance[0]
    return ReviewItem(
        compound_id=compound_id or (compound.compound_id if compound else DOCUMENT_SCOPE),
        priority=priority,
        trigger=trigger,
        reason=reason,
        page_no=prov.page_no if prov else None,
        bbox=prov.bbox if prov else None,
        crop_path=prov.crop_path if prov else None,
        smiles_from_name=compound.smiles_from_name if compound else None,
        smiles_from_image=compound.smiles_from_image if compound else None,
        inchikey_from_name=compound.inchikey_from_name if compound else None,
        inchikey_from_image=compound.inchikey_from_image if compound else None,
        crosscheck_tier=compound.crosscheck_tier if compound else None,
        measurement_id=measurement_id,
    )


def _compound_items(
    compound: Compound,
    *,
    previous_number: int | None,
    n_measurements: int,
    ocsr_conf_threshold: float,
    heavy_atom_threshold: int,
    expected_assays_per_compound: int | None,
) -> list[ReviewItem]:
    items: list[ReviewItem] = []

    if compound.opsin_status == "FAILURE":
        repair = compound.homoglyph_repair_applied
        suffix = f" after homoglyph repair ({repair})" if repair else " after homoglyph repair"
        items.append(
            _item(
                compound,
                "high",
                "opsin_parse_failure",
                f"OPSIN failed to parse the chemical name{suffix}.",
            )
        )

    if compound.crosscheck_tier == "CONFLICT":
        items.append(
            _item(
                compound,
                "high",
                "crosscheck_conflict",
                "Cross-check CONFLICT: the name and image channels give different "
                "skeletons. Do not auto-pick a winner (PRD §9).",
            )
        )

    unsanitizable = [
        channel
        for channel, smiles in (
            ("name", compound.smiles_from_name),
            ("image", compound.smiles_from_image),
            ("final", compound.smiles_final),
        )
        if not _sanitizes(smiles)
    ]
    if unsanitizable:
        items.append(
            _item(
                compound,
                "high",
                "smiles_sanitization_failure",
                f"SMILES from {' and '.join(unsanitizable)} fails RDKit sanitization.",
            )
        )

    if compound.compound_number is None:
        items.append(
            _item(
                compound,
                "high",
                "compound_number_unreadable",
                "Compound number could not be read.",
            )
        )
    elif previous_number is not None and compound.compound_number < previous_number:
        items.append(
            _item(
                compound,
                "high",
                "compound_number_monotonicity",
                f"Compound number {compound.compound_number} goes backwards after "
                f"{previous_number}.",
            )
        )

    if (
        expected_assays_per_compound is not None
        and n_measurements < expected_assays_per_compound
    ):
        items.append(
            _item(
                compound,
                "high",
                "activity_cell_blank",
                f"Only {n_measurements} of {expected_assays_per_compound} expected assay "
                "values were extracted for this compound.",
            )
        )

    if compound.crosscheck_tier == "AGREE_SKELETON":
        items.append(
            _item(
                compound,
                "medium",
                "crosscheck_agree_skeleton",
                "Cross-check AGREE_SKELETON: same skeleton, stereochemistry disagrees "
                "between channels.",
            )
        )

    # PRD R13.1 — gate on the MINIMUM atom/bond confidence, never the molecule
    # mean: one wrong atom ruins the structure while barely moving the average.
    worst = [
        (name, value)
        for name, value in (
            ("atom", compound.ocsr_confidence_min_atom),
            ("bond", compound.ocsr_confidence_min_bond),
        )
        if value is not None and value < ocsr_conf_threshold
    ]
    if worst:
        detail = ", ".join(f"min {name} {value:.2f}" for name, value in worst)
        items.append(
            _item(
                compound,
                "medium",
                "ocsr_low_confidence",
                f"MolScribe confidence below τ={ocsr_conf_threshold:.2f} ({detail}).",
            )
        )

    if compound.heavy_atoms is not None and compound.heavy_atoms > heavy_atom_threshold:
        items.append(
            _item(
                compound,
                "medium",
                "heavy_atom_count",
                f"{compound.heavy_atoms} heavy atoms (> {heavy_atom_threshold}); OCSR "
                "accuracy degrades on large structures.",
            )
        )

    if compound.markush_detected:
        items.append(
            _item(
                compound,
                "info",
                "markush_structure",
                "Classified as a Markush structure — informational, never auto-resolved.",
            )
        )

    return items


def _measurement_items(m: Measurement, compound: Compound | None) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    has_value = _looks_numeric(m.published_value) or m.standard_value is not None
    has_bin = bool(m.bin_label_raw)

    if not m.published_value.strip() and not has_bin and m.standard_value is None:
        items.append(
            _item(
                compound,
                "high",
                "activity_cell_blank",
                f"Activity cell for {m.assay_name_raw} is blank where the table implies "
                "a value.",
                provenance=m.provenance,
                compound_id=m.compound_id,
                measurement_id=m.measurement_id,
            )
        )

    bin_resolved = (
        m.bin_definition is not None
        or m.bin_lower_nM is not None
        or m.bin_upper_nM is not None
        or m.bin_score is not None
    )
    if has_bin and not bin_resolved:
        items.append(
            _item(
                compound,
                "high",
                "bin_letter_outside_legend",
                f"Bin letter {m.bin_label_raw!r} is outside the resolved legend for "
                f"{m.assay_name_raw}.",
                provenance=m.provenance,
                compound_id=m.compound_id,
                measurement_id=m.measurement_id,
            )
        )

    if has_value and not has_bin and not (m.standard_units or m.published_units):
        items.append(
            _item(
                compound,
                "high",
                "units_unrecognized",
                f"Units for {m.assay_name_raw} are implicit or unrecognized.",
                provenance=m.provenance,
                compound_id=m.compound_id,
                measurement_id=m.measurement_id,
            )
        )

    return items


def build_queue(
    compounds: Sequence[Compound],
    measurements: Sequence[Measurement],
    anomalies: Sequence[DocumentAnomaly] = (),
    *,
    ocsr_conf_threshold: float | None = None,
    heavy_atom_threshold: int | None = None,
    expected_assays_per_compound: int | None = None,
) -> list[ReviewItem]:
    """Every PRD §13.2 trigger, in discovery order. Call `sort_queue` to prioritize."""
    config = get_config()
    # PRD R13.2 — TODO: calibrate τ against the gold set (Plan Part 13); the
    # config default is a placeholder, not a measured value.
    tau = config.ocsr_conf_threshold if ocsr_conf_threshold is None else ocsr_conf_threshold
    heavy_max = (
        config.heavy_atom_review_threshold
        if heavy_atom_threshold is None
        else heavy_atom_threshold
    )

    by_compound: dict[str, list[Measurement]] = defaultdict(list)
    for m in measurements:
        by_compound[m.compound_id].append(m)
    compounds_by_id = {c.compound_id: c for c in compounds}

    items: list[ReviewItem] = []
    previous_number: int | None = None
    for compound in compounds:
        items.extend(
            _compound_items(
                compound,
                previous_number=previous_number,
                n_measurements=len(by_compound.get(compound.compound_id, ())),
                ocsr_conf_threshold=tau,
                heavy_atom_threshold=heavy_max,
                expected_assays_per_compound=expected_assays_per_compound,
            )
        )
        if compound.compound_number is not None:
            previous_number = compound.compound_number

    for m in measurements:
        items.extend(_measurement_items(m, compounds_by_id.get(m.compound_id)))

    for anomaly in anomalies:
        mapped = _ANOMALY_TRIGGERS.get(anomaly.kind)
        if mapped is None:
            continue
        priority, headline = mapped
        items.append(
            _item(
                None,
                priority,
                anomaly.kind,
                f"{headline}: {anomaly.message}",
                provenance=anomaly.provenance,
            )
        )

    return items


def sort_queue(items: Sequence[ReviewItem]) -> list[ReviewItem]:
    """High, then medium, then info; stable within a priority (PRD §13.2)."""
    return sorted(items, key=lambda item: _PRIORITY_ORDER.get(item.priority, len(_PRIORITY_ORDER)))
