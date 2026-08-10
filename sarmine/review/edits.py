"""Reviewer corrections, the audit trail, and provenance completeness.

Implements PRD R13.4–R13.6 (the original extraction is always retained, with an
audit trail of what changed; corrections are session-scoped and exportable) and
AC-8.1 (every SMILES, activity value and compound number carries provenance).

The store is a plain in-memory object with a flat, JSON-serializable dump so a
database is a later drop-in (R13.6, Plan Part 11.3).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from rdkit import Chem

from sarmine.artifacts.schema import Compound, Measurement, Provenance

TargetKind = Literal["compound", "measurement"]

_CHANNEL_KEY_FIELD = {
    "smiles_from_name": "inchikey_from_name",
    "smiles_from_image": "inchikey_from_image",
}
_SMILES_FIELDS = frozenset({"smiles_final", *_CHANNEL_KEY_FIELD})


@dataclass
class AuditEntry:
    """One reviewer edit. `original` is the value immediately before this edit."""

    timestamp: str
    target_kind: TargetKind
    target_id: str
    field: str
    original: str | None
    corrected: str | None
    note: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _inchikey(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol) or None


def _smiles_side_effects(compound: Compound, field: str, value: str | None) -> dict[str, Any]:
    """Keep identity keys consistent with a corrected SMILES (PRD R9.16)."""
    updates: dict[str, Any] = {}

    key_field = _CHANNEL_KEY_FIELD.get(field)
    if key_field:
        updates[key_field] = _inchikey(value)

    previous = getattr(compound, field)
    final = compound.smiles_final
    if field == "smiles_final" or not final or final == previous:
        updates["smiles_final"] = value
        full = _inchikey(value)
        updates["inchikey_full"] = full
        # Cleared rather than left stale: an unparseable correction must re-enter
        # the review queue instead of joining on a key from the old structure.
        updates["inchikey_skeleton"] = full[:14] if full else None

    return updates


class CorrectionStore:
    """Session-scoped reviewer corrections with a full audit trail (PRD R13.4)."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._originals: dict[tuple[str, str, str], str | None] = {}

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CorrectionStore):
            return NotImplemented
        return self._entries == other._entries

    def _record(
        self,
        target_kind: TargetKind,
        target_id: str,
        field: str,
        original: Any,
        corrected: Any,
        note: str | None,
    ) -> None:
        entry = AuditEntry(
            timestamp=_now(),
            target_kind=target_kind,
            target_id=target_id,
            field=field,
            original=_as_text(original),
            corrected=_as_text(corrected),
            note=note,
        )
        self._entries.append(entry)
        self._originals.setdefault((target_kind, target_id, field), entry.original)

    def correct_compound(
        self, compound: Compound, field: str, value: Any, *, note: str | None = None
    ) -> Compound:
        if field not in Compound.model_fields:
            raise ValueError(f"{field!r} is not a Compound field")

        original = getattr(compound, field)
        updates: dict[str, Any] = {field: value}
        if field in _SMILES_FIELDS:
            updates.update(_smiles_side_effects(compound, field, value))

        self._record("compound", compound.compound_id, field, original, value, note)
        return Compound(**{**compound.model_dump(), **updates})

    def correct_measurement(
        self, m: Measurement, field: str, value: Any, *, note: str | None = None
    ) -> Measurement:
        if field not in Measurement.model_fields:
            raise ValueError(f"{field!r} is not a Measurement field")

        original = getattr(m, field)
        self._record("measurement", m.measurement_id, field, original, value, note)
        return Measurement(**{**m.model_dump(), field: value})

    def original(self, target_kind: str, target_id: str, field: str) -> str | None:
        """The value as first extracted, however many times it was since corrected."""
        return self._originals.get((target_kind, target_id, field))

    def to_rows(self) -> list[dict[str, Any]]:
        """Export rows for XLSX/CSV: original and corrected side by side (AC-7.5)."""
        return [
            {
                "timestamp": entry.timestamp,
                "target_kind": entry.target_kind,
                "target_id": entry.target_id,
                "field": entry.field,
                "original": self.original(entry.target_kind, entry.target_id, entry.field),
                "corrected": entry.corrected,
                "note": entry.note,
            }
            for entry in self._entries
        ]

    def model_dump(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._entries]

    @classmethod
    def from_dump(cls, data: Iterable[dict[str, Any] | AuditEntry]) -> CorrectionStore:
        store = cls()
        for item in data:
            entry = item if isinstance(item, AuditEntry) else AuditEntry(**item)
            store._entries.append(entry)
            store._originals.setdefault(
                (entry.target_kind, entry.target_id, entry.field), entry.original
            )
        return store


def _provenance_gap(prov: Provenance | None) -> str | None:
    if prov is None:
        return "no provenance"
    missing: list[str] = []
    if prov.page_no < 1:
        missing.append("page_no")  # AC-8.3 — pages are 1-indexed
    x0, y0, x1, y1 = prov.bbox
    if x1 <= x0 or y1 <= y0:
        missing.append("bbox")
    if not prov.crop_path.strip():
        missing.append("crop_path")
    return "missing " + ", ".join(missing) if missing else None


def missing_provenance(
    compounds: Sequence[Compound], measurements: Sequence[Measurement]
) -> list[str]:
    """AC-8.1 — describe every extracted value that cannot be traced to the page."""
    problems: list[str] = []

    for compound in compounds:
        prov = compound.provenance[0] if compound.provenance else None
        gap = _provenance_gap(prov)
        if gap is None:
            continue
        has_structure = any(
            (compound.smiles_from_name, compound.smiles_from_image, compound.smiles_final)
        )
        if has_structure:
            problems.append(f"Compound {compound.compound_id}: SMILES has {gap}.")
        if compound.compound_number is not None:
            problems.append(f"Compound {compound.compound_id}: compound number has {gap}.")

    for m in measurements:
        gap = _provenance_gap(m.provenance)
        if gap is not None:
            problems.append(
                f"Measurement {m.measurement_id} ({m.assay_name_raw}): activity value has {gap}."
            )

    return problems
