"""Wide SAR table and XLSX/CSV export (PRD G7, §15 preamble, R13.6, AC-7.5).

Storage is LONG — one row per measurement (PRD G5) — because that is the only
shape in which a censored bin, its verbatim letter and its provenance can all
live together. Display and export are WIDE, one row per compound, because that
is the shape a medicinal chemist reads.

Two things the pivot must not lose:

* The verbatim letter AND its decoded interval, side by side (PRD C1 / R10.5).
  Exporting only the interval fabricates precision; exporting only the letter
  makes the file unusable outside the patent's own legend.
* The difference between "not reported" and "reported as low" (PRD EC-7). A
  blank stays blank; it never becomes 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from sarmine.artifacts.schema import Compound, Measurement

# Per-compound columns, in the order a chemist reads them: identity first,
# then structure, then confidence, then the computed properties.
COMPOUND_COLUMNS: tuple[tuple[str, str], ...] = (
    ("compound_number", "compound_number"),
    ("compound_local_id", "compound_local_id"),
    ("rank", "rank"),
    ("rank_tie_group", "rank_tie_group"),
    ("potency_score", "potency_score"),
    ("selectivity_score", "selectivity_score"),
    ("smiles_final", "smiles_final"),
    ("inchikey_full", "inchikey_full"),
    ("inchikey_skeleton", "inchikey_skeleton"),
    ("structure_source", "structure_source"),
    ("crosscheck_tier", "crosscheck_tier"),
    ("opsin_status", "opsin_status"),
    ("ocsr_confidence_min_atom", "ocsr_confidence_min_atom"),
    ("ocsr_confidence_min_bond", "ocsr_confidence_min_bond"),
    ("markush_detected", "markush_detected"),
    ("has_undefined_stereocenters", "has_undefined_stereocenters"),
    ("standardization_skipped", "standardization_skipped"),
    ("mw", "mw"),
    ("clogp", "clogp"),
    ("tpsa", "tpsa"),
    ("qed", "qed"),
    ("hbd_lipinski", "hbd_lipinski"),
    ("hba_lipinski", "hba_lipinski"),
    ("rotb_strict", "rotb_strict"),
    ("heavy_atoms", "heavy_atoms"),
    ("fsp3", "fsp3"),
    ("n_aromatic_rings", "n_aromatic_rings"),
    ("in_examples", "in_examples"),
    ("in_claims", "in_claims"),
    ("has_in_vivo", "has_in_vivo"),
    ("join_method", "join_method"),
    ("rdkit_version", "rdkit_version"),
)


def _interval_text(m: Measurement) -> str:
    """Human-readable decoded interval, in nM, or the standardized value."""
    low, high = m.bin_lower_nM, m.bin_upper_nM
    if low is None and high is None:
        if m.standard_value is None:
            return ""
        relation = m.standard_relation if m.standard_relation != "=" else ""
        return f"{relation}{m.standard_value:g} {m.standard_units or ''}".strip()
    if low is None:
        return f"< {high:g}"
    if high is None:
        return f"> {low:g}"
    return f"{low:g}-{high:g}"


def _structure_provenance(compound: Compound) -> tuple[int | None, str]:
    for prov in compound.provenance:
        if "structure" in prov.crop_path or "name" in prov.crop_path:
            return prov.page_no, prov.crop_path
    if compound.provenance:
        first = compound.provenance[0]
        return first.page_no, first.crop_path
    return None, ""


def to_wide_frame(
    compounds: Sequence[Compound], measurements: Sequence[Measurement]
) -> pd.DataFrame:
    """One row per compound; one column per assay, plus a decoded-interval column."""
    by_compound: dict[str, list[Measurement]] = {}
    for m in measurements:
        by_compound.setdefault(m.compound_id, []).append(m)

    assays: list[str] = []
    for m in measurements:
        if m.assay_name_raw not in assays:
            assays.append(m.assay_name_raw)

    rows: list[dict[str, Any]] = []
    for compound in compounds:
        page_no, crop_path = _structure_provenance(compound)
        row: dict[str, Any] = {
            out: getattr(compound, attr, None) for out, attr in COMPOUND_COLUMNS
        }
        row["rank_rationale"] = "; ".join(compound.rank_rationale)
        row["investment_reasons"] = "; ".join(compound.investment_reasons)
        row["structure_page"] = page_no
        row["structure_crop"] = crop_path

        for assay in assays:
            row[assay] = ""
            row[f"{assay} (nM)"] = ""
        for m in by_compound.get(compound.compound_id, []):
            # PRD EC-7 — an unreported cell stays empty; it is not a low value.
            value = m.bin_label_raw if m.bin_label_raw is not None else m.published_value
            row[m.assay_name_raw] = value or ""
            row[f"{m.assay_name_raw} (nM)"] = _interval_text(m)
        rows.append(row)

    frame = pd.DataFrame(rows)
    ordered = [out for out, _ in COMPOUND_COLUMNS if out in frame.columns]
    tail = [c for c in frame.columns if c not in ordered]
    return frame[ordered + tail]


def to_long_frame(measurements: Sequence[Measurement]) -> pd.DataFrame:
    """The stored long form — published and standardized columns side by side
    (PRD R10.1). This is what a downstream consumer should read."""
    rows = []
    for m in measurements:
        record = m.model_dump(mode="json")
        prov = record.pop("provenance", {}) or {}
        record["page_no"] = prov.get("page_no")
        record["bbox"] = str(prov.get("bbox"))
        record["crop_path"] = prov.get("crop_path")
        record["extractor"] = prov.get("extractor")
        rows.append(record)
    return pd.DataFrame(rows)


def correction_rows(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Original and corrected side by side (PRD R13.4, AC-7.5)."""
    return [
        {
            "timestamp": e.get("timestamp"),
            "target_kind": e.get("target_kind"),
            "target_id": e.get("target_id"),
            "field": e.get("field"),
            "original": e.get("original"),
            "corrected": e.get("corrected"),
            "note": e.get("note"),
        }
        for e in entries
    ]


def to_csv(
    compounds: Sequence[Compound], measurements: Sequence[Measurement], path: str | Path
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_wide_frame(compounds, measurements).to_csv(out, index=False)
    return out


def to_xlsx(
    compounds: Sequence[Compound],
    measurements: Sequence[Measurement],
    path: str | Path,
    *,
    corrections: Sequence[dict[str, Any]] = (),
    anomalies: Sequence[Any] = (),
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        to_wide_frame(compounds, measurements).to_excel(
            writer, sheet_name="SAR table", index=False
        )
        to_long_frame(measurements).to_excel(writer, sheet_name="Measurements", index=False)
        pd.DataFrame(correction_rows(corrections) or [{}]).to_excel(
            writer, sheet_name="Corrections", index=False
        )
        if anomalies:
            pd.DataFrame(
                [a.model_dump(mode="json") if hasattr(a, "model_dump") else dict(a) for a in anomalies]
            ).to_excel(writer, sheet_name="Anomalies", index=False)
    return out
