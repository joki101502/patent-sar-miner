"""Read and write the versioned artifact bundle described in PRD §15.5.

    artifacts/{pubnum}/{run_id}/
      manifest.json  compounds.json  measurements.json  anomalies.json
      crops/  svg/  source/
"""

from __future__ import annotations

import json
from pathlib import Path

from sarmine.artifacts.schema import (
    Bundle,
    Compound,
    DocumentAnomaly,
    Measurement,
    RunManifest,
)

SUBDIRS = ("crops", "svg", "source")


def bundle_dir(out_root: str | Path, pubnum: str, run_id: str) -> Path:
    return Path(out_root) / pubnum / run_id


def ensure_bundle_dir(out_root: str | Path, pubnum: str, run_id: str) -> Path:
    root = bundle_dir(out_root, pubnum, run_id)
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_bundle(
    manifest: RunManifest,
    compounds: list[Compound],
    measurements: list[Measurement],
    anomalies: list[DocumentAnomaly],
    out_root: str | Path,
) -> Path:
    root = ensure_bundle_dir(out_root, manifest.pubnum, manifest.run_id)
    _dump(root / "manifest.json", manifest.model_dump(mode="json"))
    _dump(root / "compounds.json", [c.model_dump(mode="json") for c in compounds])
    _dump(root / "measurements.json", [m.model_dump(mode="json") for m in measurements])
    _dump(root / "anomalies.json", [a.model_dump(mode="json") for a in anomalies])
    return root


def read_bundle(path: str | Path) -> Bundle:
    root = Path(path)
    manifest = RunManifest.model_validate_json((root / "manifest.json").read_text("utf-8"))
    compounds = [
        Compound.model_validate(d)
        for d in json.loads((root / "compounds.json").read_text("utf-8"))
    ]
    measurements = [
        Measurement.model_validate(d)
        for d in json.loads((root / "measurements.json").read_text("utf-8"))
    ]
    anomalies = [
        DocumentAnomaly.model_validate(d)
        for d in json.loads((root / "anomalies.json").read_text("utf-8"))
    ]
    return Bundle(
        root=str(root),
        manifest=manifest,
        compounds=compounds,
        measurements=measurements,
        anomalies=anomalies,
    )


def latest_run(out_root: str | Path, pubnum: str) -> Path | None:
    """Most recent run directory for a publication number, if any."""
    base = Path(out_root) / pubnum
    if not base.is_dir():
        return None
    runs = sorted((p for p in base.iterdir() if (p / "manifest.json").is_file()))
    return runs[-1] if runs else None


def list_runs(out_root: str | Path) -> list[Path]:
    base = Path(out_root)
    if not base.is_dir():
        return []
    return sorted(
        p for p in base.glob("*/*") if (p / "manifest.json").is_file()
    )
