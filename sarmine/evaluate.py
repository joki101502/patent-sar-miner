"""Score an artifact bundle against a committed gold set (Plan Part 13.3, PRD §20).

The point of this module is to turn accuracy from a vibe into a number, and to
report the number that is actually hard rather than the flattering one.

**The headline is end-to-end triplet accuracy** — compound + assay + value all
correct together (PRD AC-10.2). Per-component reporting is misleading: BioMiner
measures F1 = 0.32 on full bioactivity triplets while its component tasks look
far better, and BioChemInsight's ">90%" is a component-task number. Structure,
activity-cell and join accuracy are all reported too (AC-10.1), but underneath.

This module also reports how each half of the gold set was VERIFIED. The activity
half is hand-read ground truth; the structure half is thin and partly derived by
OPSIN over the description text. Presenting the second as if it carried the
authority of the first would be exactly the kind of false confidence the whole
review-queue design exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sarmine.artifacts.writer import read_bundle


def _load_gold(gold_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(gold_path).read_text("utf-8"))


def _assay_identity(name: str, target: str | None = None) -> str:
    """Collapse an assay column to a comparable identity.

    `Measurement.assay_name_raw` is the VERBATIM header and must stay verbatim
    (PRD R10.1), but the header is OCR output: on the reference patent the real
    `WIZ EC50 (uM)` reads as `WIZ ECS50 (uM)`. Scoring on the raw string would
    charge the extractor for a header typo it was right to preserve, so gold and
    extraction are matched on the resolved target instead.
    """
    if target:
        return target.strip().upper()
    text = name.upper()
    for known in ("ZBTB7A", "WIZ", "HBF", "ARNT"):
        if known in text.replace(" ", ""):
            return known
    return "".join(ch for ch in text if ch.isalnum())


def _predicted_activity(bundle) -> dict[str, dict[str, str]]:
    """compound_local_id -> {assay identity: verbatim letter/value}."""
    by_id = {c.compound_id: c.compound_local_id for c in bundle.compounds}
    out: dict[str, dict[str, str]] = {}
    for m in bundle.measurements:
        local = by_id.get(m.compound_id)
        if local is None:
            continue
        value = m.bin_label_raw or m.published_value
        if value in (None, ""):
            continue
        out.setdefault(str(local), {})[_assay_identity(m.assay_name_raw, m.target_raw)] = str(
            value
        )
    return out


def _gold_activity(gold: dict) -> dict[str, dict[str, str | None]]:
    """Gold activity re-keyed to the same assay identity as the extraction."""
    targets = {a["name"]: a.get("target") for a in gold.get("assays", [])}
    return {
        num: {_assay_identity(assay, targets.get(assay)): value for assay, value in row.items()}
        for num, row in gold.get("activity", {}).items()
    }


def _score_activity(gold: dict, predicted: dict) -> dict[str, Any]:
    """Blank gold cells are NOT scorable values (PRD EC-7): a blank means the
    patent did not report a number. Emitting one there is a SPURIOUS value, not a
    near-miss, so it is counted separately and penalised."""
    correct = wrong = missing = spurious = 0
    gold_cells = 0
    errors: list[str] = []

    for num, row in gold.items():
        pred_row = predicted.get(str(num), {})
        for assay, expected in row.items():
            got = pred_row.get(assay)
            if expected is None:
                if got not in (None, ""):
                    spurious += 1
                    errors.append(f"compound {num} / {assay}: gold blank, extracted {got!r}")
                continue
            gold_cells += 1
            if got is None or got == "":
                missing += 1
                errors.append(f"compound {num} / {assay}: expected {expected!r}, missing")
            elif got == expected:
                correct += 1
            else:
                wrong += 1
                errors.append(f"compound {num} / {assay}: expected {expected!r}, got {got!r}")

    denominator = gold_cells + spurious
    return {
        "total": sum(len(r) for r in gold.values()),
        "scorable": gold_cells,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "spurious": spurious,
        "accuracy": (correct / denominator) if denominator else 0.0,
        "errors": errors[:50],
    }


def _score_structures(gold: dict, bundle) -> dict[str, Any]:
    """Exact InChIKey match against gold. Skeleton-only agreement is reported
    separately because stereo-only disagreement is a materially different failure
    from a wrong skeleton — this chemotype's glutarimide C3 centre epimerizes in
    solution and patents draw it flat about half the time (PRD R9.22 / EC-19)."""
    gold_structures = gold.get("compounds", {})

    # Over-segmentation can emit two rows with the same local id — the real
    # compound plus an empty band. Index on the patent's own compound NUMBER
    # first (it is the join key), and break remaining ties toward the row that
    # actually carries a structure, so scoring is not decided by dict ordering.
    by_local: dict[str, Any] = {}
    for c in bundle.compounds:
        for key in filter(None, (str(c.compound_number) if c.compound_number else None,
                                 str(c.compound_local_id))):
            existing = by_local.get(key)
            if existing is None or (existing.inchikey_full is None and c.inchikey_full):
                by_local[key] = c

    scored = correct = skeleton_correct = 0
    errors: list[str] = []
    for num, entry in gold_structures.items():
        expected = entry.get("inchikey")
        if not expected:
            continue
        scored += 1
        compound = by_local.get(str(num))
        got = compound.inchikey_full if compound else None
        if got == expected:
            correct += 1
        else:
            if got and got[:14] == expected[:14]:
                skeleton_correct += 1
            errors.append(f"compound {num}: expected {expected}, got {got}")

    verification = sorted({e.get("verification", "unspecified") for e in gold_structures.values()})
    return {
        "scored": scored,
        "correct": correct,
        "skeleton_correct": skeleton_correct,
        "accuracy": (correct / scored) if scored else 0.0,
        "skeleton_accuracy": ((correct + skeleton_correct) / scored) if scored else 0.0,
        "gold_verification": verification,
        "gold_note": (
            "Structure gold is thin: only entries marked hand_checked were verified by a "
            "human. Treat this accuracy as indicative, not as a benchmark result."
        ),
        "errors": errors[:50],
    }


def _score_examples(gold: dict, bundle) -> dict[str, Any]:
    """Did we recover the Example-section structures? Their names are machine
    readable, so this isolates the name channel from the OCR channel."""
    gold_examples = gold.get("examples", {})
    if not gold_examples:
        return {"scored": 0, "correct": 0, "accuracy": 0.0}

    found = {c.inchikey_full for c in bundle.compounds if c.inchikey_full}
    found_skeletons = {k[:14] for k in found}
    correct = sum(1 for e in gold_examples.values() if e.get("inchikey") in found)
    skeleton = sum(
        1 for e in gold_examples.values() if e.get("inchikey", "")[:14] in found_skeletons
    )
    return {
        "scored": len(gold_examples),
        "correct": correct,
        "skeleton_correct": skeleton,
        "accuracy": correct / len(gold_examples),
    }


def _score_join(gold: dict, bundle, predicted: dict) -> dict[str, Any]:
    """A compound counts as joined when it carries at least one activity value —
    i.e. the compound-table row and the activity-table row were actually linked."""
    gold_activity = gold.get("activity", {})
    expected_ids = {
        num for num, row in gold_activity.items() if any(v is not None for v in row.values())
    }
    matched = {num for num in expected_ids if predicted.get(str(num))}
    interpolated = [
        c.compound_local_id
        for c in bundle.compounds
        if getattr(c, "join_method", None) in {"interpolated", "inferred_number"}
    ]
    return {
        "gold_compounds": len(expected_ids),
        "compounds_matched": len(matched),
        "accuracy": (len(matched) / len(expected_ids)) if expected_ids else 0.0,
        "unmatched": sorted(expected_ids - matched, key=lambda s: int(s))[:50],
        # PRD R11.5 / AC-5.4 — an invented compound number silently corrupts the
        # join, so its presence is a hard failure, not a scoring detail.
        "interpolated_numbers": interpolated,
    }


def _score_triplets(gold: dict, predicted: dict) -> dict[str, Any]:
    """PRD AC-10.2 — compound + assay + value, all correct together."""
    gold_total = 0
    correct = 0
    predicted_total = sum(len(r) for r in predicted.values())
    for num, row in gold.items():
        for assay, expected in row.items():
            if expected is None:
                continue
            gold_total += 1
            if predicted.get(str(num), {}).get(assay) == expected:
                correct += 1

    precision = (correct / predicted_total) if predicted_total else 0.0
    recall = (correct / gold_total) if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "gold_total": gold_total,
        "predicted_total": predicted_total,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": recall if predicted_total <= gold_total else f1,
    }


def _score_legends(gold: dict, bundle) -> dict[str, Any]:
    """AC-4.1/AC-4.2 — were the three legends recovered, and were all three
    self-contradictions caught?"""
    expected_legends = gold.get("legends", {})
    # Legends are keyed by assay column, and the extracted column name is OCR
    # output, so match on the resolved identity as elsewhere.
    got: dict[str, set[str]] = {}
    for assay, bins in (bundle.manifest.legends or {}).items():
        got.setdefault(_assay_identity(assay), set()).update(b.label for b in bins)

    recovered = 0
    for assay, bins in expected_legends.items():
        got_bins = got.get(_assay_identity(assay), set())
        if {b["label"] for b in bins} <= got_bins:
            recovered += 1
    contradictions = sum(
        1 for a in bundle.anomalies if a.kind == "legend_contradiction"
    )
    return {
        "expected_legends": len(expected_legends),
        "recovered": recovered,
        "expected_contradictions": len(gold.get("legend_contradictions", [])),
        "detected_contradictions": contradictions,
    }


def evaluate_run(run_dir: str | Path, gold_path: str | Path) -> dict[str, Any]:
    bundle = read_bundle(run_dir)
    gold = _load_gold(gold_path)
    gold_activity = _gold_activity(gold)
    predicted = _predicted_activity(bundle)

    return {
        "pubnum": gold.get("pubnum"),
        "run_dir": str(run_dir),
        "source_mode": bundle.manifest.source_mode,
        "triplet": _score_triplets(gold_activity, predicted),
        "activity_cells": _score_activity(gold_activity, predicted),
        "structures": _score_structures(gold, bundle),
        "examples": _score_examples(gold, bundle),
        "join": _score_join(gold, bundle, predicted),
        "legends": _score_legends(gold, bundle),
        "counts": {
            "compounds_extracted": len(bundle.compounds),
            "measurements_extracted": len(bundle.measurements),
            "anomalies": len(bundle.anomalies),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    t = report["triplet"]
    a = report["activity_cells"]
    s = report["structures"]
    j = report["join"]
    lg = report["legends"]

    lines = [
        f"Patent SAR Miner — evaluation for {report.get('pubnum')} ({report.get('source_mode')})",
        "",
        "HEADLINE — end-to-end triplet accuracy (compound + assay + value all correct)",
        f"  triplet F1        : {t['f1']:.3f}   "
        f"(precision {t['precision']:.3f}, recall {t['recall']:.3f})",
        f"  triplets correct  : {t['correct']} / {t['gold_total']} gold, "
        f"{t['predicted_total']} extracted",
        "",
        "Component accuracies (informative only — see PRD AC-10.2)",
        f"  activity cells    : {a['accuracy']:.3f}   "
        f"({a['correct']} correct, {a['wrong']} wrong, {a['missing']} missing, "
        f"{a['spurious']} spurious of {a['scorable']} scorable)",
        f"  structures        : {s['accuracy']:.3f}   "
        f"({s['correct']}/{s['scored']} exact InChIKey, "
        f"{s['skeleton_correct']} skeleton-only)",
        f"  join              : {j['accuracy']:.3f}   "
        f"({j['compounds_matched']}/{j['gold_compounds']} compounds carry activity)",
        f"  legends           : {lg['recovered']}/{lg['expected_legends']} recovered, "
        f"{lg['detected_contradictions']}/{lg['expected_contradictions']} contradictions caught",
        "",
        f"Gold verification   : structures = {', '.join(s['gold_verification']) or 'none'}",
        f"  {s['gold_note']}",
    ]
    if j["interpolated_numbers"]:
        lines += ["", f"  FAIL: interpolated compound numbers present: {j['interpolated_numbers']}"]
    if a["errors"]:
        lines += ["", "First activity-cell errors:"] + [f"  - {e}" for e in a["errors"][:10]]
    return "\n".join(lines)


def calibrate_threshold(
    run_dir: str | Path, gold_path: str | Path, *, steps: int = 21
) -> dict[str, Any]:
    """Sweep the OCSR min-atom/min-bond confidence threshold against the gold set
    and pick the value maximising review-queue precision at acceptable recall
    (PRD R13.2 — tau must be calibrated, never guessed).

    A "positive" is a compound the queue would flag; it is a TRUE positive when
    that compound is genuinely wrong against gold. Gating uses the MINIMUM atom or
    bond confidence, never the molecule mean (PRD R13.1): one wrong atom ruins a
    structure while barely moving the average.
    """
    bundle = read_bundle(run_dir)
    gold = _load_gold(gold_path)
    gold_structures = {k: v.get("inchikey") for k, v in gold.get("compounds", {}).items()}

    scored: list[tuple[float, bool]] = []
    for c in bundle.compounds:
        expected = gold_structures.get(str(c.compound_local_id))
        if not expected:
            continue
        confidences = [
            v
            for v in (c.ocsr_confidence_min_atom, c.ocsr_confidence_min_bond)
            if v is not None
        ]
        if not confidences:
            continue
        scored.append((min(confidences), c.inchikey_full != expected))

    sweep: list[dict[str, float]] = []
    best = {"threshold": 0.85, "f1": -1.0}
    for i in range(steps):
        tau = i / (steps - 1)
        flagged = [(conf, wrong) for conf, wrong in scored if conf < tau]
        tp = sum(1 for _, wrong in flagged if wrong)
        total_wrong = sum(1 for _, wrong in scored if wrong)
        precision = (tp / len(flagged)) if flagged else 0.0
        recall = (tp / total_wrong) if total_wrong else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        sweep.append(
            {
                "threshold": round(tau, 3),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_flagged": len(flagged),
            }
        )
        if f1 > best["f1"]:
            best = {"threshold": round(tau, 3), "f1": f1}

    return {
        "chosen_threshold": best["threshold"],
        "best_f1": max(best["f1"], 0.0),
        "n_calibration_points": len(scored),
        "sweep": sweep,
        "note": (
            "Calibrated on compounds that have BOTH an OCSR confidence and a gold "
            "structure. With a thin structure gold this is weakly determined — say so "
            "in the UI rather than presenting it as a tuned threshold."
        )
        if len(scored) < 20
        else "",
    }
