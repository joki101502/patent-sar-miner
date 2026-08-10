"""Quality rules over standardized measurements.

Implements PRD R10.14 (the 3-or-6-order transcription-error detector borrowed
from ChEMBL curation), R10.15 (outside-typical-range), R10.16 (the 0.3-log
cross-assay noise floor) and R10.17 (`assay_group_key`, ChEMBL's TOID analogue).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations
from typing import Sequence

from sarmine.artifacts.schema import DocumentAnomaly, Measurement

__all__ = [
    "assay_group_key",
    "detect_transcription_errors",
    "is_meaningful_difference",
    "outside_typical_range",
]

# PRD R10.14 — a µM/nM or M/mM mixup lands on exactly one of these.
_UNIT_MIXUP_DECADES: tuple[int, ...] = (3, 6)
_DECADE_TOLERANCE = 1e-9

# PRD R10.15 — the curated per-type window, in nM.
_TYPICAL_RANGE_TYPES: frozenset[str] = frozenset({"IC50", "Ki", "EC50"})
_TYPICAL_MIN_nM = 0.01
_TYPICAL_MAX_nM = 100_000.0  # 100 µM

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def assay_group_key(pubnum: str, header: str) -> str:
    """Group the measurements of one assay column (PRD R10.17).

    The publication number is part of the key on purpose: two columns of the same
    patent table are the same assay, but the same target in a different patent is
    not, and comparing across patents without this distinction is the most common
    way patent SAR mining produces garbage.
    """
    slug = _SLUG_STRIP.sub("-", header.casefold()).strip("-")
    return f"{pubnum}::{slug}"


def detect_transcription_errors(measurements: Sequence[Measurement]) -> list[DocumentAnomaly]:
    """Flag otherwise-identical measurements differing by exactly 3 or 6 decades.

    PRD R10.14 / EC-16. High precision by construction: an order-of-magnitude
    difference that is not a clean power of a thousand is real chemistry, not a
    unit mixup, and is left alone.
    """
    grouped: dict[tuple[str, str, str, str | None], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.standard_value is None or measurement.standard_value <= 0:
            continue
        grouped[
            (
                measurement.compound_id,
                measurement.assay_group_key,
                measurement.standard_type,
                measurement.standard_units,
            )
        ].append(measurement)

    anomalies: list[DocumentAnomaly] = []
    for (compound_id, group, standard_type, units), members in grouped.items():
        for left, right in combinations(members, 2):
            decades = _decade_gap(left, right)
            if decades is None:
                continue
            anomalies.append(
                DocumentAnomaly(
                    kind="transcription_error",
                    severity="warning",
                    message=(
                        f"Compound {compound_id} has two {standard_type} values in "
                        f"assay {group} differing by exactly {decades} orders of "
                        f"magnitude ({left.published_value!r} vs "
                        f"{right.published_value!r}, standardized "
                        f"{left.standard_value} and {right.standard_value} {units}); "
                        f"this is the signature of a unit mixup (PRD R10.14)."
                    ),
                    provenance=left.provenance,
                )
            )
    return anomalies


def outside_typical_range(m: Measurement) -> bool:
    """Whether a potency sits outside the curated window (PRD R10.15)."""
    if m.standard_type not in _TYPICAL_RANGE_TYPES:
        return False
    if m.standard_units != "nM" or m.standard_value is None:
        return False
    return not _TYPICAL_MIN_nM <= m.standard_value <= _TYPICAL_MAX_nM


def is_meaningful_difference(
    a_nM: float, b_nM: float, *, floor_log_units: float = 0.3
) -> bool:
    """Whether two potencies differ by more than experimental noise (PRD R10.16).

    Same compound-target pairs measured in different assays differ by >0.3 log
    units 47% of the time, so a smaller gap must never be presented as real.
    """
    if a_nM <= 0 or b_nM <= 0:
        return False
    return abs(math.log10(a_nM) - math.log10(b_nM)) > floor_log_units


def _decade_gap(left: Measurement, right: Measurement) -> int | None:
    assert left.standard_value is not None and right.standard_value is not None
    gap = abs(math.log10(left.standard_value) - math.log10(right.standard_value))
    for decades in _UNIT_MIXUP_DECADES:
        if abs(gap - decades) <= _DECADE_TOLERANCE:
            return decades
    return None
