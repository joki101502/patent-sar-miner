"""Part 6.5 — linking structures on one page to table rows on another (PRD §11).

Implements R11.1–R11.6. The hard edge in the reference patent is
Examples ↔ compound table: the Examples carry a name, a structure and full
characterization but never state which compound number they are
(decisions.md F10), so the bridge is canonical structure identity — the
InChIKey — not an identifier.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sarmine.artifacts.schema import Compound, DocumentAnomaly, Provenance

# A stereo descriptor at the head of, or embedded in, an IUPAC name: "(S)-",
# "(3R)-", "(1S,3R)-", "(E)-". PRD R11.3 — the glutarimide C3 centre epimerizes
# in solution and is drawn flat about half the time, so names must compare
# stereo-blind or one compound splits into two.
_STEREO_DESCRIPTOR = re.compile(
    r"\(\s*(?:\d+[a-z]?[rs]|[rs]|[ez])(?:\s*,\s*(?:\d+[a-z]?[rs]|[rs]|[ez]))*\s*\)[-\s]*"
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]")
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"})

# Measured on the reference patent's own names: a stereo-only difference scores
# 1.00 and a different compound of the same chemotype scores 0.75.
NAME_SIMILARITY_THRESHOLD = 0.90

# PRD R11.4 / EC-23 — the marker carried by an Example that joined to nothing.
NO_ACTIVITY_DATA = "no activity data"

JOIN_COMPOUND_NUMBER = "compound_number"
JOIN_INCHIKEY_FULL = "inchikey_full"
JOIN_NAME_SIMILARITY = "name_similarity"
JOIN_INCHIKEY_SKELETON = "inchikey_skeleton"
JOIN_UNJOINED_EXAMPLE = "unjoined_example"

# PRD R11.2 — tried in this order; the first to land wins, the rest corroborate.
EXAMPLE_JOIN_METHODS = (JOIN_INCHIKEY_FULL, JOIN_NAME_SIMILARITY, JOIN_INCHIKEY_SKELETON)


@dataclass
class ActivityRow:
    """One row of the activity table (PRD §3.4), values verbatim per R10.1."""

    compound_number: int | None
    values: dict[str, str]
    page_no: int
    provenance: Provenance | None = None


@dataclass
class ExampleEntry:
    """One synthesis Example (PRD §3.7) — a name and a structure, but no number."""

    local_id: str
    name: str
    inchikey: str | None
    smiles: str | None
    page_no: int
    has_nmr: bool = False
    has_ms: bool = False


@dataclass
class JoinResult:
    compounds: list[Compound]
    anomalies: list[DocumentAnomaly]
    n_joined_activity: int
    n_joined_examples: int
    unjoined_examples: list[str] = field(default_factory=list)


def _anomaly(message: str) -> DocumentAnomaly:
    return DocumentAnomaly(kind="compound_number_gap", severity="warning", message=message)


def validate_monotonic(numbers: Sequence[int | None]) -> list[DocumentAnomaly]:
    """PRD R11.5 / EC-4 / AC-5.4 — flag unreadable, duplicated, out-of-order or
    missing compound numbers. Never repair them: an invented compound number
    silently corrupts the join.
    """
    anomalies: list[DocumentAnomaly] = []

    for position, number in enumerate(numbers, start=1):
        if number is None:
            anomalies.append(
                _anomaly(
                    f"compound number at position {position} is unreadable; "
                    "flagged, never interpolated"
                )
            )

    known = [n for n in numbers if n is not None]

    duplicates = sorted(n for n, count in Counter(known).items() if count > 1)
    for number in duplicates:
        anomalies.append(_anomaly(f"duplicate compound number {number} in the sequence"))

    previous: int | None = None
    for position, number in enumerate(known, start=1):
        if previous is not None and number < previous:
            anomalies.append(
                _anomaly(
                    f"compound number {number} at position {position} breaks monotonic "
                    f"order (previous was {previous}); flagged, never interpolated"
                )
            )
        previous = number

    if known:
        missing = sorted(set(range(min(known), max(known) + 1)) - set(known))
        if missing:
            anomalies.append(
                _anomaly(
                    "compound number sequence has gaps: "
                    f"{', '.join(str(n) for n in missing)}; flagged, never interpolated"
                )
            )

    return anomalies


def _normalize_name(name: str) -> str:
    lowered = name.casefold().translate(_DASHES)
    return _NON_ALPHANUMERIC.sub("", _STEREO_DESCRIPTOR.sub("", lowered))


def normalized_name_similarity(a: str, b: str) -> float:
    """PRD R11.2 — the name-string fallback when InChIKeys are unavailable."""
    left, right = _normalize_name(a), _normalize_name(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _dedupe(anomalies: Sequence[DocumentAnomaly]) -> list[DocumentAnomaly]:
    seen: set[tuple[str, str]] = set()
    unique: list[DocumentAnomaly] = []
    for anomaly in anomalies:
        key = (anomaly.kind, anomaly.message)
        if key not in seen:
            seen.add(key)
            unique.append(anomaly)
    return unique


def link_activity_rows(
    compounds: Sequence[Compound], rows: Sequence[ActivityRow]
) -> tuple[dict[int, ActivityRow], list[DocumentAnomaly]]:
    """PRD R11.1 — the primary join: compound-table row *n* ↔ activity-table row *n*.

    Returns `{compound_number: ActivityRow}` for the rows that matched, plus the
    anomalies raised by either number sequence. Mutates nothing.
    """
    anomalies = validate_monotonic([c.compound_number for c in compounds])
    anomalies += validate_monotonic([r.compound_number for r in rows])

    by_number: dict[int, ActivityRow] = {
        row.compound_number: row for row in rows if row.compound_number is not None
    }
    compound_numbers = {c.compound_number for c in compounds if c.compound_number is not None}

    matched = {n: row for n, row in by_number.items() if n in compound_numbers}

    for number in sorted(set(by_number) - compound_numbers):
        anomalies.append(
            DocumentAnomaly(
                kind="source_unavailable",
                severity="warning",
                message=(
                    f"activity-table row for compound number {number} matches no "
                    "compound-table row"
                ),
            )
        )
    for number in sorted(compound_numbers - set(by_number)):
        anomalies.append(
            DocumentAnomaly(
                kind="source_unavailable",
                severity="info",
                message=f"compound {number} has no activity-table row",
            )
        )

    return matched, _dedupe(anomalies)


@dataclass
class _ExampleMatch:
    compound_id: str
    method: str
    agreeing_methods: list[str]


def _candidate_keys(compound: Compound) -> set[str]:
    return {k for k in (compound.inchikey_from_name, compound.inchikey_full) if k}


def _candidate_skeletons(compound: Compound) -> set[str]:
    return {k[:14] for k in _candidate_keys(compound)} | (
        {compound.inchikey_skeleton} if compound.inchikey_skeleton else set()
    )


def _match_example(
    example: ExampleEntry,
    compounds: Sequence[Compound],
    names: Mapping[str, str],
    claimed: set[str],
) -> _ExampleMatch | None:
    """PRD R11.2 — InChIKey, then name similarity, then skeleton key."""
    available = [c for c in compounds if c.compound_id not in claimed]
    picks: dict[str, str] = {}

    if example.inchikey:
        for compound in available:
            if example.inchikey in _candidate_keys(compound):
                picks[JOIN_INCHIKEY_FULL] = compound.compound_id
                break

    best_score, best_id = 0.0, None
    for compound in available:
        name = names.get(compound.compound_id)
        if not name:
            continue
        score = normalized_name_similarity(example.name, name)
        if score > best_score:
            best_score, best_id = score, compound.compound_id
    if best_id is not None and best_score >= NAME_SIMILARITY_THRESHOLD:
        picks[JOIN_NAME_SIMILARITY] = best_id

    # PRD R11.3 — fall back to the 14-character skeleton so a stereocentre drawn
    # flat does not split one chemical compound into two.
    if example.inchikey:
        for compound in available:
            if example.inchikey[:14] in _candidate_skeletons(compound):
                picks[JOIN_INCHIKEY_SKELETON] = compound.compound_id
                break

    for method in EXAMPLE_JOIN_METHODS:
        if method in picks:
            winner = picks[method]
            agreeing = [m for m in EXAMPLE_JOIN_METHODS if picks.get(m) == winner]
            return _ExampleMatch(compound_id=winner, method=method, agreeing_methods=agreeing)
    return None


def _match_examples(
    compounds: Sequence[Compound],
    examples: Sequence[ExampleEntry],
    names: Mapping[str, str] | None,
) -> tuple[dict[str, _ExampleMatch], list[DocumentAnomaly]]:
    names = names or {}
    matches: dict[str, _ExampleMatch] = {}
    anomalies: list[DocumentAnomaly] = []
    claimed: set[str] = set()

    for example in examples:
        match = _match_example(example, compounds, names, claimed)
        if match is None:
            # PRD R11.4 / EC-23 — not an error; the compound was still made.
            anomalies.append(
                DocumentAnomaly(
                    kind="source_unavailable",
                    severity="info",
                    message=(
                        f"{example.local_id} joined to no compound-table row; "
                        f"carried through as a SAR row marked '{NO_ACTIVITY_DATA}'"
                    ),
                )
            )
            continue
        claimed.add(match.compound_id)
        matches[example.local_id] = match

    return matches, anomalies


def link_examples(
    compounds: Sequence[Compound],
    examples: Sequence[ExampleEntry],
    *,
    names: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[DocumentAnomaly]]:
    """Returns {example_local_id: compound_id} plus anomalies. Mutates nothing.

    `names` maps `compound_id` to the compound-table IUPAC name; without it the
    name-similarity fallback of R11.2 has nothing to compare against, and the
    join runs on InChIKeys alone.
    """
    matches, anomalies = _match_examples(compounds, examples, names)
    return {local_id: m.compound_id for local_id, m in matches.items()}, anomalies


def _activity_channels(compound: Compound) -> list[str]:
    """PRD R11.6 / AC-5.3 — which channels contributed, and whether they agreed."""
    channels = [JOIN_COMPOUND_NUMBER, "activity_number"]
    from_name, from_image = compound.inchikey_from_name, compound.inchikey_from_image
    if from_name:
        channels.append("name")
    if from_image:
        channels.append("image")
    if from_name and from_image:
        if from_name == from_image:
            channels.append("channels_agree")
        elif from_name[:14] == from_image[:14]:
            channels.append("channels_agree_skeleton")
        else:
            channels.append("channels_conflict")
    return channels


def _id_prefix(compounds: Sequence[Compound]) -> str:
    for compound in compounds:
        prefix, sep, _ = compound.compound_id.rpartition(":")
        if sep:
            return prefix
    return ""


def _example_row(example: ExampleEntry, prefix: str) -> Compound:
    """PRD R11.4 / EC-23 — an Example that joined to nothing is still a SAR row."""
    return Compound(
        compound_id=f"{prefix}:{example.local_id}" if prefix else example.local_id,
        compound_local_id=example.local_id,
        compound_number=None,
        smiles_from_name=example.smiles,
        smiles_final=example.smiles,
        structure_source="name" if example.smiles else "none",
        inchikey_full=example.inchikey,
        inchikey_from_name=example.inchikey,
        example_local_id=example.local_id,
        in_examples=True,
        join_method=JOIN_UNJOINED_EXAMPLE,
        join_channels=["example_name"],
        rank_rationale=[NO_ACTIVITY_DATA],
    )


def _flag_duplicate_structures(compounds: Sequence[Compound]) -> list[DocumentAnomaly]:
    """PRD EC-24 — merge-worthy duplicates are flagged, never silently dropped."""
    by_key: dict[str, list[Compound]] = defaultdict(list)
    for compound in compounds:
        if compound.inchikey_skeleton:
            by_key[compound.inchikey_skeleton].append(compound)

    anomalies: list[DocumentAnomaly] = []
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        for compound in group:
            compound.potential_duplicate = True
        anomalies.append(
            DocumentAnomaly(
                kind="duplicate_structure",
                severity="warning",
                message=(
                    f"skeleton key {key} is shared by "
                    f"{', '.join(c.compound_local_id for c in group)}; flagged, not merged"
                ),
            )
        )
    return anomalies


def join(
    compounds: Sequence[Compound],
    activity_rows: Sequence[ActivityRow],
    examples: Sequence[ExampleEntry],
    *,
    names: Mapping[str, str] | None = None,
) -> JoinResult:
    """PRD §11 — the whole join, returning fresh `Compound` rows.

    `join_method` is the "+"-joined list of the methods that produced this row's
    links, activity join first (PRD R11.6).
    """
    rows = [c.model_copy(deep=True) for c in compounds]

    matched, anomalies = link_activity_rows(rows, activity_rows)
    n_joined_activity = 0
    for row in rows:
        if row.compound_number is not None and row.compound_number in matched:
            n_joined_activity += 1
            row.join_method = JOIN_COMPOUND_NUMBER
            row.join_channels = _activity_channels(row)

    matches, example_anomalies = _match_examples(rows, examples, names)
    anomalies += example_anomalies

    by_id = {row.compound_id: row for row in rows}
    for local_id, match in matches.items():
        row = by_id[match.compound_id]
        row.example_local_id = local_id
        row.in_examples = True
        row.join_method = "+".join(m for m in (row.join_method, match.method) if m)
        channels = [*row.join_channels, "example_name", *match.agreeing_methods]
        if len(match.agreeing_methods) > 1:
            channels.append("methods_agree")
        row.join_channels = channels

    prefix = _id_prefix(rows)
    unjoined = [e.local_id for e in examples if e.local_id not in matches]
    rows += [_example_row(e, prefix) for e in examples if e.local_id not in matches]

    anomalies += _flag_duplicate_structures(rows)

    return JoinResult(
        compounds=rows,
        anomalies=_dedupe(anomalies),
        n_joined_activity=n_joined_activity,
        n_joined_examples=len(matches),
        unjoined_examples=unjoined,
    )
