"""Letter-bin legend recovery from patent prose.

Implements PRD R10.6 (definitional sentence wins over a contradicting summary
sentence, with an anomaly emitted), R10.7 (the decoded intervals the reference
patent must yield), R10.12 (unit homoglyph repair before parsing), EC-5 (the
three real contradictions) and EC-6 (the Table 1 / Table 2 cross-reference
error).

A legend in this document is stated twice: a DEFINITIONAL sentence
("...having WIZ EC50 > 0.1 pM are level F") and a SUMMARY sentence
("...WIZ EC50 values ... and < .01 pM (activity level F) are shown in Table 2").
They disagree three times. The definitional sentence is authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from sarmine.artifacts.schema import BinDefinition, DocumentAnomaly
from sarmine.assay.normalize import normalize_units, to_nM

__all__ = [
    "CONTRADICTION_NOTE",
    "bin_is_contradicted",
    "decode_bin",
    "detect_cross_reference_error",
    "parse_legends",
]

CONTRADICTION_NOTE = "CONTRADICTED BY SUMMARY SENTENCE"

_DEFINITIONAL = re.compile(
    r"having\s+(?P<body>[\s\S]{1,120}?)\s*\b(?:are|is)\b\s+level\s+(?P<label>[A-Za-z])\b",
    re.IGNORECASE,
)
_SUMMARY_LABEL = re.compile(
    r"\(\s*activity\s+level\s+(?P<label>[A-Za-z])\s*\)", re.IGNORECASE
)
_BODY_NOISE = re.compile(r"\blevel\b|\bformula\b|\bcompounds\b", re.IGNORECASE)
_VALUES_ANCHOR = re.compile(r"\bvalues?\b", re.IGNORECASE)
_ENDPOINT_TOKEN = re.compile(r"\b[A-Za-z]{1,6}\d{1,2}\b")
_PARAGRAPH_SPLIT = re.compile(r"(?=\[\d{4,6}\])")
_PARAGRAPH_ID = re.compile(r"\[(\d{4,6})\]")
_TABLE_BELOW = re.compile(r"[Tt]able\s+(\d+)\s+below")

_RELATION = re.compile(
    r"between"
    r"|less\s+than\s+or\s+equal\s+to|greater\s+than\s+or\s+equal\s+to"
    r"|no\s+more\s+than|no\s+less\s+than"
    r"|less\s+than|greater\s+than"
    r"|<=|>=|\u2264|\u2265|<|>",
    re.IGNORECASE,
)
_RELATION_CANONICAL: dict[str, str] = {
    "between": "between",
    "less than": "<",
    "greater than": ">",
    "less than or equal to": "<=",
    "greater than or equal to": ">=",
    "no more than": "<=",
    "no less than": ">=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "\u2264": "<=",
    "\u2265": ">=",
}

_UNIT_TOKEN = r"%|uM|nM|mM|nm|M"
_NUMBER = r"\d+(?:\.\s?\d+)?|\.\s?\d+"
_NUMBER_WITH_UNIT = re.compile(rf"(?P<num>{_NUMBER})\s*(?P<unit>{_UNIT_TOKEN})?")

# Sentences are compared as scalars, so a legend bound is only "the same" within
# floating-point noise, not exactly.
_BOUND_TOLERANCE = 1e-9


@dataclass
class _Interval:
    lower: float | None
    upper: float | None
    units: str | None
    lower_inclusive: bool
    upper_inclusive: bool


@dataclass
class _Statement:
    label: str
    assay: str
    interval: _Interval
    sentence: str


def parse_legends(text: str) -> tuple[dict[str, list[BinDefinition]], list[DocumentAnomaly]]:
    """Decode every letter-bin legend, preferring definitional over summary text.

    Returns the bins keyed by the assay name as the prose states it, plus one
    `legend_contradiction` anomaly per bin whose summary restatement disagrees
    (PRD R10.6 / EC-5). Nothing is silently reconciled.
    """
    if not text or not text.strip():
        return {}, []

    statements = _definitional_statements(text)
    if not statements:
        return {}, []
    summaries = _summary_intervals(text)

    legends: dict[str, list[BinDefinition]] = {}
    anomalies: list[DocumentAnomaly] = []
    for statement in statements:
        units = statement.interval.units or _established_units(legends, statement.assay)
        if units is None:
            continue  # PRD R10.11 — a bound with no units is refused, never guessed
        summary = summaries.get(statement.label)
        discrepancy = (
            _discrepancy(statement.interval, summary, units) if summary is not None else None
        )
        definition_text = _collapse(statement.sentence)
        if discrepancy is not None:
            summary_desc = _describe(summary)  # type: ignore[arg-type]
            definition_text = f"{definition_text} [{CONTRADICTION_NOTE}: {summary_desc}]"
            anomalies.append(
                DocumentAnomaly(
                    kind="legend_contradiction",
                    severity="warning",
                    message=(
                        f"{statement.assay} level {statement.label}: the definitional "
                        f"sentence gives {_describe(statement.interval)} but the summary "
                        f"sentence gives {summary_desc} ({discrepancy}). Using the "
                        f"definitional sentence and marking affected rows "
                        f"reduced-confidence (PRD R10.6)."
                    ),
                )
            )
        legends.setdefault(statement.assay, []).append(
            BinDefinition(
                label=statement.label,
                assay=statement.assay,
                lower=statement.interval.lower,
                upper=statement.interval.upper,
                units=units,
                lower_inclusive=statement.interval.lower_inclusive,
                upper_inclusive=statement.interval.upper_inclusive,
                definition_text=definition_text,
            )
        )
    for bins in legends.values():
        _assign_scores(bins)
    return legends, anomalies


def decode_bin(label: str, legends: dict[str, list[BinDefinition]]) -> BinDefinition | None:
    """Look a letter bin up across all assays; an ambiguous label is refused."""
    key = (label or "").strip().strip(".,;:()[]").upper()
    if not key:
        return None
    # Tesseract duplicates an isolated capital in both cases — every one of
    # Table 2's five `C` cells comes back as `Cc`. A repeat of the SAME letter
    # is that artefact; two different letters are a genuinely unreadable cell.
    if len(key) > 1 and len(set(key)) == 1:
        key = key[0]
    hits = [b for bins in legends.values() for b in bins if b.label.upper() == key]
    if len(hits) != 1:
        return None
    return hits[0]


def bin_is_contradicted(bin_definition: BinDefinition) -> bool:
    """Whether this bin's summary restatement disagreed (PRD R10.6)."""
    return CONTRADICTION_NOTE in bin_definition.definition_text


def detect_cross_reference_error(text: str) -> list[DocumentAnomaly]:
    """Flag a paragraph that points the same results at two different tables.

    PRD EC-6: paragraph [00508] says the assay results "can be viewed" in
    Table 1 while the same paragraph places them in Table 2. Non-blocking.
    """
    anomalies: list[DocumentAnomaly] = []
    if not text or not text.strip():
        return anomalies
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        referenced = list(dict.fromkeys(_TABLE_BELOW.findall(paragraph)))
        if len(referenced) < 2:
            continue
        para_id = _PARAGRAPH_ID.match(paragraph.strip())
        located = f"paragraph [{para_id.group(1)}]" if para_id else "a paragraph"
        tables = " and ".join(f"Table {number}" for number in referenced)
        anomalies.append(
            DocumentAnomaly(
                kind="cross_reference_error",
                severity="warning",
                message=(
                    f"In {located} the same results are referred to both {tables}; "
                    f"the reference is inconsistent and must not be followed literally "
                    f"(PRD EC-6)."
                ),
            )
        )
    return anomalies


# --- definitional and summary extraction ---------------------------------


def _definitional_statements(text: str) -> list[_Statement]:
    seen: set[tuple[str, str]] = set()
    statements: list[_Statement] = []
    for match in _DEFINITIONAL.finditer(text):
        body = match.group("body")
        if _BODY_NOISE.search(body):
            continue
        parsed = _split_body(body)
        if parsed is None:
            continue
        assay, interval = parsed
        label = match.group("label").upper()
        key = (_assay_key(assay), label)
        if key in seen:
            continue
        seen.add(key)
        statements.append(
            _Statement(
                label=label,
                assay=_merge_assay_name(statements, assay),
                interval=interval,
                sentence=match.group(0),
            )
        )
    return statements


def _summary_intervals(text: str) -> dict[str, _Interval]:
    intervals: dict[str, _Interval] = {}
    previous_end = 0
    for match in _SUMMARY_LABEL.finditer(text):
        segment = text[max(previous_end, match.start() - 200) : match.start()]
        previous_end = match.end()
        anchor = None
        for anchor in _VALUES_ANCHOR.finditer(segment):
            pass
        if anchor is not None:
            segment = segment[anchor.end() :]
        interval = _parse_interval(_interval_text(segment))
        label = match.group("label").upper()
        if interval is not None and label not in intervals:
            intervals[label] = interval
    return intervals


def _split_body(body: str) -> tuple[str, _Interval] | None:
    """Split "WIZ EC50 < .01 pM" into its assay name and its interval."""
    relation = _RELATION.search(body)
    if relation is None:
        return None
    assay = _collapse(body[: relation.start()]).strip(" %.,;:-\u2013")
    interval = _parse_interval(_repair_units(body[relation.start() :]))
    if not assay or interval is None:
        return None
    return assay, interval


def _interval_text(segment: str) -> str:
    """Trim a summary clause down to the part that states the interval."""
    repaired = _repair_units(segment)
    relation = _RELATION.search(repaired)
    if relation is not None:
        return repaired[relation.start() :]
    stripped = _ENDPOINT_TOKEN.sub(" ", repaired)
    digit = re.search(r"\d", stripped)
    return stripped[digit.start() :] if digit is not None else ""


def _parse_interval(text: str) -> _Interval | None:
    if not text.strip():
        return None
    relation_match = _RELATION.search(text)
    relation = (
        _RELATION_CANONICAL.get(_collapse(relation_match.group()).lower())
        if relation_match is not None
        else None
    )
    values = list(_numbers(text))
    units = next((unit for _, unit in values if unit is not None), None)
    if len(values) >= 2:
        return _Interval(
            lower=values[0][0],
            upper=values[1][0],
            units=units,
            lower_inclusive=relation != ">",
            upper_inclusive=True,
        )
    if len(values) == 1:
        value = values[0][0]
        if relation in {"<", "<="}:
            return _Interval(None, value, units, True, relation == "<=")
        if relation in {">", ">="}:
            return _Interval(value, None, units, relation == ">=", True)
    return None


def _numbers(text: str) -> Iterator[tuple[float, str | None]]:
    for match in _NUMBER_WITH_UNIT.finditer(text):
        # "0. 1 pM" — the source's OCR splits some decimals with a space.
        yield float(match.group("num").replace(" ", "")), normalize_units(match.group("unit"))


def _repair_units(text: str) -> str:
    """PRD R10.12 / EC-15 — repair `µ` homoglyphs before any unit is parsed."""
    repaired = text.replace("\u00b5", "u").replace("\u03bc", "u")
    return re.sub(r"\bpM\b", "uM", repaired)


# --- contradiction detection ---------------------------------------------


def _discrepancy(
    definitional: _Interval, summary: _Interval, units: str
) -> str | None:
    """Describe how a summary restatement disagrees, or `None` if it agrees."""
    notes: list[str] = []
    if summary.units is not None and summary.units != units:
        notes.append(f"units restated as {summary.units} rather than {units}")
    def_lower, def_upper = _comparable(definitional, units)
    sum_lower, sum_upper = _comparable(summary, summary.units or units)
    if (def_lower is None) != (sum_lower is None) or (def_upper is None) != (sum_upper is None):
        notes.append("the interval is bounded from the opposite side")
    else:
        if not _same(def_lower, sum_lower):
            notes.append("lower bound differs")
        if not _same(def_upper, sum_upper):
            notes.append("upper bound differs")
    return "; ".join(notes) or None


def _comparable(interval: _Interval, units: str) -> tuple[float | None, float | None]:
    """Bounds on one scale. A bound of 0 on a non-negative quantity is no bound."""
    lower = None if interval.lower in (None, 0.0) else _scaled(interval.lower, units)
    upper = None if interval.upper is None else _scaled(interval.upper, units)
    return lower, upper


def _scaled(value: float, units: str) -> float:
    try:
        return to_nM(value, units)
    except ValueError:
        return value  # a percentage or other non-molar quantity compares directly


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= _BOUND_TOLERANCE * max(1.0, abs(left), abs(right))


def _describe(interval: _Interval) -> str:
    units = interval.units or ""
    if interval.lower is not None and interval.upper is not None:
        prefix = "" if interval.lower_inclusive else "> "
        return f"{prefix}{_num(interval.lower)}-{_num(interval.upper)} {units}".strip()
    if interval.upper is not None:
        return f"{'<=' if interval.upper_inclusive else '<'} {_num(interval.upper)} {units}".strip()
    if interval.lower is not None:
        return f"{'>=' if interval.lower_inclusive else '>'} {_num(interval.lower)} {units}".strip()
    return "an unbounded interval"


def _num(value: float) -> str:
    return f"{value:g}"


# --- bookkeeping ---------------------------------------------------------


def _assign_scores(bins: list[BinDefinition]) -> None:
    """Score bins by desirability so `bin(WIZ) - bin(ZBTB7A)` works (PRD AC-6.3)."""
    is_percent = bins[0].units == "%"

    def desirability(binned: BinDefinition) -> float:
        if is_percent:  # more induction is better
            return -(binned.lower if binned.lower is not None else float("-inf"))
        # a lower concentration is more potent
        return binned.upper if binned.upper is not None else float("inf")

    ordered = sorted(bins, key=desirability)
    for index, binned in enumerate(ordered):
        binned.score = len(ordered) - index


def _established_units(legends: dict[str, list[BinDefinition]], assay: str) -> str | None:
    """Units already stated by a sibling bin of the same assay in the same legend."""
    for binned in legends.get(assay, ()):
        return binned.units
    return None


def _merge_assay_name(statements: list[_Statement], assay: str) -> str:
    """Keep one display name per assay even when the prose punctuates it differently."""
    key = _assay_key(assay)
    for statement in statements:
        if _assay_key(statement.assay) == key:
            return statement.assay
    return assay


def _assay_key(assay: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", assay.casefold())


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
