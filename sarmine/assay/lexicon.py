"""Assay-header recognition against a versioned data-file lexicon.

Implements PRD R10.8 (data-file lexicon, then high-threshold fuzzy match, then
review), R10.9 (the v1 endpoint set), R10.10 / EC-27 (headers split across
physical rows with hyphenated word-wrapping) and R10.13 (BAO identifiers carried
for provenance only).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from sarmine.assay.normalize import normalize_units
from sarmine.config import get_config

__all__ = [
    "HeaderMatch",
    "Lexicon",
    "load_lexicon",
    "reconstruct_header",
]

# An alias match is exact but less specific than a header seen verbatim in a real
# document, so it is reported just below full confidence.
_ALIAS_CONFIDENCE = 0.95

_TRAILING_PARENTHETICAL = re.compile(r"[\(\[]\s*([^()\[\]]{1,24}?)\s*[\)\]]\s*$")
_TRAILING_BARE_UNIT = re.compile(
    r"(?:(?<![A-Za-z0-9])(?P<dose>\d[\d.,]*)\s*)?"
    r"(?<![A-Za-z0-9])(?P<unit>[A-Za-z%\u00b5\u03bc]{1,12})\s*$"
)
_FIXED_DOSE = re.compile(r"@\s*[\d.,]+\s*[A-Za-z%\u00b5\u03bc]*")
_WORD_WRAP_HYPHEN = re.compile(r"-\s*$")


@dataclass
class HeaderMatch:
    """One recognized activity-table column header."""

    standard_type: str
    published_type: str
    units: str | None
    target: str | None
    bao_endpoint: str | None
    is_log_form: bool
    confidence: float
    matched_alias: str | None


@dataclass
class _Entry:
    """A lexicon record: either an endpoint alias set or a verbatim header."""

    standard_type: str
    aliases: tuple[str, ...] = ()
    header: str | None = None
    bao_endpoint: str | None = None
    is_log_form: bool = False
    units: str | None = None
    target: str | None = None


@dataclass
class Lexicon:
    version: str
    headers: list[_Entry] = field(default_factory=list)
    endpoints: list[_Entry] = field(default_factory=list)

    def match(self, header: str) -> HeaderMatch | None:
        """Recognize `header`, or return `None` so the caller queues it for review.

        PRD R10.8 fixes the order: verbatim header, then alias, then a
        high-threshold fuzzy pass, then refusal. Refusing is always preferable to
        a plausible-looking wrong endpoint.
        """
        if not header or not header.strip():
            return None
        cleaned = _collapse(header)
        core, units = self._split_units(cleaned)

        entry = self._exact_header(cleaned)
        if entry is not None:
            return self._build(header, entry, core, units, 1.0, entry.header)

        alias_hit = self._alias(core)
        if alias_hit is not None:
            entry, alias = alias_hit
            return self._build(header, entry, core, units, _ALIAS_CONFIDENCE, alias)

        fuzzy_hit = self._fuzzy(cleaned, core)
        if fuzzy_hit is not None:
            entry, alias, ratio = fuzzy_hit
            return self._build(header, entry, core, units, ratio, alias)
        return None

    # --- matching stages -------------------------------------------------

    def _exact_header(self, cleaned: str) -> _Entry | None:
        lowered = cleaned.casefold()
        for entry in self.headers:
            if entry.header and _collapse(entry.header).casefold() == lowered:
                return entry
        return None

    def _alias(self, core: str) -> tuple[_Entry, str] | None:
        for entry, alias in self._aliases_longest_first():
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", core, re.IGNORECASE
            ):
                return entry, alias
        return None

    def _fuzzy(self, cleaned: str, core: str) -> tuple[_Entry, str, float] | None:
        threshold = get_config().header_fuzzy_threshold
        best: tuple[_Entry, str, float] | None = None
        candidates: list[tuple[_Entry, str, str]] = [
            (entry, entry.header, cleaned) for entry in self.headers if entry.header
        ]
        candidates += [
            (entry, alias, core) for entry, alias in self._aliases_longest_first()
        ]
        for entry, candidate, subject in candidates:
            ratio = difflib.SequenceMatcher(
                None, subject.casefold(), _collapse(candidate).casefold()
            ).ratio()
            if ratio >= threshold and (best is None or ratio > best[2]):
                best = (entry, candidate, ratio)
        return best

    def _bao_for(self, standard_type: str) -> str | None:
        """A BAO id belongs to the endpoint type, so verbatim headers inherit it."""
        for entry in self.endpoints:
            if entry.standard_type == standard_type and entry.bao_endpoint:
                return entry.bao_endpoint
        return None

    def _aliases_longest_first(self) -> list[tuple[_Entry, str]]:
        pairs = [(entry, alias) for entry in self.endpoints for alias in entry.aliases]
        pairs.sort(key=lambda pair: len(pair[1]), reverse=True)
        return pairs

    # --- assembly --------------------------------------------------------

    def _split_units(self, cleaned: str) -> tuple[str, str | None]:
        """Return the header minus its units, and the units (PRD R10.11)."""
        parenthetical = _TRAILING_PARENTHETICAL.search(cleaned)
        if parenthetical is not None:
            units = normalize_units(parenthetical.group(1))
            if units is not None:
                return _collapse(cleaned[: parenthetical.start()]), units
            return cleaned, None
        bare = _TRAILING_BARE_UNIT.search(cleaned)
        # A trailing unit preceded by a number is a fixed dose ("@ 10 uM"), not
        # the units of the column's values.
        if bare is not None and bare.group("dose") is None:
            units = normalize_units(bare.group("unit"))
            if units is not None:
                return _collapse(cleaned[: bare.start("unit")]), units
        return cleaned, None

    def _build(
        self,
        published_type: str,
        entry: _Entry,
        core: str,
        units: str | None,
        confidence: float,
        matched_alias: str | None,
    ) -> HeaderMatch:
        target = entry.target or _residual_target(core, matched_alias)
        resolved_units = units or entry.units
        if entry.is_log_form:
            resolved_units = None  # a p-scale value is unitless (PRD R10.2)
        return HeaderMatch(
            standard_type=entry.standard_type,
            published_type=published_type,
            units=resolved_units,
            target=target,
            bao_endpoint=entry.bao_endpoint or self._bao_for(entry.standard_type),
            is_log_form=entry.is_log_form,
            confidence=confidence,
            matched_alias=matched_alias,
        )


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _residual_target(core: str, matched_alias: str | None) -> str | None:
    """Whatever qualifies the endpoint in the header is the target ("WIZ EC50")."""
    residual = core
    if matched_alias:
        residual = re.sub(re.escape(matched_alias), " ", residual, flags=re.IGNORECASE)
    residual = _FIXED_DOSE.sub(" ", residual)
    residual = _collapse(residual).strip(" -,;:()[]%.")
    return residual or None


def _entry_from_mapping(raw: Any) -> _Entry:
    data = dict(raw)
    aliases = tuple(str(a) for a in data.get("aliases", ()))
    return _Entry(
        standard_type=str(data["standard_type"]),
        aliases=aliases,
        header=str(data["header"]) if data.get("header") else None,
        bao_endpoint=str(data["bao_endpoint"]) if data.get("bao_endpoint") else None,
        is_log_form=bool(data.get("is_log_form", False)),
        units=str(data["units"]) if data.get("units") else None,
        target=str(data["target"]) if data.get("target") else None,
    )


_CACHE: dict[tuple[str, int], Lexicon] = {}


def load_lexicon(path: Path | None = None) -> Lexicon:
    """Load and cache the lexicon. Nothing is read at import time."""
    resolved = Path(path) if path is not None else get_config().assay_lexicon
    key = (str(resolved), resolved.stat().st_mtime_ns)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    raw = yaml.safe_load(resolved.read_text("utf-8")) or {}
    lexicon = Lexicon(
        version=str(raw.get("version", "")),
        headers=[_entry_from_mapping(item) for item in raw.get("headers", ())],
        endpoints=[_entry_from_mapping(item) for item in raw.get("endpoints", ())],
    )
    _CACHE[key] = lexicon
    return lexicon


def reconstruct_header(rows: Sequence[str]) -> str:
    """Join header fragments split across physical rows (PRD R10.10 / EC-27).

    A fragment ending in a hyphen is a word-wrap and joins with no separator;
    anything else joins with a single space.
    """
    parts: list[str] = []
    for row in rows:
        fragment = _collapse(row or "")
        if not fragment:
            continue
        if parts and _WORD_WRAP_HYPHEN.search(parts[-1]):
            parts[-1] = _WORD_WRAP_HYPHEN.sub("", parts[-1]) + fragment
        else:
            parts.append(fragment)
    return " ".join(parts)
