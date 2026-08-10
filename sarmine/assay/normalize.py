"""Published/standardized value normalization for assay cells.

Implements PRD R10.1 (dual published/standardized columns), R10.2 (concentrations
to nM, log forms unwound), R10.3 (pChEMBL gating), R10.4 (pDC50 in its own
column), R10.5 (letter bins are interval-censored, never midpoint-imputed),
R10.11 (refuse implicit units), R10.12 (unit homoglyph repair) and R10.13
(`pint` for arithmetic, ontology identifiers for provenance only).
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pint

from sarmine.artifacts.schema import BinDefinition, CensorDirection, Measurement, Provenance

if TYPE_CHECKING:  # only for typing: `lexicon` imports this module at runtime
    from sarmine.assay.lexicon import HeaderMatch

__all__ = [
    "ParsedCell",
    "build_measurement",
    "normalize_units",
    "parse_cell",
    "pchembl_value",
    "pdc50_value",
    "to_nM",
]

# PRD R10.3 — ChEMBL's pChEMBL-eligible types, verbatim.
PCHEMBL_TYPES: frozenset[str] = frozenset(
    {"IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50"}
)

# ChEMBL emits a pChEMBL when the validity comment is null or manual validation.
_NON_BLOCKING_VALIDITY_COMMENTS: frozenset[str] = frozenset({"manually validated"})

# PRD R10.12 — MICRO SIGN (U+00B5) and GREEK SMALL LETTER MU (U+03BC) both
# stand in for `u`, and this source's OCR renders every `µM` as `pM`. The `pM`
# symbol is therefore reserved as the corruption, which is why v1 cannot
# represent genuine picomolar; such a column is refused rather than guessed.
_MICRO_CHARS = "\u00b5\u03bc"

_CONCENTRATION_UNITS: tuple[str, ...] = ("M", "mM", "uM", "nM")

# Symbol prefixes, applied only to a form ending in a capital `M`. Requiring the
# capital is what keeps `nm`/`um`/`mm` (metres) out of the molar space (R10.12).
_SI_PREFIXES: dict[str, str] = {
    "": "M",
    "m": "mM",
    "u": "uM",
    "p": "uM",  # PRD R10.12 — the OCR corruption of `µ`, not pico
    "n": "nM",
}

_UNIT_WORDS: dict[str, str] = {
    "molar": "M",
    "millimolar": "mM",
    "micromolar": "uM",
    "nanomolar": "nM",
    "percent": "%",
    "pct": "%",
}


def normalize_units(raw: str | None) -> str | None:
    """Canonicalize a unit string, or return `None` when it is unusable.

    Returning `None` is the refusal required by PRD R10.11 / EC-14: units are
    never inferred from the magnitude of the value.
    """
    if raw is None:
        return None
    text = raw.strip().strip("()[]{}").strip().rstrip(".,;:").strip()
    if not text:
        return None
    for char in _MICRO_CHARS:
        text = text.replace(char, "u")
    if text == "nm":
        return "nm"  # PRD R10.12 — nanometre, a different quantity entirely
    if text == "%":
        return "%"
    if text.endswith("M"):
        prefix = _SI_PREFIXES.get(text[:-1].lower())
        if prefix is not None:
            return prefix
        return None
    return _UNIT_WORDS.get(text.lower())


@functools.lru_cache(maxsize=1)
def _registry() -> pint.UnitRegistry:
    return pint.UnitRegistry()


def to_nM(value: float, units: str) -> float:
    """Convert `value` in `units` to nanomolar (PRD R10.2, arithmetic via pint)."""
    canonical = normalize_units(units)
    if canonical is None or canonical not in _CONCENTRATION_UNITS:
        raise ValueError(f"not a molar concentration unit: {units!r} (PRD R10.11)")
    ureg = _registry()
    return float(ureg.Quantity(value, canonical).to("nanomolar").magnitude)


@dataclass
class ParsedCell:
    """One activity-table cell, split into relation, value and residual text."""

    raw: str
    relation: str
    value: float | None
    text: str | None
    is_blank: bool


_BLANK_PLACEHOLDERS: frozenset[str] = frozenset({"", "-", "\u2013", "\u2014", "--"})
_RELATIONS: tuple[tuple[str, str], ...] = (
    (">=", ">="),
    ("<=", "<="),
    ("\u2265", ">="),
    ("\u2264", "<="),
    (">", ">"),
    ("<", "<"),
)
_LEADING_NUMBER = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_cell(raw: str) -> ParsedCell:
    """Split one activity cell into relation, numeric value and residual text.

    The raw string is preserved untouched: `published_value` is always the
    verbatim cell (PRD R10.1). A blank cell stays blank — it never becomes a
    zero or a low value (PRD EC-7).
    """
    raw = "" if raw is None else raw
    body = raw.strip()
    if body in _BLANK_PLACEHOLDERS:
        return ParsedCell(raw=raw, relation="=", value=None, text=None, is_blank=True)

    relation = "="
    for token, canonical in _RELATIONS:
        if body.startswith(token):
            relation = canonical
            body = body[len(token) :].strip()
            break

    number = _LEADING_NUMBER.match(body)
    if number is None:
        return ParsedCell(raw=raw, relation=relation, value=None, text=body, is_blank=False)
    value = float(number.group().replace(",", ""))
    return ParsedCell(raw=raw, relation=relation, value=value, text=None, is_blank=False)


def pchembl_value(
    standard_type: str,
    relation: str,
    value_nM: float | None,
    units: str | None,
    *,
    validity_comment: str | None = None,
) -> float | None:
    """`9 - log10(nM)`, but only when every PRD R10.3 condition holds.

    Returning `None` is the normal case for a third of real data: censored
    values, non-nM units and degrader endpoints all get nothing (PRD EC-17).
    """
    if standard_type not in PCHEMBL_TYPES:
        return None
    if relation != "=":
        return None
    if units != "nM":
        return None
    if value_nM is None or value_nM <= 0:
        return None
    if (
        validity_comment
        and validity_comment.strip().casefold() not in _NON_BLOCKING_VALIDITY_COMMENTS
    ):
        return None
    return 9.0 - math.log10(value_nM)


def pdc50_value(value_nM: float | None, relation: str) -> float | None:
    """`9 - log10(DC50_nM)`, kept in its own column and never pooled (PRD R10.4)."""
    if relation != "=":
        return None
    if value_nM is None or value_nM <= 0:
        return None
    return 9.0 - math.log10(value_nM)


# PRD R10.13 — Units Ontology identifiers, recorded for provenance only.
_UO_UNITS: dict[str, str] = {"nM": "UO_0000065", "%": "UO_0000187"}

_INVERTED_RELATIONS: dict[str, str] = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}
_CENSOR_DIRECTIONS: dict[str, CensorDirection] = {
    ">": "lower_bound",
    ">=": "lower_bound",
    "<": "upper_bound",
    "<=": "upper_bound",
}


def build_measurement(
    *,
    compound_id: str,
    header: HeaderMatch,
    raw_cell: str,
    provenance: Provenance,
    assay_group_key: str,
    legends: dict[str, list[BinDefinition]] | None = None,
    is_off_target: bool = False,
    cell_line: str | None = None,
    timepoint_h: float | None = None,
) -> Measurement | None:
    """Turn one activity cell into a `Measurement`, or refuse it.

    `None` means "this cell yields no trustworthy measurement" and the caller
    should queue it for review: a blank cell (EC-7), a numeric value whose units
    are absent or unusable (R10.11 / EC-14), an undecodable letter, or text that
    is neither a number nor a known bin.
    """
    # Imported here because `lexicon` imports this module at import time.
    from sarmine.assay.legend import bin_is_contradicted, decode_bin

    cell = parse_cell(raw_cell)
    if cell.is_blank:
        return None  # PRD EC-7 — a gap stays a gap

    fields: dict[str, Any] = {
        "measurement_id": f"{assay_group_key}::{compound_id}",
        "compound_id": compound_id,
        "assay_group_key": assay_group_key,
        "assay_name_raw": header.published_type,
        "target_raw": header.target,
        "is_off_target": is_off_target,
        "cell_line": cell_line,
        "timepoint_h": timepoint_h,
        "published_type": header.published_type,
        "published_value": cell.raw,
        "published_units": header.units,
        "standard_type": header.standard_type,
        "bao_endpoint": header.bao_endpoint,
        "provenance": provenance,
    }

    if cell.value is None:
        binned = decode_bin(cell.text or "", legends or {})
        if binned is None:
            return None
        fields.update(_bin_fields(binned))
        fields["published_text_value"] = cell.text
        # The verbatim string is preserved in `published_value` (R10.1); this
        # field carries the bin it resolved to, so a cell OCR'd `Cc` ranks and
        # exports as the `C` it is.
        fields["bin_label_raw"] = binned.label
        fields["reduced_confidence"] = bin_is_contradicted(binned)
        return Measurement(**fields)

    standardized = _standardize(header, cell)
    if standardized is None:
        return None  # PRD R10.11 — never infer units from magnitude
    value, units, relation = standardized
    fields.update(
        standard_value=value,
        standard_units=units,
        standard_relation=relation,
        is_censored=relation != "=",
        censor_direction=_CENSOR_DIRECTIONS.get(relation),
        uo_units=_UO_UNITS.get(units),
        pchembl_value=pchembl_value(header.standard_type, relation, value, units),
    )
    if header.standard_type == "DC50" and units == "nM":
        fields["pdc50_value"] = pdc50_value(value, relation)
        fields["pchembl_value"] = None  # PRD R10.4 — the two never pool
    return Measurement(**fields)


def _bin_fields(binned: BinDefinition) -> dict[str, Any]:
    """PRD R10.5 — store the decoded interval, and nothing in `standard_value`.

    `bin_lower_nM`/`bin_upper_nM` hold nanomolar bounds for a concentration bin
    and the assay's own units for anything else (HbF induction is a percentage),
    with `standard_units` recording which.
    """
    molar = binned.units in _CONCENTRATION_UNITS
    lower = _bound(binned.lower, binned.units, molar)
    upper = _bound(binned.upper, binned.units, molar)
    if upper is not None and lower is None:
        relation, direction = "<", "upper_bound"
    elif lower is not None and upper is None:
        relation, direction = ">", "lower_bound"
    else:
        relation, direction = "=", None
    return {
        # PRD R10.5 — a midpoint here would fabricate precision the patent
        # does not contain, so `standard_value` stays None for every bin.
        "standard_value": None,
        "standard_units": "nM" if molar else binned.units,
        "standard_relation": relation,
        "is_censored": True,
        "censor_direction": direction,
        "bin_definition": binned.definition_text,
        "bin_lower_nM": lower,
        "bin_upper_nM": upper,
        "bin_score": binned.score,
    }


def _bound(value: float | None, units: str, molar: bool) -> float | None:
    if value is None:
        return None
    return to_nM(value, units) if molar else value


def _standardize(header: HeaderMatch, cell: ParsedCell) -> tuple[float, str, str] | None:
    """Return the standardized value, its units and its relation (PRD R10.2)."""
    assert cell.value is not None
    if header.is_log_form:
        # A larger p-value is a smaller concentration, so the censoring flips.
        return 10.0 ** (9.0 - cell.value), "nM", _INVERTED_RELATIONS.get(cell.relation, "=")
    if header.units is None:
        return None
    if header.units == "%":
        return cell.value, "%", cell.relation
    try:
        return to_nM(cell.value, header.units), "nM", cell.relation
    except ValueError:
        return None
