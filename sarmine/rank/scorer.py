"""Part 9 — potency, selectivity, computed properties and the investment signal.

Implements PRD R12.1–R12.10. Two directions matter and are easy to invert:
ZBTB7A is the *off-target*, so a high bin score there is undesirable (R12.1),
and the target/off-target assignment is an input, not a constant (R12.5).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

from sarmine.artifacts.schema import Compound, Measurement

# PRD R12.1 — higher is more potent. HbF A/B/C, WIZ D/E/F, ZBTB7A G/H/I.
BIN_SCORES: dict[str, int] = {
    "A": 3, "B": 2, "C": 1,
    "D": 3, "E": 2, "F": 1,
    "G": 3, "H": 2, "I": 1,
}

# PRD R12.6 — both gotchas are recorded rather than left implicit: `CalcTPSA`
# defaults to excluding S and P, and Lipinski HBA (all N+O) is not `CalcNumHBA`.
TPSA_INCLUDE_S_AND_P = False
HBA_DEFINITION = "lipinski"

PROPERTY_FIELDS: tuple[str, ...] = (
    "mw",
    "clogp",
    "tpsa",
    "qed",
    "hbd_lipinski",
    "hba_lipinski",
    "rotb_strict",
    "heavy_atoms",
    "fsp3",
    "n_aromatic_rings",
)

def compute_properties(smiles: str | None) -> dict[str, float | int | None]:
    """PRD R12.6 — the descriptor set. Ro5 descriptors are meaningful for this
    chemotype: CRBN molecular glues are Ro5-compliant, unlike PROTACs.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return dict.fromkeys(PROPERTY_FIELDS, None)

    return {
        "mw": Descriptors.MolWt(mol),
        "clogp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol, includeSandP=TPSA_INCLUDE_S_AND_P),
        "qed": QED.qed(mol),
        "hbd_lipinski": rdMolDescriptors.CalcNumLipinskiHBD(mol),
        "hba_lipinski": rdMolDescriptors.CalcNumLipinskiHBA(mol),
        "rotb_strict": rdMolDescriptors.CalcNumRotatableBonds(
            mol, rdMolDescriptors.NumRotatableBondsOptions.Strict
        ),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def efficiency_metrics(
    pchembl: float | None,
    mw: float | None,
    tpsa: float | None,
    heavy_atoms: int | None,
    clogp: float | None,
) -> dict[str, float | None]:
    """PRD R12.8 — LE, BEI, SEI, LLE.

    All four require a pChEMBL. A censored value (every letter bin in the
    reference patent) has none, so these stay null rather than being imputed
    (EC-17); censored compounds are ranked in their own bucket instead.
    """
    metrics: dict[str, float | None] = {"LE": None, "BEI": None, "SEI": None, "LLE": None}
    if pchembl is None:
        return metrics

    if heavy_atoms:
        metrics["LE"] = 1.37 * pchembl / heavy_atoms
    if mw:
        metrics["BEI"] = pchembl * 1000.0 / mw
    if tpsa:
        metrics["SEI"] = pchembl * 100.0 / tpsa
    if clogp is not None:
        metrics["LLE"] = pchembl - clogp
    return metrics


# PRD R12.9 — the in vivo vocabulary. Reported term, then the pattern that finds
# it; inflections collapse onto the reported term.
IN_VIVO_TERMS: tuple[str, ...] = (
    "in vivo",
    "mouse",
    "murine",
    "rat",
    "xenograft",
    "pharmacokinetic",
    "PK",
    "oral bioavailability",
    "%F",
    "AUC",
    "Cmax",
)
_IN_VIVO_PATTERNS: dict[str, re.Pattern[str]] = {
    "in vivo": re.compile(r"\bin\s+vivo\b", re.IGNORECASE),
    "mouse": re.compile(r"\b(?:mouse|mice)\b", re.IGNORECASE),
    "murine": re.compile(r"\bmurine\b", re.IGNORECASE),
    "rat": re.compile(r"\brats?\b", re.IGNORECASE),
    "xenograft": re.compile(r"\bxenograft\w*\b", re.IGNORECASE),
    "pharmacokinetic": re.compile(r"\bpharmacokinetic\w*\b", re.IGNORECASE),
    "PK": re.compile(r"\bPK\b"),
    "oral bioavailability": re.compile(r"\boral\s+bioavailability\b", re.IGNORECASE),
    "%F": re.compile(r"%\s?F\b"),
    "AUC": re.compile(r"\bAUC\b"),
    "Cmax": re.compile(r"\bC\s?max\b", re.IGNORECASE),
}

# decisions.md F9 — the reference patent's only two hits are a PE-labelled Mouse
# Anti-Human antibody reagent in a flow-cytometry protocol and boilerplate about
# isotopic labels for in vivo imaging agents. Neither is in vivo efficacy data.
IN_VIVO_EXCLUSION_CONTEXT = re.compile(
    r"anti\s*-?\s*human|antibod|monoclonal|\bclone\b|isotype|conjugat|labell?ed\s+\w+\s+anti"
    r"|flow\s+cytometr|stain(?:ed|ing)|biosciences|biolegend|reagent"
    r"|imaging|isotop|radiolabel|deuter|tritium|scintigraph",
    re.IGNORECASE,
)
IN_VIVO_CONTEXT_WINDOW = 120


def detect_in_vivo_terms(text: str) -> list[str]:
    """Raw vocabulary hits — a hit is not yet a finding (see `detect_in_vivo`)."""
    if not text:
        return []
    return [term for term in IN_VIVO_TERMS if _IN_VIVO_PATTERNS[term].search(text)]


def detect_in_vivo(text: str) -> list[str]:
    """PRD R12.9 / AC-6.6 — terms whose surrounding context survives the
    false-positive filter. On the reference patent this correctly returns [].
    """
    if not text:
        return []

    confirmed: list[str] = []
    for term in IN_VIVO_TERMS:
        for match in _IN_VIVO_PATTERNS[term].finditer(text):
            start = max(0, match.start() - IN_VIVO_CONTEXT_WINDOW)
            context = text[start : match.end() + IN_VIVO_CONTEXT_WINDOW]
            if not IN_VIVO_EXCLUSION_CONTEXT.search(context):
                confirmed.append(term)
                break
    return confirmed


def apply_investment_signal(
    compound: Compound,
    *,
    in_examples: bool,
    in_claims: bool,
    in_prose: bool,
    in_vivo_terms: Sequence[str] = (),
) -> None:
    """PRD R12.9 — the "someone cared" proxy, as a badge of reasons in place of
    the in vivo signal the brief asked for, which this patent does not contain.
    """
    compound.in_examples = in_examples
    compound.in_claims = in_claims
    compound.in_prose = in_prose
    compound.has_in_vivo = bool(in_vivo_terms)

    reasons: list[str] = []
    if in_examples:
        reasons.append("appears in the Examples with full synthesis and NMR/MS characterization")
    if in_claims:
        reasons.append("named in the claims")
    if in_prose:
        reasons.append("discussed in prose outside the tables")
    if in_vivo_terms:
        reasons.append(f"in vivo data reported ({', '.join(in_vivo_terms)})")
    compound.investment_reasons = reasons


TOP_BIN_SCORE = 3
# PRD R11.4 / EC-23 — an Example that joined to no activity row still shows up.
NO_ACTIVITY_DATA = "no activity data"


def _scored(measurements: Sequence[Measurement]) -> list[Measurement]:
    return [m for m in measurements if m.bin_score is not None]


def score_potency(measurements: Sequence[Measurement]) -> int | None:
    """PRD R12.1 — the sum of the available bin scores.

    A blank or undecoded cell contributes nothing at all; it is never scored as
    if it were the lowest bin (R12.9 / EC-7).
    """
    scored = _scored(measurements)
    if not scored:
        return None
    return sum(m.bin_score for m in scored)


def _for_assay(measurements: Sequence[Measurement], assay: str) -> Measurement | None:
    wanted = assay.casefold()
    for measurement in _scored(measurements):
        if (measurement.target_raw or "").casefold() == wanted:
            return measurement
    for measurement in _scored(measurements):
        if wanted in measurement.assay_name_raw.casefold():
            return measurement
    return None


def score_selectivity(
    measurements: Sequence[Measurement], *, target: str, off_target: str | None
) -> int | None:
    """PRD R12.4 — `bin_score(target) − bin_score(off_target)`.

    For the reference patent WIZ is the intended degradation target and ZBTB7A
    the off-target, so D/I scores +2. The roles are arguments, not constants
    (R12.5): swapping them inverts the ranking, and they differ per patent.
    """
    if off_target is None:
        return None
    on = _for_assay(measurements, target)
    off = _for_assay(measurements, off_target)
    if on is None or off is None:
        return None
    return on.bin_score - off.bin_score


def _top_bin_count(measurements: Sequence[Measurement]) -> int:
    return sum(1 for m in _scored(measurements) if m.bin_score == TOP_BIN_SCORE)


def _sort_key(compound: Compound, top_bins: int) -> tuple[int, int, float, float]:
    # PRD R12.2 — potency, then top-bin count, then QED descending, then MW
    # ascending. Missing descriptors sort last rather than winning a tie.
    return (
        -(compound.potency_score or 0),
        -top_bins,
        -(compound.qed if compound.qed is not None else float("-inf")),
        compound.mw if compound.mw is not None else float("inf"),
    )


def _decisive_signal(
    previous: tuple[int, int, float, float], current: tuple[int, int, float, float]
) -> str | None:
    labels = ("potency", "top-bin count", "QED", "MW")
    for value_a, value_b, label in zip(previous, current, labels):
        if value_a != value_b:
            return label
    return None


def _potency_breakdown(measurements: Sequence[Measurement], n_assays: int) -> list[str]:
    scored = _scored(measurements)
    parts = ", ".join(
        f"{m.target_raw or m.assay_name_raw} {m.bin_label_raw}→{m.bin_score}" for m in scored
    )
    lines = [f"potency {sum(m.bin_score for m in scored)} = {parts}"]
    if len(scored) < n_assays:
        # PRD R12.9 / EC-7 — show the gap; never treat a blank as a low value.
        lines.append(
            f"scored on {len(scored)} of {n_assays} assays; blank cells are omitted, "
            "not counted as low"
        )
    if any(m.is_censored for m in scored):
        # PRD R12.8 / EC-17 — letter bins are censored, so LE/LLE stay undefined.
        lines.append("letter-bin values are censored; efficiency metrics undefined")
    return lines


def rank_compounds(
    compounds: Sequence[Compound],
    measurements: Sequence[Measurement],
    *,
    target: str = "WIZ",
    off_target: str | None = "ZBTB7A",
) -> None:
    """PRD R12.1–R12.5 — mutates `potency_score`, `selectivity_score`, `rank`,
    `rank_tie_group` and `rank_rationale` in place.
    """
    by_compound: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        by_compound.setdefault(measurement.compound_id, []).append(measurement)

    n_assays = len({m.assay_group_key for m in measurements}) or 1

    ranked: list[tuple[Compound, tuple[int, int, float, float]]] = []
    for compound in compounds:
        own = by_compound.get(compound.compound_id, [])
        compound.potency_score = score_potency(own)
        compound.selectivity_score = score_selectivity(own, target=target, off_target=off_target)

        if compound.potency_score is None:
            compound.rank = None
            compound.rank_tie_group = None
            compound.rank_rationale = [NO_ACTIVITY_DATA]
            continue

        top_bins = _top_bin_count(own)
        rationale = _potency_breakdown(own, n_assays)
        if compound.selectivity_score is not None:
            on, off = _for_assay(own, target), _for_assay(own, off_target or "")
            rationale.append(
                f"selectivity {compound.selectivity_score:+d} = {target} {on.bin_label_raw} "
                f"({on.bin_score}) − {off_target} {off.bin_label_raw} ({off.bin_score}); "
                f"{off_target} is the off-target, so a high score there is undesirable"
            )
        rationale.append(f"{top_bins} assay(s) at the top bin")
        compound.rank_rationale = rationale
        ranked.append((compound, _sort_key(compound, top_bins)))

    ranked.sort(key=lambda pair: pair[1])

    tie_group = 0
    previous_key: tuple[int, int, float, float] | None = None
    for position, (compound, key) in enumerate(ranked, start=1):
        compound.rank = position
        # PRD R12.3 — the group boundary is where the assay data stops
        # discriminating; QED and MW only order rows inside a group.
        if previous_key is None or key[:2] != previous_key[:2]:
            tie_group += 1
        compound.rank_tie_group = tie_group

        if previous_key is None:
            compound.rank_rationale.append("top of the ranking on potency")
        else:
            signal = _decisive_signal(previous_key, key)
            compound.rank_rationale.append(
                f"ranked below #{position - 1} on {signal}"
                if signal
                else f"tied with #{position - 1} on every ranking signal"
            )
        previous_key = key

    group_sizes: dict[int, int] = {}
    for compound, _ in ranked:
        group_sizes[compound.rank_tie_group] = group_sizes.get(compound.rank_tie_group, 0) + 1
    for compound, _ in ranked:
        size = group_sizes[compound.rank_tie_group]
        if size > 1:
            compound.rank_rationale.append(
                f"tie group {compound.rank_tie_group}: {size} compounds indistinguishable "
                "on the assay data"
            )


def shortlist(compounds: Sequence[Compound], n: int = 10) -> list[Compound]:
    """PRD R12.10 — the top n by rank, each carrying its tie group so the UI can
    show that #10 and #14 are indistinguishable.
    """
    return sorted((c for c in compounds if c.rank is not None), key=lambda c: c.rank)[:n]
