"""Bounded homoglyph-repair retry loop (PRD R9.4/R9.5, EC-8, Plan 5.2).

OPSIN has zero tolerance for OCR corruption and that is the dominant failure mode
of the name channel: all four OPSIN failures in spike S2 were single-character
`l`/`1`/`]` confusions, not nomenclature limits. Candidates are generated lazily
and capped, and the whole candidate pool for a depth is parsed in ONE call —
per-call JVM startup is ~5 s per molecule (PRD R9.2).

R9.4's positional loop was validated on names carrying a single corruption. The real
corpus is harsher: Google Patents' "machine-readable" description text is itself OCR
output, and one published Example name carries five simultaneous `l`-for-`1`
corruptions, which no depth-2 positional search can reach. So each depth tries
class-wide repairs — one candidate per confusion, every occurrence replaced at once —
before the positional candidates, which remain correct for single-corruption names.

A name that survives repair unparsed is a high-precision review-queue trigger,
not a defect (PRD R9.5).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

CONFUSIONS: list[tuple[str, str]] = [
    ("I", "l"),
    ("l", "I"),
    ("l", "1"),
    ("1", "l"),
    ("O", "0"),
    ("0", "O"),
    ("rn", "m"),
    ("m", "rn"),
    ("ii", "n"),
    ("]", "l"),
    ("|", "l"),
    ("\u2013", "-"),
    ("\u2014", "-"),
]

ParseBatch = Callable[[list[str]], list[str | None]]

# A blind class-wide `l`->`1` also destroys `methyl`, `phenyl` and `isoindoline`, so
# the `l`/`1` class additionally gets a positional-context variant. These two rules
# were measured to fix `lH-indazol`, `lH-benzo[d]`, `-l,3-dione`, `((l-methyl` and
# `(l,2-benzoxazol` without touching a legitimate `l`.
_CONTEXT_L_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"l(?=H[-\]])"), "1"),
    (re.compile(r"([-(\[,]\s*)l(?=\s*[,)\]-])"), r"\g<1>1"),
]

# OCR of a stereo descriptor: `(R)` is read as `(7?)` or `(l?)`, `(S)` as `(5)`.
_STEREO_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\((?:7\?|l\?|R\?)\)"), "(R)"),
    (re.compile(r"\(5\)"), "(S)"),
]

ENANTIOMER_SEPARATOR = "&"

_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"})
# Two ordinary words either side of a break really are two words ("benzoic acid");
# anything else is one name that the cell wrapped.
_WORD_BREAK = re.compile(r"(?<=[A-Za-z])\n(?=[A-Za-z])")


def clean_ocr_name(raw: str) -> str:
    """Reassemble a name cell OCR'd as several physical lines.

    A table cell wraps a long IUPAC name mid-name, and the hyphens it wraps on
    are part of the name. Joining the lines with a space therefore inserts a
    space OPSIN rejects: compound 5's name cell comes back as four lines, and
    joined with spaces it fails to parse even though the characters are right.
    """
    text = raw.translate(_DASHES).strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    # Mark the genuine word boundaries, close up everything else, restore them.
    text = _WORD_BREAK.sub("\x00", text)
    text = text.replace("\n", "")
    text = text.replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


@dataclass
class RepairResult:
    """One name's outcome. `substitution` goes into `Compound.homoglyph_repair_applied`."""

    original: str
    repaired: str | None
    substitution: str | None
    smiles: str | None
    depth: int
    enantiomer_pair: bool = False


def split_enantiomer_pair(name: str) -> tuple[str, int]:
    """Return the first listed enantiomer and how many members were listed.

    PRD R9.22 — stereochemistry is kept as named and never enumerated, so an
    `&`-joined heading is represented by its first member. The count is reported so
    the caller can flag the row rather than silently hiding that a pair was present.
    """
    if ENANTIOMER_SEPARATOR not in name:
        return name, 1
    segments = [segment.strip() for segment in name.split(ENANTIOMER_SEPARATOR)]
    segments = [segment for segment in segments if segment]
    if not segments:
        return name, 1
    return segments[0], len(segments)


def _apply_context_rules(name: str, rules: Sequence[tuple[re.Pattern[str], str]]) -> str:
    for pattern, replacement in rules:
        name = pattern.sub(replacement, name)
    return name


def _global_transforms(name: str) -> list[tuple[str, str]]:
    """Whole-name repairs: `(candidate, description)` for each one that changes `name`.

    Class-wide substitution is what survives the real corpus: an OCR'd name routinely
    carries the same confusion at five positions at once, which positional depth 2
    cannot reach.
    """
    out: list[tuple[str, str]] = []
    for src, dst in CONFUSIONS:
        occurrences = name.count(src)
        candidate = name.replace(src, dst)
        if occurrences and candidate != name:
            plural = "occurrence" if occurrences == 1 else "occurrences"
            out.append((candidate, f"{src}->{dst} (all {occurrences} {plural})"))

    contextual = _apply_context_rules(name, _CONTEXT_L_RULES)
    if contextual != name:
        out.append((contextual, "l->1 (context-aware)"))

    stereo = _apply_context_rules(name, _STEREO_RULES)
    if stereo != name:
        out.append((stereo, "stereo descriptor repair"))
    return out


def _single_substitutions(name: str) -> Iterator[tuple[str, str]]:
    for src, dst in CONFUSIONS:
        start = 0
        while True:
            idx = name.find(src, start)
            if idx < 0:
                break
            yield name[:idx] + dst + name[idx + len(src) :], f"{src}->{dst} @ idx {idx}"
            start = idx + 1


def _iter_candidates(
    name: str, max_candidates: int, max_depth: int
) -> Iterator[tuple[str, str, int]]:
    """Yield `(candidate, description, depth)`, one "edit class" per depth.

    A depth's pool is class-wide repairs first — one candidate per confusion, which is
    where the real-corpus wins are — then the positional candidates the PRD's R9.4 loop
    describes, which remain correct for single-corruption names.
    """
    seen = {name}
    produced = 0
    globals_at_depth: list[tuple[str, str]] = []
    frontier: list[tuple[str, str]] = [(name, "")]

    for depth in range(1, max_depth + 1):
        next_globals: list[tuple[str, str]] = []
        next_frontier: list[tuple[str, str]] = []

        # Depth 2's class-wide stage composes two whole-name repairs, which is what
        # `(7?)`-plus-`l`-for-`1` headings need.
        global_bases = [(name, "")] if depth == 1 else globals_at_depth
        for base, base_desc in global_bases:
            for candidate, desc in _global_transforms(base):
                if candidate in seen:
                    continue
                seen.add(candidate)
                full_desc = f"{base_desc}; {desc}" if base_desc else desc
                next_globals.append((candidate, full_desc))
                next_frontier.append((candidate, full_desc))
                yield candidate, full_desc, depth
                produced += 1
                if produced >= max_candidates:
                    return

        for base, base_desc in frontier:
            for candidate, desc in _single_substitutions(base):
                if candidate in seen:
                    continue
                seen.add(candidate)
                full_desc = f"{base_desc}; {desc}" if base_desc else desc
                next_frontier.append((candidate, full_desc))
                yield candidate, full_desc, depth
                produced += 1
                if produced >= max_candidates:
                    return

        if not next_frontier:
            return
        globals_at_depth = next_globals
        frontier = next_frontier


def generate_candidates(
    name: str, *, max_candidates: int = 200, max_depth: int = 2
) -> Iterator[tuple[str, str]]:
    """Lazily yield `(candidate_name, substitution_description)`.

    Ordering is class-wide repairs then positional ones, depth 1 before depth 2.
    """
    for candidate, desc, _depth in _iter_candidates(name, max_candidates, max_depth):
        yield candidate, desc


def repair_batch(
    names: Sequence[str],
    parse_batch: ParseBatch,
    *,
    max_candidates: int = 200,
    max_depth: int = 2,
) -> list[RepairResult]:
    """Repair a whole batch of names with at most `1 + max_depth` parser calls."""
    originals = list(names)
    if not originals:
        return []

    seeds: list[str] = []
    pair_notes: list[str | None] = []
    for name in originals:
        seed, members = split_enantiomer_pair(name)
        seeds.append(seed)
        pair_notes.append(
            f"enantiomer pair: first of {members} taken" if members > 1 else None
        )

    parsed = parse_batch(seeds)
    results = [
        RepairResult(
            original=original,
            repaired=seed if smiles else None,
            substitution=note if smiles else None,
            smiles=smiles or None,
            depth=0,
            enantiomer_pair=note is not None,
        )
        for original, seed, note, smiles in zip(
            originals, seeds, pair_notes, parsed, strict=False
        )
    ]

    pending = [i for i, result in enumerate(results) if result.smiles is None]
    iterators = {i: _iter_candidates(seeds[i], max_candidates, max_depth) for i in pending}
    lookahead: dict[int, tuple[str, str, int]] = {}

    for depth in range(1, max_depth + 1):
        if not pending:
            break
        pool: list[tuple[int, str, str]] = []
        for i in pending:
            item = lookahead.pop(i, None) or next(iterators[i], None)
            while item is not None and item[2] == depth:
                pool.append((i, item[0], item[1]))
                item = next(iterators[i], None)
            if item is not None:
                lookahead[i] = item
        if not pool:
            break

        # PRD R9.2 — one call for the entire pool, never one call per candidate.
        candidate_smiles = parse_batch([candidate for _, candidate, _ in pool])
        for (i, candidate, desc), smiles in zip(pool, candidate_smiles, strict=False):
            if not smiles or results[i].smiles is not None:
                continue
            note = pair_notes[i]
            results[i] = RepairResult(
                original=originals[i],
                repaired=candidate,
                substitution=f"{note}; {desc}" if note else desc,
                smiles=smiles,
                depth=depth,
                enantiomer_pair=note is not None,
            )
        pending = [i for i in pending if results[i].smiles is None]

    return results
