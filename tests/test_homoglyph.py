"""Homoglyph-repair tests (PRD R9.4, R9.5, EC-8, Plan 5.2).

Every corruption exercised here was measured on this chemotype in spike S2 — all
four of its OPSIN failures were single-character `l`/`1`/`]` confusions. These
tests use a fake parser: the repair loop's contract is about batching and
candidate generation, and it must be testable without a JVM.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import pytest
from lxml import html as lhtml

from sarmine.ocr.homoglyph import CONFUSIONS, RepairResult, generate_candidates, repair_batch
from sarmine.structure.opsin import parse_names, smiles_to_inchikey

FIXTURE_HTML = Path(__file__).parent / "fixtures" / "source" / "WO2024097932A1.html"
REFERENCE_INCHIKEY = "WZPDSZGYLXZFEK-UHFFFAOYSA-N"

# Example 31 as published, carrying five simultaneous `l`-for-`1` corruptions.
MULTI_CORRUPTION = (
    "(S)-4-((6-(l,5-dimethyl-6-oxo-l,6-dihydropyridin-3-yl)-l-methyl-2-oxo-"
    "l,2,3,4-tetrahydroquinolin-7-yl)amino)-2-(2,6-dioxopiperidin-3-yl)isoindoline-l,3-dione"
)
MULTI_TRUE = MULTI_CORRUPTION.replace("(l,", "(1,").replace("-l,", "-1,").replace("-l-", "-1-")


def read_example_names() -> list[str]:
    """The 32 published Example names out of Google Patents' description text.

    The paragraph markers appear as both `[00203]` and `[00203 ]`; missing the
    optional space silently truncates every extracted name.
    """
    flat = re.sub(r"\s+", " ", lhtml.parse(str(FIXTURE_HTML)).getroot().text_content())
    hits = re.findall(r"Example\s+(\d+)\s*:\s*(.+?)(?=\[\s*\d{5}\s*\]|$)", flat)
    return [name.strip() for _number, name in hits]

POMALIDOMIDE = "4-amino-2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione"
# PRD Appendix B.2 — compound 5, long enough for depth 2 to blow past any cap.
REFERENCE_NAME = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-1H-"
    "benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
)

# PRD R9.4 — the measured corruption table, applied to a real full IUPAC name.
CORRUPTIONS = {
    "l->I": "4-amino-2-(2,6-dioxopiperidin-3-yl)isoindoIine-1,3-dione",
    "1->l": "4-amino-2-(2,6-dioxopiperidin-3-yl)isoindoline-l,3-dione",
    "m->rn": "4-arnino-2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione",
    "l->1": "4-amino-2-(2,6-dioxopiperidin-3-y1)isoindoline-1,3-dione",
    "l->]": "4-amino-2-(2,6-dioxopiperidin-3-y])isoindoline-1,3-dione",
}


class CountingParser:
    """Stands in for OPSIN: knows exactly one molecule, counts its invocations."""

    def __init__(self, known: dict[str, str] | None = None) -> None:
        self.known = known if known is not None else {POMALIDOMIDE: "POMA-SMILES"}
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, names: list[str]) -> list[str | None]:
        self.calls += 1
        self.batch_sizes.append(len(names))
        return [self.known.get(n) for n in names]


def test_confusion_table_matches_the_prd() -> None:
    expected = [
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
    assert CONFUSIONS == expected


def test_every_measured_corruption_is_recoverable_at_depth_one() -> None:
    # PRD R9.4 / EC-8 — each row of the corruption table.
    for label, corrupted in CORRUPTIONS.items():
        candidates = {
            name: desc for name, desc in generate_candidates(corrupted, max_depth=1)
        }
        assert POMALIDOMIDE in candidates, label
        # The winning edit is always attributable, whichever strategy found it.
        assert "->" in candidates[POMALIDOMIDE], label


def test_positional_substitution_description_names_the_index() -> None:
    candidates = dict(generate_candidates("1-methy1", max_depth=1))
    assert candidates["1-methyl"] == "1->l @ idx 7"


def test_class_wide_substitution_description_says_it_was_global() -> None:
    # `Compound.homoglyph_repair_applied` must show that every occurrence moved.
    candidates = dict(generate_candidates("1-methy1", max_depth=1))
    assert candidates["l-methyl"] == "1->l (all 2 occurrences)"


def test_candidates_are_generated_lazily() -> None:
    long_name = POMALIDOMIDE * 40
    started = time.perf_counter()
    gen = generate_candidates(long_name, max_candidates=10_000, max_depth=2)
    assert isinstance(gen, Iterator)
    first = list(islice(gen, 5))
    elapsed = time.perf_counter() - started
    assert len(first) == 5
    # A non-lazy depth-2 expansion of this name is millions of strings.
    assert elapsed < 0.5


def test_candidate_generation_respects_the_cap() -> None:
    # PRD R9.4 — depth 2 over a long name is combinatorially large.
    produced = list(generate_candidates(REFERENCE_NAME, max_candidates=25, max_depth=2))
    assert len(produced) == 25
    assert len({name for name, _ in produced}) == 25


def test_depth_one_candidates_come_before_depth_two() -> None:
    produced = list(generate_candidates("methy1-oI", max_candidates=200, max_depth=2))
    depths = [desc.count(";") + 1 for _, desc in produced]
    assert depths == sorted(depths)
    assert 1 in depths and 2 in depths


def test_a_name_that_parses_as_is_is_returned_at_depth_zero() -> None:
    parser = CountingParser()
    (result,) = repair_batch([POMALIDOMIDE], parser)
    assert result == RepairResult(
        original=POMALIDOMIDE,
        repaired=POMALIDOMIDE,
        substitution=None,
        smiles="POMA-SMILES",
        depth=0,
    )
    assert parser.calls == 1


def test_repair_batch_recovers_every_measured_corruption() -> None:
    parser = CountingParser()
    results = repair_batch(list(CORRUPTIONS.values()), parser)
    assert [r.repaired for r in results] == [POMALIDOMIDE] * len(CORRUPTIONS)
    assert all(r.smiles == "POMA-SMILES" for r in results)
    assert all(r.depth == 1 for r in results)
    assert all(r.substitution and "->" in r.substitution for r in results)


def test_parse_batch_is_called_a_bounded_number_of_times() -> None:
    # PRD R9.2 — per-call JVM startup is ~5 s per molecule; one call per candidate
    # would be hours. At most one call for the originals plus one per depth.
    parser = CountingParser(known={})
    names = list(CORRUPTIONS.values()) + [POMALIDOMIDE] * 20
    repair_batch(names, parser, max_depth=2)
    assert parser.calls <= 3


def test_candidate_pools_are_parsed_in_one_call_per_depth() -> None:
    parser = CountingParser()
    repair_batch(list(CORRUPTIONS.values()), parser, max_depth=2)
    assert parser.calls == 2
    assert parser.batch_sizes[0] == len(CORRUPTIONS)
    assert parser.batch_sizes[1] > len(CORRUPTIONS)


def test_an_unrepairable_name_is_a_review_trigger_not_an_error() -> None:
    # PRD R9.5 / EC-8 — failing loudly is the feature.
    parser = CountingParser(known={})
    (result,) = repair_batch(["definitely not a chemical name"], parser)
    assert result.repaired is None
    assert result.smiles is None
    assert result.substitution is None


def test_results_align_one_to_one_with_the_input_order() -> None:
    parser = CountingParser()
    names = [CORRUPTIONS["l->I"], "garbage", POMALIDOMIDE]
    results = repair_batch(names, parser)
    assert [r.original for r in results] == names
    assert [r.depth for r in results] == [1, 0, 0]
    assert results[1].repaired is None


def test_no_candidates_are_generated_when_every_name_parses() -> None:
    parser = CountingParser()
    repair_batch([POMALIDOMIDE, POMALIDOMIDE], parser)
    assert parser.calls == 1


# ------------------------------------------- multi-corruption names (real corpus)


def _positional_only(name: str, *, max_depth: int) -> set[str]:
    return {
        candidate
        for candidate, desc in generate_candidates(
            name, max_candidates=50_000, max_depth=max_depth
        )
        if all("@ idx" in part for part in desc.split("; "))
    }


def test_multiple_simultaneous_corruptions_are_out_of_reach_of_positional_repair() -> None:
    # Why the class-wide strategy exists: Google Patents' description text is itself
    # OCR output, and a single Example name carries five `l`-for-`1` corruptions.
    assert sum(1 for a, b in zip(MULTI_CORRUPTION, MULTI_TRUE) if a != b) == 5

    assert MULTI_TRUE not in _positional_only(MULTI_CORRUPTION, max_depth=2)

    class_wide = {
        candidate
        for candidate, desc in generate_candidates(
            MULTI_CORRUPTION, max_candidates=200, max_depth=1
        )
        if "@ idx" not in desc
    }
    assert MULTI_TRUE in class_wide


def test_the_context_aware_rule_leaves_legitimate_l_characters_alone() -> None:
    # A blind global `l`->`1` also destroys `methyl`, `phenyl` and `isoindoline`.
    candidates = dict(generate_candidates(MULTI_CORRUPTION, max_candidates=200, max_depth=1))
    assert candidates[MULTI_TRUE] == "l->1 (context-aware)"
    assert MULTI_CORRUPTION.replace("l", "1") in candidates


def test_ocr_corrupted_stereo_descriptors_are_repaired() -> None:
    # Example 24 begins `(7?)-4-((l,3-dimethyl...`; Example 25's pair member reads `(5)-`.
    for corrupted, expected in (("(7?)-x", "(R)-x"), ("(l?)-x", "(R)-x"), ("(5)-x", "(S)-x")):
        candidates = dict(generate_candidates(corrupted, max_candidates=200, max_depth=1))
        assert expected in candidates, corrupted


def test_an_enantiomer_pair_is_reduced_to_its_first_member_and_flagged() -> None:
    # PRD R9.22 — keep stereochemistry as named, never enumerate; but never hide
    # that a pair was present either.
    parser = CountingParser(known={POMALIDOMIDE: "POMA-SMILES"})
    joined = f"{POMALIDOMIDE} & (R)-something-else"
    (result,) = repair_batch([joined], parser)

    assert result.original == joined
    assert result.repaired == POMALIDOMIDE
    assert result.enantiomer_pair is True
    assert result.substitution and "enantiomer pair" in result.substitution


def test_a_plain_name_is_not_flagged_as_an_enantiomer_pair() -> None:
    parser = CountingParser()
    (result,) = repair_batch([POMALIDOMIDE], parser)
    assert result.enantiomer_pair is False


def test_the_call_budget_survives_the_class_wide_strategy() -> None:
    # PRD R9.2 — still at most one call for the originals plus one per depth, no
    # matter how many strategies contribute candidates to a depth's pool.
    parser = CountingParser(known={})
    names = list(CORRUPTIONS.values()) + [MULTI_CORRUPTION] * 10
    repair_batch(names, parser, max_depth=2)
    assert parser.calls <= 3


@pytest.mark.slow
def test_the_real_thirty_two_example_names_are_recovered() -> None:
    """The acceptance bar: the published Example names, as Google Patents has them.

    That text is itself OCR output and carries the same `l`-for-`1` corruption as the
    scan. Only Example 3, which happens to contain no corrupted locant, parses untouched.
    """
    names = read_example_names()
    assert len(names) == 32

    def parse_batch(batch: list[str]) -> list[str | None]:
        return [r.smiles for r in parse_names(batch)]

    untouched = [i for i, r in enumerate(parse_names(names), start=1) if r.smiles]
    assert untouched == [3]

    results = repair_batch(names, parse_batch)
    recovered = [r for r in results if r.smiles]
    assert len(recovered) >= 30, [r.original for r in results if not r.smiles]

    example_14 = results[13]
    assert smiles_to_inchikey(example_14.smiles or "") == REFERENCE_INCHIKEY

    # Every repair that was actually applied is attributable for the review queue.
    assert all(r.substitution for r in recovered if r.depth > 0 or r.enantiomer_pair)
