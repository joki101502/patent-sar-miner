"""Name-channel tests: batched OPSIN and the PubChem fallback.

Covers PRD R9.1–R9.6, AC-3.1/AC-3.2 (RDKit-valid SMILES), AC-3.4 (the verified
reference compound), EC-8 (OCR-corrupted names), and the R9.2 batching budget.
The default run is offline; live PubChem calls are marked `network`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from rdkit import Chem

from sarmine.ocr.homoglyph import repair_batch
from sarmine.structure import pubchem
from sarmine.structure.opsin import (
    OpsinResult,
    opsin_version,
    parse_names,
    smiles_to_inchikey,
)

# PRD Appendix B.2 / AC-3.4 — compound 5, page 63. Ground truth.
REFERENCE_NAME = (
    "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-1H-"
    "benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
)
REFERENCE_INCHIKEY = "WZPDSZGYLXZFEK-UHFFFAOYSA-N"

requires_network = pytest.mark.skipif(
    os.environ.get("SARMINE_TEST_NETWORK") != "1",
    reason="set SARMINE_TEST_NETWORK=1 to exercise the live PubChem fetch",
)

POMALIDOMIDE = "4-amino-2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione"
POMALIDOMIDE_INCHIKEY = "UVSMNLNDYGZFPF-UHFFFAOYSA-N"

# PRD R9.4 — the measured corruption table applied to a real full IUPAC name.
CORRUPTED_NAMES = [
    "4-amino-2-(2,6-dioxopiperidin-3-yl)isoindoIine-1,3-dione",
    "4-amino-2-(2,6-dioxopiperidin-3-yl)isoindoline-l,3-dione",
    "4-arnino-2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione",
    "4-amino-2-(2,6-dioxopiperidin-3-y1)isoindoline-1,3-dione",
    "4-amino-2-(2,6-dioxopiperidin-3-y])isoindoline-1,3-dione",
]

KNOWN_GOOD_NAMES = [
    "ethane",
    "benzene",
    "pyridine",
    "2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione",
    POMALIDOMIDE,
    REFERENCE_NAME,
]


# --------------------------------------------------------------------------- OPSIN


def test_opsin_version_is_reported_for_provenance() -> None:
    assert opsin_version() == "opsin@2.9.0"


def test_empty_input_never_launches_a_jvm(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("py2opsin must not be called for an empty batch")

    monkeypatch.setattr("sarmine.structure.opsin.py2opsin", explode)
    assert parse_names([]) == []


def test_known_names_parse_and_every_smiles_is_rdkit_valid() -> None:
    # AC-3.2 — every OPSIN-derived SMILES must survive RDKit.
    results = parse_names(KNOWN_GOOD_NAMES)
    assert [r.name for r in results] == KNOWN_GOOD_NAMES
    assert all(isinstance(r, OpsinResult) for r in results)
    for result in results:
        assert result.status == "SUCCESS", result.name
        assert result.smiles
        assert Chem.MolFromSmiles(result.smiles) is not None, result.name
        assert result.inchikey and len(result.inchikey) == 27


def test_reference_compound_reproduces_the_published_inchikey() -> None:
    # AC-3.4 / PRD Appendix B.2 — ground truth, not a regression baseline.
    (result,) = parse_names([REFERENCE_NAME])
    assert result.inchikey == REFERENCE_INCHIKEY


def test_a_garbage_name_fails_loudly() -> None:
    # PRD R9.5 / EC-8 — OPSIN failing loudly is the feature.
    (result,) = parse_names(["definitely not a chemical name at all"])
    assert result.status == "FAILURE"
    assert result.smiles is None
    assert result.inchikey is None


def test_ambiguity_and_status_are_captured_per_name() -> None:
    # PRD R9.3 — free, uncorrelated confidence signals; do not discard them.
    ethane, xylene, garbage, reference = parse_names(
        ["ethane", "xylene", "zzznotachemicalzzz", REFERENCE_NAME]
    )
    assert (ethane.status, ethane.ambiguous) == ("SUCCESS", False)
    assert xylene.ambiguous is True
    assert xylene.status == "WARNING"
    assert xylene.smiles
    assert (garbage.status, garbage.ambiguous) == ("FAILURE", False)
    assert (reference.status, reference.ambiguous) == ("SUCCESS", False)


def test_multiline_ocr_text_does_not_break_batch_alignment() -> None:
    results = parse_names(["ethane\n", "benz\nene", "pyridine"])
    assert [r.name for r in results] == ["ethane\n", "benz\nene", "pyridine"]
    assert results[0].smiles == "CC"
    assert results[2].smiles


def test_fifty_four_names_parse_in_one_batched_call_under_five_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PRD R9.2 / Plan 5.5 — un-batched this is ~5 s per molecule, i.e. 4.5 minutes.
    import sarmine.structure.opsin as opsin_module

    real = opsin_module.py2opsin
    calls: list[int] = []

    def spy(chemical_name, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(len(chemical_name))
        return real(chemical_name, *args, **kwargs)

    monkeypatch.setattr(opsin_module, "py2opsin", spy)

    names = [KNOWN_GOOD_NAMES[i % len(KNOWN_GOOD_NAMES)] for i in range(54)]
    started = time.perf_counter()
    results = parse_names(names)
    elapsed = time.perf_counter() - started

    assert len(results) == 54
    assert all(r.smiles for r in results)
    assert len(calls) == 1, "one JVM start for the whole batch"
    assert elapsed < 5.0, f"batched OPSIN took {elapsed:.2f}s"


def test_smiles_to_inchikey_rejects_unparseable_smiles() -> None:
    assert smiles_to_inchikey("not-a-smiles(((") is None
    assert smiles_to_inchikey("CC") is not None


def test_homoglyph_repair_recovers_the_pomalidomide_inchikey_through_opsin() -> None:
    # EC-8 end to end: OCR corruption -> repair loop -> OPSIN -> the right molecule.
    def parse_batch(names: list[str]) -> list[str | None]:
        return [r.smiles for r in parse_names(names)]

    results = repair_batch(CORRUPTED_NAMES, parse_batch)
    assert all(r.repaired == POMALIDOMIDE for r in results)
    assert all(r.depth == 1 for r in results)
    assert [smiles_to_inchikey(r.smiles or "") for r in results] == [
        POMALIDOMIDE_INCHIKEY
    ] * len(CORRUPTED_NAMES)


# ------------------------------------------------------------------------- PubChem


def _payload(smiles_key: str) -> dict[str, object]:
    return {
        "PropertyTable": {
            "Properties": [
                {"CID": 2244, smiles_key: "CC(=O)Oc1ccccc1C(=O)O", "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}
            ]
        }
    }


@pytest.mark.parametrize("smiles_key", ["SMILES", "ConnectivitySMILES", "CanonicalSMILES"])
def test_every_pubchem_smiles_key_spelling_is_understood(smiles_key: str) -> None:
    # PRD R9.6 — PubChem renamed this property; older docs would KeyError.
    smiles, inchikey = pubchem.parse_property_payload(_payload(smiles_key))
    assert smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_an_empty_payload_yields_nothing_rather_than_raising() -> None:
    assert pubchem.parse_property_payload({}) == (None, None)
    assert pubchem.parse_property_payload({"PropertyTable": {"Properties": []}}) == (None, None)


def test_the_cache_prevents_a_second_network_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetches: list[str] = []

    def fake_fetch(name: str) -> dict[str, object]:
        fetches.append(name)
        return _payload("SMILES")

    monkeypatch.setattr(pubchem, "_fetch", fake_fetch)
    first = pubchem.resolve_name("aspirin", tmp_path)
    second = pubchem.resolve_name("aspirin", tmp_path)

    assert first == second == ("CC(=O)Oc1ccccc1C(=O)O", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert fetches == ["aspirin"]


def test_a_cached_answer_is_returned_even_with_the_network_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pubchem, "_fetch", lambda name: _payload("ConnectivitySMILES"))
    pubchem.resolve_name("aspirin", tmp_path)

    monkeypatch.setattr(
        pubchem,
        "_fetch",
        lambda name: pytest.fail("offline resolution must not touch the network"),
    )
    assert pubchem.resolve_name("aspirin", tmp_path, allow_network=False)[1] == (
        "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    )


def test_a_cold_cache_offline_returns_nothing_rather_than_raising(tmp_path: Path) -> None:
    assert pubchem.resolve_name("aspirin", tmp_path, allow_network=False) == (None, None)


def test_a_network_failure_is_absorbed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str) -> dict[str, object]:
        raise OSError("no route to host")

    monkeypatch.setattr(pubchem, "_fetch", boom)
    assert pubchem.resolve_name("aspirin", tmp_path) == (None, None)


def test_requests_are_throttled_below_the_published_rate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PRD R9.6 — <=5 requests/second.
    monkeypatch.setattr(pubchem, "_fetch", lambda name: _payload("SMILES"))
    pubchem.reset_throttle()
    started = time.perf_counter()
    for i in range(3):
        pubchem.resolve_name(f"compound-{i}", tmp_path)
    elapsed = time.perf_counter() - started
    assert elapsed >= 2 * pubchem.MIN_REQUEST_INTERVAL_S


def test_a_trivial_name_that_opsin_cannot_handle_is_the_fallback_case() -> None:
    # PRD R9.6 — trade/trivial names are structurally out of OPSIN's reach.
    (result,) = parse_names(["pomalidomide"])
    assert result.status == "FAILURE"


@pytest.mark.network
@requires_network
def test_live_pubchem_resolves_a_trivial_name(tmp_path: Path) -> None:
    smiles, inchikey = pubchem.resolve_name("aspirin", tmp_path)
    assert inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert smiles and Chem.MolFromSmiles(smiles) is not None


@pytest.mark.network
@requires_network
def test_live_pubchem_writes_a_reusable_cache_entry(tmp_path: Path) -> None:
    pubchem.resolve_name("aspirin", tmp_path)
    cached = list(tmp_path.glob("*.json"))
    assert cached
    assert json.loads(cached[0].read_text("utf-8"))
