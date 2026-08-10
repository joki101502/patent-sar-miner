"""The image channel must pair OCSR results with the right compound (PRD R8.7, R9.7).

`_run_image_channel` walks the structure crops and writes each result onto a
compound. It previously did that with an index taken from the *cell* list, so a
compound list that was even one element shorter raised `IndexError` and threw
away a four-minute run. Pairing, not indexing, is the invariant under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sarmine.artifacts.schema import Compound
from sarmine.pipeline import CompoundCell, _run_image_channel


def _crop(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    Image.new("L", (40, 40), 255).save(path)
    return path


def _cell(crop: Path | None) -> CompoundCell:
    return CompoundCell(
        page_no=63, compound_number=None, name_raw="", structure_crop=crop, provenance=[]
    )


def _compound(local_id: str) -> Compound:
    return Compound(compound_id=f"X:{local_id}", compound_local_id=local_id)


class _FakeRunner:
    """Stands in for MolScribe: one result per crop, in order."""

    def __init__(self, *args, **kwargs) -> None:
        self.freed = False

    def predict(self, paths):
        from sarmine.structure.molscribe import OcsrResult

        return [
            OcsrResult(
                smiles=f"C{i}",
                inchikey=f"KEY{i}",
                confidence_molecule=0.9,
                confidence_min_atom=0.8,
                confidence_min_bond=0.7,
                rdkit_valid=True,
            )
            for i, _ in enumerate(paths)
        ]

    def free(self) -> None:
        self.freed = True


@pytest.fixture
def patched(monkeypatch):
    import sarmine.segment.classify as classify
    import sarmine.structure.molscribe as molscribe

    monkeypatch.setattr(
        classify, "classify_crops", lambda paths: [("clean", 0.99) for _ in paths]
    )
    monkeypatch.setattr(molscribe, "MolScribeRunner", _FakeRunner)


def test_more_cells_than_compounds_does_not_raise(patched, tmp_path):
    """The exact shape that killed a real run: three crops, two compounds."""
    cells = [_cell(_crop(tmp_path, f"c{i}.png")) for i in range(3)]
    compounds = [_compound("1"), _compound("2")]

    _run_image_channel(compounds, cells)

    assert compounds[0].smiles_from_image == "C0"
    assert compounds[1].smiles_from_image == "C1"


def test_results_land_on_the_compound_that_owns_the_crop(patched, tmp_path):
    """A cell with no crop must not shift every later result by one."""
    cells = [
        _cell(_crop(tmp_path, "a.png")),
        _cell(None),
        _cell(_crop(tmp_path, "b.png")),
    ]
    compounds = [_compound("1"), _compound("2"), _compound("3")]

    _run_image_channel(compounds, cells)

    assert compounds[0].smiles_from_image == "C0"
    assert compounds[1].smiles_from_image is None
    assert compounds[2].smiles_from_image == "C1"


def test_confidences_are_recorded_for_the_review_gate(patched, tmp_path):
    """R13.1 — the queue gates on minimum atom/bond confidence, so both must survive."""
    cells = [_cell(_crop(tmp_path, "a.png"))]
    compounds = [_compound("1")]

    _run_image_channel(compounds, cells)

    assert compounds[0].ocsr_confidence_min_atom == 0.8
    assert compounds[0].ocsr_confidence_min_bond == 0.7


def test_markush_crops_are_flagged_and_never_sent_to_ocsr(monkeypatch, tmp_path):
    """R8.7 / EC-12 — OCSR would hallucinate a concrete structure for a generic one."""
    import sarmine.segment.classify as classify
    import sarmine.structure.molscribe as molscribe

    monkeypatch.setattr(
        classify, "classify_crops", lambda paths: [("markush", 0.95) for _ in paths]
    )

    class _Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError("OCSR must never run on a Markush crop")

    monkeypatch.setattr(molscribe, "MolScribeRunner", _Forbidden)

    cells = [_cell(_crop(tmp_path, "m.png"))]
    compounds = [_compound("1")]

    _run_image_channel(compounds, cells)

    assert compounds[0].markush_detected is True
    assert compounds[0].smiles_from_image is None
