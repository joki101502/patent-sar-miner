"""OCSR runs in bounded chunks (PRD R9.12, R17.4, AC-9.3).

The PRD's 835 MB figure for MolScribe was measured on a batch of **two**. A real
run puts ~47 structure crops through `predict_image_files` in a single call and
peaks at 2843 MB, because activations for the whole batch are live at once.

Batching still matters — the model load is what R9.12 amortizes, and that
happens once regardless — so the fix is to chunk the inference rather than
abandon batching or reload the model per image.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sarmine.config import get_config
from sarmine.structure.molscribe import MolScribeRunner


class _RecordingModel:
    """Stands in for the loaded MolScribe model and records each call's size."""

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def predict_image_files(self, paths, **kwargs):
        self.call_sizes.append(len(paths))
        return [
            {"smiles": f"C{Path(p).stem}", "molfile": "", "confidence": 0.9}
            for p in paths
        ]


@pytest.fixture
def runner(monkeypatch):
    r = MolScribeRunner()
    model = _RecordingModel()
    r._model = model
    monkeypatch.setattr(r, "load", lambda: None)
    return r, model


def _images(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i:02d}.png"
        Image.new("L", (20, 20), 255).save(p)
        paths.append(p)
    return paths


def test_a_large_batch_is_split_into_bounded_chunks(runner, tmp_path):
    r, model = runner
    r.batch_size = 4

    r.predict(_images(tmp_path, 10))

    assert model.call_sizes == [4, 4, 2]


def test_every_input_gets_exactly_one_result_in_order(runner, tmp_path):
    r, model = runner
    r.batch_size = 3
    paths = _images(tmp_path, 7)

    results = r.predict(paths)

    assert len(results) == len(paths)
    assert [res.smiles for res in results] == [f"C{p.stem}" for p in paths]


def test_a_batch_smaller_than_the_chunk_is_a_single_call(runner, tmp_path):
    r, model = runner
    r.batch_size = 8

    r.predict(_images(tmp_path, 2))

    assert model.call_sizes == [2]


def test_a_failing_chunk_does_not_lose_the_other_chunks(runner, tmp_path):
    """One bad crop must not cost the run every other structure."""
    r, model = runner
    r.batch_size = 2

    calls = {"n": 0}
    original = model.predict_image_files

    def flaky(paths, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(paths, **kwargs)

    model.predict_image_files = flaky

    results = r.predict(_images(tmp_path, 4))

    assert len(results) == 4
    assert results[0].error is not None
    assert results[2].smiles is not None


def test_the_chunk_size_has_a_configured_default():
    assert get_config().ocsr_batch_size >= 1
    assert MolScribeRunner().batch_size == get_config().ocsr_batch_size
