"""Models must not stack (PRD §17.1, §17.2, R17.1, R17.4, AC-9.3).

The pipeline order is a memory requirement: peak RSS must equal the largest
stage, not the sum of stages, because the deploy target has 2.7 GB and the
budget is 2400 MB. A measured run came in at 2895 MB — the segmentation model
was still resident when MolScribe loaded, so the two summed instead of peaking.
"""

from __future__ import annotations

import pytest

from sarmine.resources import PEAK_RSS_BUDGET_MB


def test_tatr_can_release_its_cached_model():
    """R17.4 — the segmentation model has to be droppable between stages."""
    from sarmine.segment import tatr

    tatr._MODEL_CACHE["sentinel"] = (object(), object())
    tatr.free()
    assert tatr._MODEL_CACHE == {}


def test_release_calls_free_on_a_module():
    """`release` already frees objects; it must also handle module-level caches."""
    from sarmine.resources import release
    from sarmine.segment import tatr

    tatr._MODEL_CACHE["sentinel"] = (object(), object())
    release(tatr)
    assert tatr._MODEL_CACHE == {}


def test_the_segment_stage_releases_tatr_before_the_image_channel(monkeypatch, tmp_path):
    """The ordering guarantee itself, not just the ability to free."""
    from sarmine.segment import tatr

    freed: list[str] = []
    monkeypatch.setattr(tatr, "free", lambda: freed.append("tatr"))

    import sarmine.pipeline as pipeline

    monkeypatch.setattr(pipeline, "find_table_pages", lambda images, work: {})
    monkeypatch.setattr(
        pipeline,
        "_run_image_channel",
        lambda compounds, cells: freed.append("image_channel"),
    )

    class _Resolved:
        pubnum = "WO2024097932A1"
        source_mode = "pdf_ocr"
        structured = None
        pdf_path = None
        cache_dir = tmp_path
        n_pages = 1
        anomalies: list = []

    monkeypatch.setattr(pipeline, "resolve", lambda *a, **k: _Resolved())
    monkeypatch.setattr(pipeline, "_collect_pages", lambda *a, **k: ({}, "prose"))

    pipeline.run_pipeline("x.pdf", out_root=tmp_path, run_ocsr=True, run_id="RELEASE")

    assert freed.index("tatr") < freed.index("image_channel")


def test_budget_constant_matches_the_acceptance_criterion():
    assert PEAK_RSS_BUDGET_MB == pytest.approx(2400.0)
