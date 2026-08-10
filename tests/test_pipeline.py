"""Part 12 — staged pipeline orchestration (PRD §17.1, §17.2, R17.1–R17.6, AC-9.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sarmine.artifacts.writer import read_bundle
from sarmine.pipeline import (
    PipelineResult,
    classify_table_page,
    find_table_pages,
    run_pipeline,
)
from sarmine.resources import STAGE_ORDER
from sarmine.segment.rulings import Grid

REF_PDF = Path("data/patents/WO2024097932A1.pdf")
FIXTURES = Path(__file__).parent / "fixtures" / "pages"
requires_ref_pdf = pytest.mark.skipif(not REF_PDF.is_file(), reason="reference PDF not present")


def _grid(n_rows: int, n_cols: int) -> Grid:
    xs = [0] + [(i + 1) * 100 for i in range(n_cols)]
    ys = [0] + [(i + 1) * 100 for i in range(n_rows)]
    return Grid(
        n_rows=n_rows,
        n_cols=n_cols,
        y_rulings=ys,
        x_rulings=xs,
        width=xs[-1],
        height=ys[-1],
    )


def test_compound_table_is_identified_by_having_a_structure_column():
    """PRD §3.3 — number | structure | name."""
    kind = classify_table_page(
        _grid(2, 3),
        column_text={0: "3 4", 1: "N HN O", 2: "2-(2,6-dioxopiperidin-3-yl)isoindoline-1,3-dione"},
        column_ink={0: 0.01, 1: 0.30, 2: 0.05},
    )
    assert kind == "compound_table"


def test_activity_table_is_identified_by_its_assay_header():
    """PRD §3.4 — Compound No. | HbF Induction (%) | WIZ EC50 (uM) | ZBTB7A EC50 (uM)."""
    kind = classify_table_page(
        _grid(6, 4),
        column_text={
            0: "Compound No. 1 2 3",
            1: "HbF Induction (%) A A A",
            2: "WIZ EC50 (uM) E D E",
            3: "ZBTB7A EC50 (uM) G G G",
        },
        column_ink={0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01},
    )
    assert kind == "activity_table"


def test_a_page_with_no_grid_is_not_a_table():
    assert classify_table_page(_grid(0, 0), column_text={}, column_ink={}) is None


@requires_ref_pdf
@pytest.mark.slow
def test_table_pages_are_found_in_the_reference_patent(tmp_path):
    """PRD §3.3, §3.4 — Table 1 around pages 61-88, Table 2 on 186-187."""
    from sarmine.sources.pdf import extract_page_images

    pages = extract_page_images(REF_PDF, tmp_path, first=61, last=64)
    found = find_table_pages({p.page_no: p.path for p in pages}, tmp_path)
    assert set(found) == {61, 62, 63, 64}
    assert all(page.kind == "compound_table" for page in found.values())
    assert all(page.rotation.applied in (90, 270) for page in found.values())


@requires_ref_pdf
@pytest.mark.slow
def test_end_to_end_pdf_path_produces_a_bundle(tmp_path):
    """AC-1.3 — with the network disabled the run completes via pdf_ocr and still
    produces a SAR table; AC-9.2 — every stage records its peak RSS."""
    result = run_pipeline(
        REF_PDF,
        out_root=tmp_path,
        allow_network=False,
        run_ocsr=False,
        page_ranges=[(61, 64), (183, 187)],
    )
    assert isinstance(result, PipelineResult)

    bundle = read_bundle(result.bundle_dir)
    assert bundle.manifest.source_mode == "pdf_ocr"
    assert bundle.manifest.pubnum == "WO2024097932A1"
    assert bundle.compounds, "the pdf_ocr path must still produce a SAR table"
    assert bundle.measurements

    # AC-9.2 / R17.6 — the memory budget must be measured, not assumed.
    assert bundle.manifest.stage_peak_rss_mb
    assert set(bundle.manifest.stage_peak_rss_mb).issubset(set(STAGE_ORDER))
    assert bundle.manifest.stage_timings_s

    # AC-8.1 — every compound carries provenance back to a page.
    assert all(c.provenance for c in bundle.compounds)
    assert all(m.provenance.page_no > 0 for m in bundle.measurements)


@requires_ref_pdf
@pytest.mark.slow
def test_progress_callback_reports_every_stage(tmp_path):
    """R14.1 — a blocking run with per-stage status."""
    seen: list[tuple[str, str]] = []
    run_pipeline(
        REF_PDF,
        out_root=tmp_path,
        allow_network=False,
        run_ocsr=False,
        page_ranges=[(63, 63)],
        on_progress=lambda stage, phase, i, n: seen.append((stage, phase)),
    )
    stages = [s for s, phase in seen if phase == "start"]
    assert stages == sorted(stages, key=STAGE_ORDER.index)
    assert "write" in stages
