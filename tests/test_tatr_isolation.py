"""TATR runs out of process so its memory is reclaimed (PRD R17.1, AC-9.3).

Measured on this machine: importing torch costs 207 MB, transformers another
65 MB, and a single TATR inference takes the process to **1119 MB** — the
attention over a 1000 px page, not the 110 MB of weights. MolScribe loaded
afterwards adds nothing to the high-water mark, so TATR, not OCSR, is what puts
a full run over the 2400 MB budget.

`ru_maxrss` is a high-water mark and never falls, so freeing the model in-process
cannot recover the peak. R17.1 already relies on this for tesseract and OPSIN:
"their memory is reclaimed on exit". TATR gets the same treatment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sarmine.segment.rulings import Grid
from sarmine.segment.tatr import detect_grids_isolated, grid_to_dict, grid_from_dict

PAGE = Path(__file__).parent / "fixtures" / "pages" / "p-186-000.png"


def test_no_pages_means_no_subprocess():
    assert detect_grids_isolated([]) == {}


def test_grid_survives_a_json_round_trip():
    """The worker can only return JSON, so the Grid has to serialize losslessly."""
    grid = Grid(
        n_rows=2,
        n_cols=3,
        y_rulings=[0, 100, 200],
        x_rulings=[0, 50, 150, 300],
        width=300,
        height=200,
        detector="tatr",
    )
    restored = grid_from_dict(json.loads(json.dumps(grid_to_dict(grid))))

    assert restored.y_rulings == grid.y_rulings
    assert restored.x_rulings == grid.x_rulings
    assert (restored.n_rows, restored.n_cols) == (2, 3)


def test_missing_image_yields_none_rather_than_an_exception(tmp_path):
    """A second opinion is optional; its absence must never abort a run."""
    assert detect_grids_isolated([tmp_path / "nope.png"]) == {}


@pytest.mark.slow
@pytest.mark.skipif(not PAGE.is_file(), reason="fixture page missing")
def test_worker_returns_a_grid_for_a_real_table_page():
    grids = detect_grids_isolated([PAGE])
    grid = grids.get(str(PAGE))
    assert grid is not None
    assert grid.n_rows >= 2
    assert grid.n_cols >= 2
    assert grid.detector == "tatr"


@pytest.mark.slow
@pytest.mark.skipif(not PAGE.is_file(), reason="fixture page missing")
def test_the_parent_process_never_imports_torch():
    """The whole point: the parent must stay small. If torch is imported in the
    parent, the 207 MB and everything TATR allocates land on the run's peak."""
    script = (
        "import sys;"
        "from pathlib import Path;"
        "from sarmine.segment.tatr import detect_grids_isolated;"
        f"detect_grids_isolated([Path({str(PAGE)!r})]);"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr
