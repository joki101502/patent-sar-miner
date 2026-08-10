"""Table Transformer — the second table-structure detector (PRD R8.3, Plan 4.3).

Morphology alone found only **37 of 54** compound-table cells on the reference
patent (PRD §8.1), missing compounds 3, 7, 9, 14, 16, 20, 22, 26, 28, 30, 34,
36, 38, 40, 41, 42, 46, 48. A second, architecturally independent detector is
therefore run and reconciled (`reconcile.py`); their disagreement is a more
reliable uncertainty signal than either model's self-reported confidence.

`transformers` is treated as optional: when it is absent `is_available()` returns
False and `detect_grid_tatr()` returns None, so the pipeline degrades to
morphology-only rather than failing. Neither ever raises.

⚠️ Pinned to `transformers<5`. Release 5.x validates model configs with strict
dataclasses and rejects this checkpoint's `dilation: null` outright, with no
override that survives `from_pretrained`.

Import-safe: torch and transformers are imported lazily inside the functions
(PRD §17.5).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence

import logging
from pathlib import Path

from sarmine.segment.rulings import GROUPING_GAP_PX, Grid, build_cells

logger = logging.getLogger(__name__)

TATR_MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"
DEFAULT_MODEL = TATR_MODEL_ID

# TATR's structure-recognition label set. Only rows and columns are needed to
# derive a cell lattice; spanning cells and headers are ignored for now.
ROW_LABELS = {"table row", "table projected row header", "table column header"}
COLUMN_LABELS = {"table column"}

DETECTION_THRESHOLD = 0.6

# The checkpoint's saved preprocessor config carries only `longest_edge`, which
# transformers rejects with "Size must contain 'height' and 'width' keys or
# 'shortest_edge' and 'longest_edge' keys". Supplying the pair explicitly is the
# fix; these are TATR's own structure-recognition values.
PROCESSOR_SIZE = {"shortest_edge": 800, "longest_edge": 1000}

Box = tuple[float, float, float, float]

# Loading the ~110 MB model costs seconds; the compound table spans ~28 pages.
_MODEL_CACHE: dict[str, tuple[object, object]] = {}


def grid_to_dict(grid: Grid) -> dict:
    return {
        "y_rulings": list(grid.y_rulings),
        "x_rulings": list(grid.x_rulings),
        "width": grid.width,
        "height": grid.height,
        "detector": grid.detector,
    }


def grid_from_dict(payload: dict) -> Grid:
    from sarmine.segment.rulings import cells_from_rulings

    y_rulings = [int(v) for v in payload["y_rulings"]]
    x_rulings = [int(v) for v in payload["x_rulings"]]
    return Grid(
        y_rulings=y_rulings,
        x_rulings=x_rulings,
        cells=cells_from_rulings(y_rulings, x_rulings),
        width=int(payload.get("width", 0)),
        height=int(payload.get("height", 0)),
        detector=payload.get("detector", "tatr"),
    )


def detect_grids_isolated(
    image_paths: Sequence[Path], *, timeout: float = 900.0
) -> dict[str, Grid]:
    """Run TATR over every page in ONE short-lived subprocess (PRD R17.1).

    A single TATR inference takes a process to ~1.1 GB — mostly attention
    activations, not weights — and `ru_maxrss` never falls, so freeing the model
    in-process cannot recover that peak. Running it out of process means the OS
    reclaims all of it on exit and the parent never imports torch at all, which
    is the same reasoning R17.1 already applies to tesseract and the OPSIN JVM.

    One subprocess for the whole batch, not one per page: the model load is the
    part worth amortizing across the ~28 table pages.

    Never raises. A missing second opinion is normal and must not abort a run.
    """
    paths = [Path(p) for p in image_paths]
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return {}

    job = json.dumps({"images": [str(p) for p in existing]})
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "sarmine.segment.tatr_worker"],
            input=job,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except Exception:
        return {}

    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}

    return {
        key: grid_from_dict(value) for key, value in payload.items() if value is not None
    }


def free() -> None:
    """Drop the cached detector (PRD R17.4).

    The cache is worth having across ~28 table pages, but it must not outlive
    the segmentation stage: holding it while MolScribe loads makes peak RSS the
    *sum* of the two models rather than the maximum, which measured 2895 MB
    against a 2400 MB budget (AC-9.3).
    """
    import gc

    _MODEL_CACHE.clear()
    gc.collect()


def is_available() -> bool:
    """True when the Table Transformer stack can actually be imported."""
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoImageProcessor,
            TableTransformerForObjectDetection,
        )
    except Exception:  # pragma: no cover - exercised only where deps exist
        return False
    return True


def _rulings_from_boxes(boxes: list[Box], axis: int, limit: int) -> list[int]:
    """Collapse predicted band edges along one axis into ruling coordinates.

    Adjacent bands nominally share a boundary but TATR's regressed edges land a
    few pixels apart, so near-coincident edges are merged with the same gap
    tolerance the morphological detector uses.
    """
    edges: list[int] = []
    for box in boxes:
        low = int(round(min(max(float(box[axis]), 0.0), float(limit))))
        high = int(round(min(max(float(box[axis + 2]), 0.0), float(limit))))
        if high > low:
            edges.extend((low, high))

    merged: list[int] = []
    for edge in sorted(edges):
        if merged and edge - merged[-1] <= GROUPING_GAP_PX:
            merged[-1] = (merged[-1] + edge) // 2
        else:
            merged.append(edge)
    return merged


def grid_from_boxes(rows: list[Box], columns: list[Box], *, width: int, height: int) -> Grid:
    """Derive a cell lattice by intersecting predicted rows with columns."""
    y_rulings = _rulings_from_boxes(rows, axis=1, limit=height)
    x_rulings = _rulings_from_boxes(columns, axis=0, limit=width)
    cells = build_cells(y_rulings, x_rulings)
    return Grid(
        n_rows=max(len(y_rulings) - 1, 0),
        n_cols=max(len(x_rulings) - 1, 0),
        cells=cells,
        y_rulings=y_rulings,
        x_rulings=x_rulings,
        width=width,
        height=height,
        detector="tatr",
    )


def detect_grid_tatr(
    image_path: Path | str,
    *,
    threshold: float = DETECTION_THRESHOLD,
    model_name: str = TATR_MODEL_ID,
) -> Grid | None:
    """Run Table Transformer on one page and return its grid, or None.

    Returns None — never raises — when `transformers` is missing, the weights
    cannot be fetched, the file is unreadable, or the model predicts nothing
    usable. The caller treats a missing second opinion as "morphology only".
    """
    path = Path(image_path)
    if not path.is_file():
        logger.debug("TATR: no such image %s", path)
        return None
    if not is_available():
        logger.info("TATR unavailable: `transformers` is not installed")
        return None

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        torch.set_num_threads(2)  # PRD R9.11

        image = Image.open(path).convert("RGB")
        if model_name not in _MODEL_CACHE:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = TableTransformerForObjectDetection.from_pretrained(model_name)
            model.eval()
            _MODEL_CACHE[model_name] = (processor, model)
        processor, model = _MODEL_CACHE[model_name]

        inputs = processor(images=image, return_tensors="pt", size=PROCESSOR_SIZE)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]])
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        rows: list[Box] = []
        columns: list[Box] = []
        for label_id, box in zip(results["labels"].tolist(), results["boxes"].tolist()):
            label = model.config.id2label[int(label_id)]
            if label in ROW_LABELS:
                rows.append(tuple(box))  # type: ignore[arg-type]
            elif label in COLUMN_LABELS:
                columns.append(tuple(box))  # type: ignore[arg-type]

        if not rows or not columns:
            logger.info("TATR predicted no usable rows/columns on %s", path.name)
            return None

        grid = grid_from_boxes(rows, columns, width=image.width, height=image.height)
        return grid if grid.cells else None
    except Exception as exc:  # noqa: BLE001 - a missing second opinion is not fatal
        logger.warning("TATR detection failed on %s: %s", path.name, exc)
        return None
