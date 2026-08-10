"""Out-of-process TATR runner (PRD R17.1, AC-9.3).

Reads `{"images": [path, ...]}` on stdin and writes `{path: grid|null}` on
stdout. Exists so that torch, transformers and TATR's ~1.1 GB of inference
activations live in a process that exits, instead of adding themselves to the
pipeline's peak RSS for the rest of the run.

Run as `python -m sarmine.segment.tatr_worker`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        job = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        json.dump({}, sys.stdout)
        return 0

    from sarmine.segment.tatr import detect_grid_tatr, grid_to_dict, is_available

    results: dict[str, dict | None] = {}
    if is_available():
        for raw in job.get("images", []):
            path = Path(raw)
            try:
                grid = detect_grid_tatr(path)
            except Exception:
                grid = None
            results[str(path)] = grid_to_dict(grid) if grid is not None else None

    json.dump(results, sys.stdout)
    return 0


if __name__ == "__main__":  # PRD R9.10 — every entry point stays guarded
    sys.exit(main())
