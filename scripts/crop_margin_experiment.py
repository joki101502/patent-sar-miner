"""Does the cell ruling line corrupt OCSR? Measured, not assumed (PRD R9.7).

Sweeps an inset around each structure cell and scores agreement against the
OPSIN-derived InChIKey for the same row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MARGINS = (0, 6, 12, 20, 30)


def main() -> int:
    from PIL import Image

    from sarmine.pipeline import _best_grid
    from sarmine.segment.roles import assign_column_roles
    from sarmine.segment.rulings import Grid
    from sarmine.sources.pdf import extract_page_images, resolve_rotation
    from sarmine.structure.molscribe import MolScribeRunner

    compounds = json.loads((REPO / "artifacts/WO2024097932A1/fullrun/compounds.json").read_text())
    by_number = {c["compound_number"]: c for c in compounds if c.get("compound_number")}

    work = REPO / "scratch_margin"
    work.mkdir(exist_ok=True)
    targets = [3, 4, 7, 13, 16, 20, 5, 2]

    jobs: list[tuple[int, int, Path]] = []
    for number in targets:
        compound = by_number.get(number)
        if not compound or not compound["provenance"]:
            continue
        page_no = compound["provenance"][0]["page_no"]
        page = extract_page_images(REPO / "data/patents/WO2024097932A1.pdf", work, first=page_no, last=page_no)
        if not page:
            continue
        rotated = resolve_rotation(page[0].path, work / "rot", page_no=page_no)
        grid: Grid | None = _best_grid(rotated.path)
        if grid is None:
            continue
        from sarmine.pipeline import _column_text

        roles = assign_column_roles(grid, {c: _column_text(rotated.path, grid, c) for c in range(grid.n_cols)})
        struct_cols = [c for c, r in roles.items() if r == "structure"]
        num_cols = [c for c, r in roles.items() if r == "number"]
        if not struct_cols or not num_cols:
            continue

        from sarmine.pipeline import _cell_text

        row_index = None
        for row in range(grid.n_rows):
            cell = grid.cell(row, num_cols[0])
            if cell and _cell_text(rotated.path, cell.bbox).strip() == str(number):
                row_index = row
                break
        if row_index is None:
            continue

        cell = grid.cell(row_index, struct_cols[0])
        if cell is None:
            continue
        x0, y0, x1, y1 = cell.bbox
        with Image.open(rotated.path) as image:
            for margin in MARGINS:
                box = (x0 + margin, y0 + margin, x1 - margin, y1 - margin)
                out = work / f"c{number}_m{margin}.png"
                image.crop(box).save(out)
                jobs.append((number, margin, out))

    runner = MolScribeRunner()
    results = runner.predict([p for _, _, p in jobs])
    runner.free()

    scores: dict[int, int] = {m: 0 for m in MARGINS}
    print(f"{'cmpd':>5} {'opsin key':>28} " + " ".join(f"{m:>6}" for m in MARGINS))
    for number in targets:
        compound = by_number.get(number)
        if not compound:
            continue
        expected = compound.get("inchikey_from_name")
        row = []
        for margin in MARGINS:
            match = next(
                (r for (n, m, _), r in zip(jobs, results) if n == number and m == margin), None
            )
            hit = bool(match and expected and match.inchikey == expected)
            scores[margin] += hit
            row.append("  MATCH" if hit else "      .")
        print(f"{number:>5} {str(expected):>28} " + " ".join(row))
    print("\ntotals by margin:", scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
