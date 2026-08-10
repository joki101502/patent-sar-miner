"""Is the structured source's 8-bit raster worth more to OCSR than the PDF's 1-bit?

PRD §3.2 claims the patentimages rasters are "materially better input" than the
PDF's 1-bit bitonal pages. This measures that claim on the same structure cells,
scoring each against the OPSIN-derived InChIKey for the same row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MARGIN = 8
IMAGES = REPO / "artifacts/_staging/DEBUG2/source/images"


def cells_for(page_path: Path, work: Path, page_no: int):
    from sarmine.pipeline import _best_grid, _cell_text, _column_text
    from sarmine.segment.roles import assign_column_roles
    from sarmine.sources.pdf import resolve_rotation

    rotated = resolve_rotation(page_path, work / f"rot{page_no}", page_no=page_no)
    grid = _best_grid(rotated.path)
    if grid is None:
        return None, {}
    roles = assign_column_roles(
        grid, {c: _column_text(rotated.path, grid, c) for c in range(grid.n_cols)}
    )
    struct = [c for c, r in roles.items() if r == "structure"]
    nums = [c for c, r in roles.items() if r == "number"]
    if not struct or not nums:
        return None, {}

    found = {}
    for row in range(grid.n_rows):
        num_cell = grid.cell(row, nums[0])
        struct_cell = grid.cell(row, struct[0])
        if num_cell is None or struct_cell is None:
            continue
        text = _cell_text(rotated.path, num_cell.bbox).strip()
        if text.isdigit():
            found[int(text)] = struct_cell.bbox
    return rotated.path, found


def main() -> int:
    from PIL import Image

    from sarmine.sources.pdf import extract_page_images
    from sarmine.structure.molscribe import MolScribeRunner

    compounds = json.loads(
        (REPO / "artifacts/WO2024097932A1/fullrun_structured/compounds.json").read_text()
    )
    expected = {
        c["compound_number"]: c["inchikey_from_name"]
        for c in compounds
        if c.get("compound_number") and c.get("inchikey_from_name")
    }

    work = REPO / "scratch_gray"
    work.mkdir(exist_ok=True)
    jobs: list[tuple[int, str, Path]] = []

    for page_no in range(61, 76):
        grayscale = IMAGES / f"imgf{page_no:06d}_0001.png"
        pdf_pages = extract_page_images(
            REPO / "data/patents/WO2024097932A1.pdf", work / "pdf", first=page_no, last=page_no
        )
        sources = {"bitonal": pdf_pages[0].path if pdf_pages else None}
        if grayscale.is_file() and len(list(IMAGES.glob(f"imgf{page_no:06d}_*.png"))) == 1:
            sources["grayscale"] = grayscale

        for kind, path in sources.items():
            if path is None:
                continue
            rotated, cells = cells_for(path, work / kind, page_no)
            if rotated is None:
                continue
            with Image.open(rotated) as image:
                for number, bbox in cells.items():
                    if number not in expected:
                        continue
                    x0, y0, x1, y1 = bbox
                    out = work / f"{kind}_c{number}.png"
                    image.crop((x0 + MARGIN, y0 + MARGIN, x1 - MARGIN, y1 - MARGIN)).save(out)
                    jobs.append((number, kind, out))

    runner = MolScribeRunner()
    results = runner.predict([p for _, _, p in jobs])
    runner.free()

    hits: dict[str, int] = {}
    seen: dict[str, set[int]] = {}
    for (number, kind, _), result in zip(jobs, results):
        seen.setdefault(kind, set()).add(number)
        if result.inchikey and result.inchikey == expected.get(number):
            hits[kind] = hits.get(kind, 0) + 1

    both = set.intersection(*seen.values()) if len(seen) > 1 else set()
    print("compounds attempted per source:", {k: len(v) for k, v in seen.items()})
    print("exact InChIKey agreement:", hits)
    if both:
        paired = {
            kind: sum(
                1
                for (n, k, _), r in zip(jobs, results)
                if k == kind and n in both and r.inchikey == expected.get(n)
            )
            for kind in seen
        }
        print(f"on the {len(both)} compounds both sources reached:", paired)
    return 0


if __name__ == "__main__":
    sys.exit(main())
