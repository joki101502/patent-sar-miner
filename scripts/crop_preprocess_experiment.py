"""Does upsampling the bitonal crop help OCSR? Measured (PRD §3.1, decisions.md).

The PDF's pages are 1-bit CCITT G4: no antialiasing on thin bond lines, which is
the exact failure mode OCSR models are sensitive to. The recorded expectation is
that preprocessing moves accuracy more than model choice, so this measures it
against the OPSIN-derived key for the same row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MARGIN = 8


def variants(image, out_dir: Path, stem: str) -> dict[str, Path]:
    from PIL import Image, ImageFilter

    gray = image.convert("L")
    width, height = gray.size
    made: dict[str, Path] = {}

    def save(name: str, img) -> None:
        path = out_dir / f"{stem}_{name}.png"
        img.save(path)
        made[name] = path

    save("base", gray)
    for scale in (2, 3):
        save(f"up{scale}", gray.resize((width * scale, height * scale), Image.LANCZOS))
    up2 = gray.resize((width * 2, height * 2), Image.LANCZOS)
    save("up2_blur", up2.filter(ImageFilter.GaussianBlur(radius=1.0)))
    save("up2_blur05", up2.filter(ImageFilter.GaussianBlur(radius=0.5)))
    save("blur_up2", gray.filter(ImageFilter.GaussianBlur(radius=0.5)).resize(
        (width * 2, height * 2), Image.LANCZOS
    ))
    return made


def main() -> int:
    from PIL import Image

    from sarmine.pipeline import _best_grid, _cell_text, _column_text
    from sarmine.segment.roles import assign_column_roles
    from sarmine.sources.pdf import extract_page_images, resolve_rotation
    from sarmine.structure.molscribe import MolScribeRunner

    compounds = json.loads((REPO / "artifacts/WO2024097932A1/fullrun/compounds.json").read_text())
    by_number = {c["compound_number"]: c for c in compounds if c.get("compound_number")}
    targets = [n for n in sorted(by_number) if by_number[n].get("inchikey_from_name")][:16]

    work = REPO / "scratch_pre"
    work.mkdir(exist_ok=True)

    jobs: list[tuple[int, str, Path]] = []
    for number in targets:
        compound = by_number[number]
        if not compound["provenance"]:
            continue
        page_no = compound["provenance"][0]["page_no"]
        pages = extract_page_images(REPO / "data/patents/WO2024097932A1.pdf", work, first=page_no, last=page_no)
        if not pages:
            continue
        rotated = resolve_rotation(pages[0].path, work / "rot", page_no=page_no)
        grid = _best_grid(rotated.path)
        if grid is None:
            continue
        roles = assign_column_roles(
            grid, {c: _column_text(rotated.path, grid, c) for c in range(grid.n_cols)}
        )
        struct = [c for c, r in roles.items() if r == "structure"]
        nums = [c for c, r in roles.items() if r == "number"]
        if not struct or not nums:
            continue
        row_index = next(
            (
                r
                for r in range(grid.n_rows)
                if grid.cell(r, nums[0])
                and _cell_text(rotated.path, grid.cell(r, nums[0]).bbox).strip() == str(number)
            ),
            None,
        )
        if row_index is None:
            continue
        cell = grid.cell(row_index, struct[0])
        if cell is None:
            continue
        x0, y0, x1, y1 = cell.bbox
        box = (x0 + MARGIN, y0 + MARGIN, x1 - MARGIN, y1 - MARGIN)
        with Image.open(rotated.path) as image:
            for name, path in variants(image.crop(box), work, f"c{number}").items():
                jobs.append((number, name, path))

    runner = MolScribeRunner()
    results = runner.predict([p for _, _, p in jobs])
    runner.free()

    names = sorted({n for _, n, _ in jobs})
    totals = {n: 0 for n in names}
    print(f"{'cmpd':>5} " + " ".join(f"{n:>10}" for n in names))
    for number in targets:
        expected = by_number[number].get("inchikey_from_name")
        row = []
        for name in names:
            match = next(
                (r for (nn, vn, _), r in zip(jobs, results) if nn == number and vn == name), None
            )
            hit = bool(match and expected and match.inchikey == expected)
            totals[name] += hit
            row.append(f"{'MATCH' if hit else '.':>10}")
        print(f"{number:>5} " + " ".join(row))
    print(f"\nof {len(targets)} compounds:", totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
