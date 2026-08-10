"""Cell crops to disk, with provenance (PRD §15.3, §15.5, R17.3, Plan 4.6).

**Write crops to disk and carry only paths.** 223 pages of 2480×3508 grayscale
held in memory is ~1.9 GB, against a 2.7 GB deploy budget (PRD R17.3).

⚠️ Write crops inside the working/artifact directory, never `/tmp` — Tesseract
cannot read `/tmp` files under some sandboxes (PRD §17.5). This module does not
police the destination, because pytest's own temp directories are legitimate;
callers in the pipeline must pass a path inside the artifact bundle.

Import-safe: no side effects at import.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from sarmine.artifacts.schema import Provenance, SourceMode

CROP_FILENAME = "p{page:03d}_{kind}_{idx}.png"


def crop_filename(page_no: int, kind: str, idx: int) -> str:
    """The PRD §15.5 crop filename: `p{page:03d}_{kind}_{idx}.png`."""
    return CROP_FILENAME.format(page=page_no, kind=kind, idx=idx)


def write_crop(
    image_path: Path | str,
    bbox: tuple[int, int, int, int],
    out_dir: Path | str,
    *,
    page_no: int,
    kind: str,
    idx: int,
    source: SourceMode,
    extractor: str,
    rotation_applied: int = 0,
    pad: int = 0,
) -> Provenance:
    """Write one cell crop and return its fully populated `Provenance`.

    `bbox` is `(x0, y0, x1, y1)` in the source page's pixel space; it is padded
    by `pad` and clamped to the page. The returned `crop_path` is relative to
    the artifact bundle root — i.e. `crops/p063_name_0.png` when `out_dir` is
    the bundle's `crops/` directory.
    """
    source_path = Path(image_path)
    destination = Path(out_dir)

    with Image.open(source_path) as page:
        raster_width, raster_height = page.size

        x0, y0, x1, y1 = (int(v) for v in bbox)
        x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1 = min(x1 + pad, raster_width)
        y1 = min(y1 + pad, raster_height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"degenerate crop bbox {bbox} (padded to {(x0, y0, x1, y1)}) on a "
                f"{raster_width}x{raster_height} page"
            )

        destination.mkdir(parents=True, exist_ok=True)
        filename = crop_filename(page_no, kind, idx)
        page.crop((x0, y0, x1, y1)).save(destination / filename, format="PNG")

    return Provenance(
        page_no=page_no,
        bbox=(x0, y0, x1, y1),
        raster_width=raster_width,
        raster_height=raster_height,
        crop_path=f"{destination.name}/{filename}",
        source=source,
        extractor=extractor,
        rotation_applied=rotation_applied,
    )
