"""Review-item rendering (PRD R13.3, R13.7, AC-7.2, AC-8.2, Plan Part 10.2).

Bounding-box overlays are composited server-side with PIL — requiring a JS
canvas is what would make Streamlit unviable (R13.7).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

_LABEL_PADDING = 3


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=16)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    origin: tuple[int, int] | None,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Translate page-space coordinates into crop space and clamp to the image.

    A bbox drifting outside the crop means an upstream mis-registration; the
    reviewer still needs to see the crop, so clamp instead of raising.
    """
    dx, dy = origin or (0, 0)
    x0, y0, x1, y1 = (bbox[0] - dx, bbox[1] - dy, bbox[2] - dx, bbox[3] - dy)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    width, height = size
    x0 = max(0, min(x0, width - 1))
    x1 = max(x0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0, min(y1, height - 1))
    return x0, y0, x1, y1


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    anchor: tuple[int, int],
    color: str,
    image_size: tuple[int, int],
) -> None:
    font = _label_font()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    box_w = text_w + 2 * _LABEL_PADDING
    box_h = text_h + 2 * _LABEL_PADDING

    x = max(0, min(anchor[0], image_size[0] - box_w))
    y = anchor[1] - box_h
    if y < 0:
        y = min(anchor[1], max(0, image_size[1] - box_h))

    draw.rectangle((x, y, x + box_w, y + box_h), fill=color)
    draw.text((x + _LABEL_PADDING - left, y + _LABEL_PADDING - top), label, fill="white", font=font)


def render_crop_with_bbox(
    crop_path: Path,
    bbox: tuple[int, int, int, int] | None,
    out_path: Path,
    *,
    label: str | None = None,
    crop_origin: tuple[int, int] | None = None,
    color: str = "#d62728",
    width: int = 4,
) -> Path:
    """Draw `bbox` (page pixel space, PRD §15.3) onto the crop and save it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(crop_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    if bbox is not None:
        x0, y0, x1, y1 = _clamp_bbox(bbox, crop_origin, image.size)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        anchor = (x0, y0)
    else:
        anchor = (0, 0)

    if label:
        _draw_label(draw, label, anchor, color, image.size)

    image.save(out_path)
    return out_path


def _mol(smiles: str | None) -> Chem.Mol | None:
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def render_structure_svg(
    smiles: str | None, out_path: Path, *, size: tuple[int, int] = (350, 300)
) -> Path | None:
    """Render `smiles` to SVG. A failed extraction renders as nothing, not an error."""
    mol = _mol(smiles)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(drawer.GetDrawingText(), "utf-8")
    return out_path


def render_structure_png(
    smiles: str | None, out_path: Path, *, size: tuple[int, int] = (350, 300)
) -> Path | None:
    """Render `smiles` to PNG. A failed extraction renders as nothing, not an error."""
    mol = _mol(smiles)
    if mol is None:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cairo = getattr(rdMolDraw2D, "MolDraw2DCairo", None)
    if cairo is not None:
        drawer = cairo(*size)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        out_path.write_bytes(drawer.GetDrawingText())
        return out_path

    from rdkit.Chem import Draw  # Cairo-less builds fall back to the PIL renderer

    Draw.MolToImage(mol, size=size).save(out_path)
    return out_path
