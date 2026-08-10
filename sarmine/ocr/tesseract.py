"""Region OCR (PRD §8.2, Plan Part 5.1).

Routing rule R8.1 is enforced by the caller, not here: a generic OCR engine must
never be pointed at a structure drawing. Spike A measured what happens when it is
— atom labels interleave into the IUPAC name and OPSIN parses 0 of 61 names.

Implements the §8.2 routing table: `--psm 6` for name cells (uniform block),
`--psm 7` plus a digit whitelist for compound-number cells, and word-level boxes
in source-image coordinates so every extracted field can carry `Provenance`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from sarmine.config import get_config

DIGITS = "0123456789"

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Word:
    """One OCR word. `bbox` is absolute in the SOURCE image (PRD §15.3)."""

    text: str
    conf: float
    bbox: BBox
    line_num: int


@dataclass(frozen=True)
class WordBox:
    text: str
    conf: float
    bbox: BBox


def _args(psm: int, whitelist: str | None, lang: str) -> list[str]:
    args = ["-l", lang, "--psm", str(psm)]
    if whitelist:
        args += ["-c", f"tessedit_char_whitelist={whitelist}"]
    return args


def _run(cmd: list[str], timeout: float) -> str:
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    raw = proc.stdout
    if isinstance(raw, str):
        return raw
    # Plan 5.1 — tesseract emits non-UTF-8 bytes on some crops; decoding strictly
    # raises UnicodeDecodeError mid-run.
    return raw.decode("utf-8", errors="replace")


@contextmanager
def _staged_input(path: Path, bbox: BBox | None, work_dir: Path | None) -> Iterator[Path]:
    """Yield the image tesseract should read, cropping to `bbox` when asked.

    PRD §17.5 / decisions.md S4 — crops must never be written to `/tmp`; tesseract
    cannot read them there under some sandboxes.
    """
    if bbox is None:
        yield path
        return

    owned = work_dir is None
    if work_dir is None:
        try:
            base = Path(tempfile.mkdtemp(prefix=".sarmine-ocr-", dir=path.parent))
        except OSError:
            base = Path(tempfile.mkdtemp(prefix="sarmine-ocr-"))
    else:
        base = Path(work_dir)
        base.mkdir(parents=True, exist_ok=True)

    crop_dir = Path(tempfile.mkdtemp(prefix="crop-", dir=base))
    crop_path = crop_dir / f"{path.stem}-{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"
    try:
        with Image.open(path) as im:
            im.crop(bbox).save(crop_path)
        yield crop_path
    finally:
        shutil.rmtree(crop_dir, ignore_errors=True)
        if owned:
            shutil.rmtree(base, ignore_errors=True)


@lru_cache(maxsize=1)
def tesseract_version() -> str:
    """`Provenance.extractor` value for every OCR-derived field."""
    cfg = get_config()
    out = _run([cfg.tesseract_bin, "--version"], timeout=30.0)
    first = out.splitlines()[0].strip() if out else ""
    number = first.replace("tesseract", "").strip() or "unknown"
    return f"tesseract@{number}"


def ocr_region(
    image_path: Path,
    bbox: BBox | None = None,
    *,
    psm: int = 6,
    whitelist: str | None = None,
    work_dir: Path | None = None,
    lang: str = "eng",
    timeout: float = 180.0,
) -> str:
    cfg = get_config()
    with _staged_input(Path(image_path), bbox, work_dir) as target:
        cmd = [cfg.tesseract_bin, str(target), "-", *_args(psm, whitelist, lang)]
        return _run(cmd, timeout)


def ocr_words(
    image_path: Path,
    bbox: BBox | None = None,
    *,
    psm: int = 6,
    whitelist: str | None = None,
    work_dir: Path | None = None,
    lang: str = "eng",
    timeout: float = 180.0,
    min_conf: float = 0.0,
) -> list[Word]:
    """Word boxes translated back into the source image's coordinate space."""
    cfg = get_config()
    with _staged_input(Path(image_path), bbox, work_dir) as target:
        cmd = [cfg.tesseract_bin, str(target), "-", *_args(psm, whitelist, lang), "tsv"]
        raw = _run(cmd, timeout)

    dx, dy = (bbox[0], bbox[1]) if bbox else (0, 0)
    words: list[Word] = []
    line_index: dict[tuple[int, int, int], int] = {}
    for row in _tsv_rows(raw):
        text = row["text"].strip()
        if not text:
            continue
        try:
            conf = float(row["conf"])
            left, top = int(row["left"]), int(row["top"])
            width, height = int(row["width"]), int(row["height"])
            key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        except (ValueError, KeyError):
            continue
        if conf < min_conf:
            continue
        # Tesseract restarts line_num inside every paragraph; provenance needs an
        # identifier that is unique across the whole region.
        line_num = line_index.setdefault(key, len(line_index) + 1)
        words.append(
            Word(
                text=text,
                conf=conf,
                bbox=(dx + left, dy + top, dx + left + width, dy + top + height),
                line_num=line_num,
            )
        )
    return words


def ocr_number_cell(
    image_path: Path,
    bbox: BBox | None = None,
    *,
    work_dir: Path | None = None,
    timeout: float = 60.0,
) -> int | None:
    """PRD §8.2 — digits only. Returns None when nothing readable came back.

    PRD R8.4 / EC-4: a compound number is flagged, never guessed, because an
    invented number corrupts the join.

    A Table 1 number cell is a small glyph adrift in a tall ruled box — measured
    on page 63, a 40x49 px digit inside a 190x510 px cell. Passing that region
    straight to tesseract returns punctuation noise at every psm, because the
    ruling lines dominate it. So the cell is inset past its borders, cropped to
    the remaining ink and upscaled before OCR.
    """
    prepared = _isolate_glyphs(image_path, bbox, work_dir=work_dir)
    if prepared is None:
        return None

    # Glyph isolation crops to the ink, which can split a two-digit number into
    # two blobs and lose one: measured on page 66, `11` read as `1`. Reading the
    # merely-inset cell as well keeps both digits together, and the longest
    # reading below arbitrates. A dropped digit is worse than an unread cell —
    # it points the row at another compound's activity data (R11.1).
    inset_readings: list[str] = []
    inset = _inset_bbox(image_path, bbox)
    if inset is not None:
        for psm in (7, 6):
            text = ocr_region(
                image_path, inset, psm=psm, whitelist=DIGITS, work_dir=work_dir, timeout=timeout
            )
            inset_readings.extend(re.findall(r"\d+", text))

    isolated_readings: list[str] = []

    # psm 8 (single word) and 13 (raw line) read isolated digits that psm 7
    # misses entirely; 10 (single character) covers the one-digit case. The modes
    # disagree on two-digit numbers — measured on the reference patent, compound
    # 41 reads as "4" under psm 8 but "41" under psm 10 — so all modes are tried
    # and the LONGEST reading wins rather than the first non-empty one. Dropping a
    # leading digit turns compound 41 into compound 4 and silently mis-joins a row.
    for psm in (8, 13, 10, 7):
        text = ocr_region(
            prepared, None, psm=psm, whitelist=DIGITS, work_dir=work_dir, timeout=timeout
        )
        isolated_readings.extend(re.findall(r"\d+", text))

    readings = isolated_readings + inset_readings
    if not readings:
        return None

    # Longest wins first, because a dropped leading digit mis-joins the row. At
    # equal length the isolated reading wins: upscaling a lone glyph is what makes
    # a single small digit legible, while the inset read of the same cell can
    # confuse it — measured, a `5` reading as `6`.
    def rank(value: str) -> tuple[int, int, int]:
        return (len(value), int(value in isolated_readings), readings.count(value))

    return int(max(readings, key=rank))


# Ruling lines sit within a few px of the cell edge; the morphological detector
# groups rulings with the same tolerance.
_CELL_INSET_PX = 10
_INK_PAD_PX = 14
_UPSCALE = 5
_MIN_INK_PX = 12


def _inset_bbox(image_path: Path, bbox: BBox | None) -> BBox | None:
    """The cell minus its ruling lines, in the source image's coordinates."""
    from PIL import Image

    if bbox is None:
        with Image.open(image_path) as handle:
            bbox = (0, 0, handle.width, handle.height)
    x0, y0, x1, y1 = bbox
    if x1 - x0 <= 2 * _CELL_INSET_PX or y1 - y0 <= 2 * _CELL_INSET_PX:
        return bbox
    return (x0 + _CELL_INSET_PX, y0 + _CELL_INSET_PX, x1 - _CELL_INSET_PX, y1 - _CELL_INSET_PX)


def _isolate_glyphs(
    image_path: Path, bbox: BBox | None, *, work_dir: Path | None
) -> Path | None:
    """Inset past the rulings, crop to the ink, upscale, and pad with white.

    Returns None when the cell holds no ink beyond its borders, which is how a
    genuinely blank number cell stays unread instead of being invented.
    """
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(image_path) as handle:
        image = handle.convert("L")
        cell = image.crop(bbox) if bbox else image.copy()

    inset = min(_CELL_INSET_PX, cell.width // 4, cell.height // 4)
    cell = cell.crop((inset, inset, cell.width - inset, cell.height - inset))
    if cell.width < 4 or cell.height < 4:
        return None

    ys, xs = np.where(np.array(cell) < 128)
    if len(xs) < _MIN_INK_PX:
        return None

    glyph = cell.crop(
        (
            max(int(xs.min()) - _INK_PAD_PX, 0),
            max(int(ys.min()) - _INK_PAD_PX, 0),
            min(int(xs.max()) + _INK_PAD_PX, cell.width),
            min(int(ys.max()) + _INK_PAD_PX, cell.height),
        )
    )
    glyph = glyph.resize((glyph.width * _UPSCALE, glyph.height * _UPSCALE), Image.LANCZOS)
    glyph = ImageOps.expand(glyph, border=40, fill=255)

    # PRD §17.5 — never /tmp; tesseract cannot read it under some sandboxes.
    out_dir = Path(work_dir) if work_dir else Path(image_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"_numcell_{abs(hash((str(image_path), bbox))) & 0xFFFFFFFF:08x}.png"
    glyph.save(out)
    return out


def ocr_image(
    path: Path,
    *,
    psm: int = 6,
    whitelist: str | None = None,
    lang: str = "eng",
    timeout: float = 180.0,
) -> str:
    return ocr_region(path, None, psm=psm, whitelist=whitelist, lang=lang, timeout=timeout)


def ocr_tsv(
    path: Path,
    *,
    psm: int = 6,
    whitelist: str | None = None,
    lang: str = "eng",
    timeout: float = 180.0,
    min_conf: float = 0.0,
) -> list[WordBox]:
    words = ocr_words(
        path,
        None,
        psm=psm,
        whitelist=whitelist,
        lang=lang,
        timeout=timeout,
        min_conf=min_conf,
    )
    return [WordBox(text=w.text, conf=w.conf, bbox=w.bbox) for w in words]


def ocr_number(path: Path, *, timeout: float = 60.0) -> int | None:
    return ocr_number_cell(path, None, timeout=timeout)


def _tsv_rows(raw: str) -> Iterator[dict[str, str]]:
    lines = raw.splitlines()
    if not lines:
        return
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        yield dict(zip(header, parts, strict=False))
