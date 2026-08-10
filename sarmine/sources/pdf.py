"""The fallback source path — poppler + tesseract over the raw PDF (PRD §7.2, R7.1).

This module is not a stub and never may be (PRD R7.1): the reference patent has
no text layer at all (223 characters across 223 pages, no embedded fonts — PRD
§3.1, EC-1), so every character it yields comes from OCR.

Implements PRD R7.4 (extract, do not re-render), the publication-number
extraction of PRD §7.2 step 1 / AC-1.1, and page rotation detect-then-verify
(PRD §7.4, R7.6–R7.8).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..config import get_config

# PRD R7.4 — a page raster is at least this wide/tall at 300 dpi; anything
# smaller that pdfimages emits is a logo or a figure, not the page itself.
_MIN_USABLE_PX = 1000

# A text layer counts as real only when this many non-whitespace characters per
# page come out of pdftotext. The reference patent yields zero (PRD §3.1).
_MIN_TEXT_CHARS_PER_PAGE = 8

_PAGES_RE = re.compile(r"^Pages:\s+(\d+)", re.M)

# PRD §7.2 step 1 — verbatim.
_PUBNUM_RE = re.compile(r"(?P<cc>[A-Z]{2})\s?(?P<num>[\d/ ]{6,15})\s?(?P<kind>[A-Z]\d?)")
_PUBNUM_SHAPE_RE = re.compile(r"^[A-Z]{2}\d{6,13}[A-Z]\d?$")
_PUBNUM_LABEL_RE = re.compile(r"\(\s*10\s*\)|publication\s+number|pub\.?\s*no\.?", re.I)

# The front page reads `A1` as `Al`; repair the kind code the same way names are
# repaired downstream (PRD §9.2 homoglyph repair).
_KIND_HOMOGLYPHS = {"l": "1", "I": "1", "i": "1", "|": "1", "!": "1", "O": "0", "o": "0"}

# Publishing authorities seen on a front page; used only to rank candidates.
_AUTHORITIES = frozenset(
    "WO US EP GB DE FR JP CN KR CA AU IN BR RU MX IL ZA SG NZ EA AP OA AR CL TW".split()
)

_OSD_ROTATE_RE = re.compile(r"^Rotate:\s*(\d+)", re.M)
_OSD_CONFIDENCE_RE = re.compile(r"^Orientation confidence:\s*([\d.]+)", re.M)

# PIL rotates counter-clockwise; OSD's `Rotate:` is the clockwise correction.
# Transposes are used rather than `rotate()` so a 1-bit CCITT raster survives
# de-rotation without being resampled (PRD R7.4).
_CLOCKWISE_TRANSPOSE = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}

_SCORE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9()\[\]'\u2019,.\-]*")
_SCORE_NUMBERED_SUBSTITUENT_RE = re.compile(r"\d+-\(")

# Fragments of real IUPAC names in this patent family. A page rotated the wrong
# way OCRs to character-reversed gibberish, which contains none of them — that
# asymmetry is the whole verification mechanism (PRD R7.7).
_CHEMICAL_FRAGMENTS = (
    "acetamid",
    "amino",
    "azetidin",
    "benzo",
    "carbox",
    "chloro",
    "dione",
    "dioxo",
    "ethyl",
    "fluoro",
    "hydroxy",
    "imidazol",
    "indazol",
    "indol",
    "isoindolin",
    "methoxy",
    "methyl",
    "morpholin",
    "nitril",
    "oxopiperidin",
    "phenoxy",
    "phenyl",
    "piperidin",
    "pyrazol",
    "pyridin",
    "pyrimidin",
    "pyrrolidin",
    "quinolin",
    "sulfon",
    "triazol",
    "yl)",
)

_DICTIONARY = frozenset(
    """
    about above according acid activity added addition administered after against all also
    amount among analysis and any appropriate are assay assays associated between both bound
    but cell cells characterized chemical claim claims comprising compound compounds
    composition compositions concentration containing control corresponding data date
    described description determined disclosed disclosure disease dose each effect effective
    embodiment embodiments entirety example examples excipient filing following for form
    formula from further group groups have herein human hydrogen include includes including
    incorporated independently induction inhibition inhibitor international into invention
    least level levels may measured method methods mixture more not number obtained
    one optionally other pharmaceutical preferred prepared present priority protein provided
    provides publication purified range reaction reference references reported respectively
    result results salt salts same selected shown solution some spectrum standard structure
    subject substituted such suitable table target the their therapeutic thereof these this
    those three time title treating treatment two under use used useful using values
    vitro vivo was were wherein which with within
    application applicant abstract agent designated inventors patent states published
    hbf ec50 ic50 dc50 pct wipo nmr esi mhz calcd found mass
    """.split()
)


@dataclass
class PdfPage:
    """One rasterized source page."""

    page_no: int  # 1-indexed PDF page
    path: Path
    width: int
    height: int


@dataclass
class RotationResult:
    """The outcome of detect-then-verify for one page (PRD §7.4, R7.8)."""

    page_no: int
    detected: int  # what OSD claimed
    applied: int  # what we actually used, after verification
    osd_confidence: float
    score: float  # re-OCR quality of the applied rotation
    alt_score: float
    uncertain: bool
    path: Path  # the de-rotated image on disk


def pdf_page_count(pdf: Path) -> int:
    """Page count via `pdfinfo`; 0 when the file is unreadable."""
    match = _PAGES_RE.search(_run([_poppler("pdfinfo"), str(pdf)]))
    return int(match.group(1)) if match else 0


def has_text_layer(pdf: Path) -> bool:
    """True only when text can actually be pulled out without OCR (PRD EC-1)."""
    fonts = _run([_poppler("pdffonts"), str(pdf)]).splitlines()
    # pdffonts prints a two-line header even when the font table is empty.
    if not [line for line in fonts[2:] if line.strip()]:
        return False
    text = _run([_poppler("pdftotext"), str(pdf), "-"])
    n_chars = len(re.sub(r"\s", "", text))
    return n_chars >= _MIN_TEXT_CHARS_PER_PAGE * max(1, pdf_page_count(pdf))


def extract_page_images(
    pdf: Path, out_dir: Path, *, first: int | None = None, last: int | None = None
) -> list[PdfPage]:
    """Rasterize pages `first..last` into `out_dir` as `p-{page:03d}-{n:03d}.png`.

    PRD R7.4 — `pdfimages -png` hands back the embedded bitmap untouched.
    Re-rendering a 1-bit CCITT G4 page with pdftoppm resamples and antialiases
    it, which measurably degrades OCR and OCSR on thin bond lines, so pdftoppm
    is only the fallback for pages that carry no usable embedded raster.
    Output always lands inside `out_dir`, never /tmp (PRD §17.5).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pages = pdf_page_count(pdf)
    first_page = 1 if first is None else max(1, first)
    last_page = n_pages if last is None else min(n_pages or last, last)

    pages: list[PdfPage] = []
    for page_no in range(first_page, last_page + 1):
        extracted = _pdfimages_page(pdf, out_dir, page_no)
        if not extracted:
            extracted = _pdftoppm_page(pdf, out_dir, page_no)
        pages.extend(extracted)
    return pages


def normalize_pubnum(raw: str) -> str:
    """PRD §7.2 step 1 — strip spaces and slashes, upper-case the rest."""
    return re.sub(r"[\s/,.\u2010-\u2015-]", "", raw).upper()


def publication_number_from_text(text: str) -> str | None:
    """Pick the publication number out of OCR'd front-page text.

    The front page also carries the PCT application number and the priority
    number, so candidates are ranked: a `(10) International Publication Number`
    label nearby beats a bare match, and a known publishing authority beats an
    unknown two-letter pair. Nothing is returned unless one of those holds —
    inventing a number is worse than reporting none (PRD EC-22).
    """
    best: str | None = None
    best_score = 0
    for match in _PUBNUM_RE.finditer(text or ""):
        if text[max(0, match.start() - 1) : match.start()] == "/":
            continue  # the tail of an application number such as PCT/US2023/078600
        candidate = normalize_pubnum(
            match.group("cc") + match.group("num") + _repair_kind(text, match)
        )
        if not _PUBNUM_SHAPE_RE.match(candidate):
            continue
        score = 0
        if _PUBNUM_LABEL_RE.search(text[max(0, match.start() - 240) : match.start()]):
            score += 2
        if match.group("cc") in _AUTHORITIES:
            score += 1
        if score > best_score:
            best, best_score = candidate, score
    return best


def publication_number_from_image(image_path: Path) -> str | None:
    """OCR one page raster and read the publication number off it (AC-1.1)."""
    return publication_number_from_text(ocr_image(image_path))


def extract_publication_number(pdf: Path, work_dir: Path) -> str | None:
    """PRD §7.2 step 1 — page 1 at 300 dpi, OCR, then the §7.2 regex."""
    pages = extract_page_images(pdf, Path(work_dir), first=1, last=1)
    if not pages:
        return None
    return publication_number_from_image(pages[0].path)


def ocr_image(image_path: Path, *, psm: int | None = None) -> str:
    """Tesseract to stdout, decoded with `errors="replace"` because tesseract
    emits non-UTF-8 bytes on some crops."""
    command = [get_config().tesseract_bin, str(image_path), "-"]
    if psm is not None:
        command += ["--psm", str(psm)]
    return _run(command)


def detect_rotation_osd(image_path: Path) -> tuple[int, float]:
    """PRD R7.6 — `tesseract <page> - --psm 0`, parsed.

    Returns `(0, 0.0)` when OSD refuses the page (too few characters), which is
    a claim of nothing, not a claim of "upright".
    """
    output = _run([get_config().tesseract_bin, str(image_path), "-", "--psm", "0"], stderr=True)
    rotate = _OSD_ROTATE_RE.search(output)
    confidence = _OSD_CONFIDENCE_RE.search(output)
    if rotate is None:
        return 0, 0.0
    return int(rotate.group(1)) % 360, float(confidence.group(1)) if confidence else 0.0


def score_ocr_quality(text: str) -> float:
    """Fraction of word-shaped tokens that are dictionary-like or chemical.

    Tokens with fewer than three letters (bare numbers, single-letter bin
    values, ruling-line noise) carry no orientation signal and are excluded
    from the denominator rather than counted against the page.
    """
    recognized = total = 0
    for token in _SCORE_TOKEN_RE.findall(text or ""):
        lowered = token.lower()
        letters = re.sub(r"[^a-z]", "", lowered)
        if len(letters) < 3:
            continue
        total += 1
        if (
            letters in _DICTIONARY
            or any(fragment in lowered for fragment in _CHEMICAL_FRAGMENTS)
            or _SCORE_NUMBERED_SUBSTITUENT_RE.search(lowered)
        ):
            recognized += 1
    return recognized / total if total else 0.0


def resolve_rotation(image_path: Path, out_dir: Path, *, page_no: int) -> RotationResult:
    """Detect a page's rotation, then verify it by re-OCR quality (PRD R7.7).

    The OSD confidence is deliberately ignored: on this patent's Table 1 pages
    the correct answer (`Rotate: 90`) comes back at confidence 13.62, low enough
    that any threshold would reject it. When the applied rotation OCRs badly the
    opposite rotation is tried and the better of the two wins; when neither
    scores, `uncertain` is set so the caller can raise `rotation_uncertain`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detected, confidence = detect_rotation_osd(image_path)
    applied = detected % 360
    path = _write_rotated(image_path, out_dir, page_no, applied)
    score = score_ocr_quality(ocr_image(path))
    alt_score = 0.0

    threshold = get_config().rotation_score_threshold
    if score < threshold:
        alternative = (applied + 180) % 360
        alt_path = _write_rotated(image_path, out_dir, page_no, alternative)
        alt_score = score_ocr_quality(ocr_image(alt_path))
        if alt_score > score:
            path.unlink(missing_ok=True)
            applied, path, score, alt_score = alternative, alt_path, alt_score, score
        else:
            alt_path.unlink(missing_ok=True)

    return RotationResult(
        page_no=page_no,
        detected=detected,
        applied=applied,
        osd_confidence=confidence,
        score=score,
        alt_score=alt_score,
        uncertain=max(score, alt_score) < threshold,
        path=path,
    )


def _write_rotated(source: Path, out_dir: Path, page_no: int, degrees: int) -> Path:
    destination = out_dir / f"p-{page_no:03d}-rot{degrees:03d}.png"
    with Image.open(source) as image:
        transpose = _CLOCKWISE_TRANSPOSE.get(degrees)
        rotated = image if transpose is None else image.transpose(transpose)
        rotated.save(destination)
    return destination


def _repair_kind(text: str, match: re.Match[str]) -> str:
    kind = match.group("kind")
    if len(kind) == 1:
        tail = text[match.end() : match.end() + 1]
        if tail in _KIND_HOMOGLYPHS:
            return kind + _KIND_HOMOGLYPHS[tail]
    return kind


def _pdfimages_page(pdf: Path, out_dir: Path, page_no: int) -> list[PdfPage]:
    root = out_dir / f"p-{page_no:03d}"
    _run(
        [
            _poppler("pdfimages"),
            # `-all` emits raw .ccitt + .params files for this PDF, which are not
            # loadable images; `-png` is the lossless container that works.
            "-png",
            "-f",
            str(page_no),
            "-l",
            str(page_no),
            str(pdf),
            str(root),
        ]
    )
    emitted = sorted(out_dir.glob(f"p-{page_no:03d}-*.png"))
    usable = [(path, _dimensions(path)) for path in emitted]
    usable = [(path, size) for path, size in usable if size is not None]
    if not any(min(size) >= _MIN_USABLE_PX for _, size in usable):
        for path in emitted:
            path.unlink(missing_ok=True)
        return []
    return [PdfPage(page_no, path, size[0], size[1]) for path, size in usable]


def _pdftoppm_page(pdf: Path, out_dir: Path, page_no: int) -> list[PdfPage]:
    root = out_dir / f"_ppm-{page_no:03d}"
    _run(
        [
            _poppler("pdftoppm"),
            "-png",
            "-r",
            "300",
            "-gray",
            "-f",
            str(page_no),
            "-l",
            str(page_no),
            str(pdf),
            str(root),
        ]
    )
    pages: list[PdfPage] = []
    for index, path in enumerate(sorted(out_dir.glob(f"_ppm-{page_no:03d}-*.png"))):
        size = _dimensions(path)
        if size is None:
            path.unlink(missing_ok=True)
            continue
        final = path.replace(out_dir / f"p-{page_no:03d}-{index:03d}.png")
        pages.append(PdfPage(page_no, final, size[0], size[1]))
    return pages


def _dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _poppler(tool: str) -> str:
    return get_config().poppler(tool)


def _run(command: list[str], *, stderr: bool = False) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except (OSError, ValueError):
        return ""
    output = completed.stdout.decode("utf-8", errors="replace")
    if stderr:
        output += completed.stderr.decode("utf-8", errors="replace")
    return output
