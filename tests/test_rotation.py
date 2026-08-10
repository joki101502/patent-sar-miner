"""Tests for Part 3 — page rotation detect + verify (PRD §7.4, R7.6–R7.8, AC-2.1, EC-2).

The whole point of R7.7 is that the OSD confidence is untrustworthy: on this
patent's Table 1 pages the *correct* answer comes back at confidence ~13, so the
decision has to be made by re-OCR quality instead. These tests pin that
mechanism against the real rotated fixture pages. No network, no models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sarmine.config import get_config
from sarmine.sources import pdf as pdfsrc

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PAGES = FIXTURES / "pages"

ROTATED_TABLE1_PAGE = PAGES / "p-063-000.png"  # PRD §3.3 — Table 1, rotated 90
UPRIGHT_PAGE = PAGES / "p-187-000.png"  # PRD §3.4 — Table 2 continuation, upright

CHEMICAL_TOKENS = (
    "dioxopiperidin",
    "isoindoline",
    "dione",
    "amino",
    "methyl",
    "indazol",
    "imidazol",
)

# Real tesseract output for p-063 rotated the right way (90° clockwise)...
DEROTATED_TEXT = """co

2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-
4-methoxypheny])-1-methyl-1H-

benzo[d]imidazol-6-yl)amino)isoindoline-
1,3-dione

HN
O
0
N

6) 2-(2,6-dioxopiperidin-3-yl)-4-((5-(2-fluoro-
6-methylphenoxy)-1-methyl-1H-indazol-4-
yl)amino)isoindoline-1,3-dione

NH F
"""

# ...and for the same page rotated the wrong way: character-reversed gibberish.
REVERSED_TEXT = """PCT/US2023/078600

WO 2024/097932

4 HN
QUOIp-\u00a2 \u2018[-oUl[OpuTOs(OUuTUTR({A
-p-JOzepul-H [-[Ayjow- [-(AxousydAyjow-9 9
\u201cosony-Z)-\u00a2))-b-(1&\u00a2-ulpLiadidoxorp-9\u00b0Z)-z O
N
O
O
N
\\o
NH

uOIp-\u00a2"T
-ouTjopuros1(ourwme([A-9-ozeprunt[p ]ozusaq

-HI-[Ayjow-[-({AusydAxoyjowl-p
\u201cOONIJ-\u20ac)-S))-p-(IA-\u20ac-UlpHadidoxolp-9\u00b0Z)-Z
"""

ENGLISH_PROSE = (
    "A number of references have been cited, the disclosures of which are "
    "incorporated herein by reference in their entirety."
)


def _rotate(src: Path, dst: Path, degrees_clockwise: int) -> Path:
    """Lossless 90° multiples, so the test fixture is not resampled either."""
    transpose = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }[degrees_clockwise]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.transpose(transpose).save(dst)
    return dst


# --------------------------------------------------------------------------
# score_ocr_quality — the mechanism R7.7 rests on (pure strings, no OCR)
# --------------------------------------------------------------------------


def test_score_ocr_quality_separates_chemistry_from_reversed_gibberish() -> None:
    threshold = get_config().rotation_score_threshold
    good = pdfsrc.score_ocr_quality(DEROTATED_TEXT)
    bad = pdfsrc.score_ocr_quality(REVERSED_TEXT)
    assert good >= threshold
    assert bad < threshold
    assert good > 3 * bad


def test_score_ocr_quality_recognizes_plain_english_prose() -> None:
    assert pdfsrc.score_ocr_quality(ENGLISH_PROSE) >= get_config().rotation_score_threshold


def test_score_ocr_quality_of_nothing_is_zero() -> None:
    assert pdfsrc.score_ocr_quality("") == 0.0
    assert pdfsrc.score_ocr_quality("\n\n  \n") == 0.0
    assert pdfsrc.score_ocr_quality("| { } 4 7 = ~") == 0.0


def test_score_ocr_quality_is_a_fraction() -> None:
    for text in (DEROTATED_TEXT, REVERSED_TEXT, ENGLISH_PROSE, ""):
        assert 0.0 <= pdfsrc.score_ocr_quality(text) <= 1.0


# --------------------------------------------------------------------------
# detect_rotation_osd  (PRD R7.6, R7.7)
# --------------------------------------------------------------------------


def test_detect_rotation_osd_reports_90_at_low_confidence() -> None:
    """PRD R7.7 / EC-2 — the correct answer arrives with a rejectable confidence."""
    detected, confidence = pdfsrc.detect_rotation_osd(ROTATED_TABLE1_PAGE)
    assert detected == 90
    assert confidence < 20.0  # the PRD measured 13.62; any sane threshold rejects it


def test_detect_rotation_osd_reports_zero_for_an_upright_page() -> None:
    detected, _ = pdfsrc.detect_rotation_osd(UPRIGHT_PAGE)
    assert detected == 0


def test_detect_rotation_osd_survives_an_image_osd_cannot_read(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("L", (900, 900), 255).save(blank)
    assert pdfsrc.detect_rotation_osd(blank) == (0, 0.0)


# --------------------------------------------------------------------------
# resolve_rotation  (PRD AC-2.1, EC-2, R7.8)
# --------------------------------------------------------------------------


def test_resolve_rotation_corrects_a_table_one_page(tmp_path: Path) -> None:
    """PRD AC-2.1 — corrected Table 1 pages OCR to recognizable chemical tokens."""
    result = pdfsrc.resolve_rotation(ROTATED_TABLE1_PAGE, tmp_path, page_no=63)

    assert result.page_no == 63
    assert result.detected == 90
    assert result.applied == 90
    assert result.uncertain is False
    assert result.score >= get_config().rotation_score_threshold
    assert result.path.is_file()
    assert tmp_path in result.path.parents  # PRD §17.5 — never /tmp

    text = pdfsrc.ocr_image(result.path).lower()
    found = [token for token in CHEMICAL_TOKENS if token in text]
    assert len(found) >= 3, f"only found {found} in {text[:400]!r}"


def test_resolve_rotation_keeps_the_bitonal_raster_bitonal(tmp_path: Path) -> None:
    """PRD R7.4 — de-rotation must transpose, never resample."""
    with Image.open(ROTATED_TABLE1_PAGE) as original:
        assert original.mode == "1"
        width, height = original.size
    result = pdfsrc.resolve_rotation(ROTATED_TABLE1_PAGE, tmp_path, page_no=63)
    with Image.open(result.path) as rotated:
        assert rotated.mode == "1"
        assert rotated.size == (height, width)


def test_resolve_rotation_leaves_an_upright_page_upright(tmp_path: Path) -> None:
    result = pdfsrc.resolve_rotation(UPRIGHT_PAGE, tmp_path, page_no=187)
    assert result.applied == 0
    assert result.uncertain is False
    assert "incorporated herein by reference" in pdfsrc.ocr_image(result.path).lower()


def test_resolve_rotation_recovers_a_deliberately_mis_rotated_page(tmp_path: Path) -> None:
    """PRD EC-2 — a page rotated the wrong way is detected and put back."""
    mis_rotated = _rotate(UPRIGHT_PAGE, tmp_path / "mis-rotated.png", 270)
    result = pdfsrc.resolve_rotation(mis_rotated, tmp_path / "out", page_no=187)

    assert result.applied == 90
    assert result.uncertain is False
    assert "incorporated herein by reference" in pdfsrc.ocr_image(result.path).lower()


def test_resolve_rotation_overrides_a_wrong_osd_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PRD R7.7 — the re-OCR score, not the OSD number, decides.

    OSD is forced to the opposite (wrong) angle at a *high* confidence; the
    verification step must still land on 90.
    """
    monkeypatch.setattr(pdfsrc, "detect_rotation_osd", lambda path: (270, 99.0))
    result = pdfsrc.resolve_rotation(ROTATED_TABLE1_PAGE, tmp_path, page_no=63)

    assert result.detected == 270
    assert result.applied == 90
    assert result.score > result.alt_score
    assert result.uncertain is False
    text = pdfsrc.ocr_image(result.path).lower()
    assert sum(token in text for token in CHEMICAL_TOKENS) >= 3


def test_resolve_rotation_flags_uncertain_when_both_candidates_are_poor(
    tmp_path: Path,
) -> None:
    """PRD R7.8 / §13.2 — the caller turns this into a `rotation_uncertain` anomaly."""
    blank = tmp_path / "blank.png"
    Image.new("L", (1200, 1600), 255).save(blank)
    result = pdfsrc.resolve_rotation(blank, tmp_path / "out", page_no=7)

    assert result.uncertain is True
    assert result.score == 0.0
    assert result.path.is_file()


def test_resolve_rotation_writes_only_the_applied_rotation(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = pdfsrc.resolve_rotation(ROTATED_TABLE1_PAGE, out_dir, page_no=63)
    assert sorted(path.name for path in out_dir.glob("*.png")) == [result.path.name]
