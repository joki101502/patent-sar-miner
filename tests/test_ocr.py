"""Region OCR tests (Plan 5.1, PRD §8.2 routing, §17.5 pitfalls).

Covers AC-2.3 (name-cell OCR carries no atom labels from the neighbouring
structure cell), EC-4 / R8.4 (a compound number is never guessed) and the
sandbox pitfall from decisions.md S4 (crops must not be written to `/tmp`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from sarmine.ocr.tesseract import (
    Word,
    ocr_number_cell,
    ocr_region,
    ocr_words,
    tesseract_version,
)

FIXTURES = Path(__file__).parent / "fixtures"

# The fixture page is stored 90°-rotated (EC-2); tests de-rotate it first.
PAGE_63 = FIXTURES / "imgf000063_0001.png"
DEROTATE_DEGREES = 270

# Cells of the first data row of Table 1 page 63 — compound 5 (AC-3.4).
NAME_BBOX = (1880, 270, 2800, 530)
NUMBER_BBOX = (120, 350, 260, 440)


@pytest.fixture(scope="session")
def upright_page(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pages") / "p-063-upright.png"
    Image.open(PAGE_63).rotate(DEROTATE_DEGREES, expand=True).save(out)
    return out


def test_tesseract_version_is_reported_for_provenance() -> None:
    version = tesseract_version()
    assert version.startswith("tesseract@")
    assert version.split("@", 1)[1][0].isdigit()


def test_ocr_region_reads_a_name_cell(upright_page: Path, tmp_path: Path) -> None:
    text = ocr_region(upright_page, NAME_BBOX, psm=6, work_dir=tmp_path)
    assert "dioxopiperidin" in text
    assert "isoindoline" in text


def test_name_cell_text_has_no_atom_labels_from_the_structure_cell(
    upright_page: Path, tmp_path: Path
) -> None:
    # AC-2.3 — Spike A's contamination (`re)fe)`, bare `NN`/`HN`) must be absent.
    text = ocr_region(upright_page, NAME_BBOX, psm=6, work_dir=tmp_path)
    for label in ("HN", "NN", "re)fe)"):
        assert label not in text


def test_ocr_region_without_a_bbox_reads_the_whole_image(
    upright_page: Path, tmp_path: Path
) -> None:
    text = ocr_region(upright_page, None, psm=6, work_dir=tmp_path)
    assert "isoindoline" in text


def test_ocr_words_returns_source_image_coordinates(
    upright_page: Path, tmp_path: Path
) -> None:
    words = ocr_words(upright_page, NAME_BBOX, psm=6, work_dir=tmp_path)
    assert words
    assert all(isinstance(w, Word) for w in words)
    left, top, right, bottom = NAME_BBOX
    for w in words:
        x0, y0, x1, y1 = w.bbox
        assert left <= x0 <= x1 <= right
        assert top <= y0 <= y1 <= bottom
        assert 0.0 <= w.conf <= 100.0
        assert w.text.strip() == w.text and w.text
    assert any("isoindoline" in w.text for w in words)
    assert len({w.line_num for w in words}) > 1


def test_ocr_words_on_an_empty_region_returns_no_words(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("L", (400, 120), color=255).save(blank)
    assert ocr_words(blank, None, work_dir=tmp_path) == []


def test_ocr_number_cell_reads_the_compound_number(
    upright_page: Path, tmp_path: Path
) -> None:
    assert ocr_number_cell(upright_page, NUMBER_BBOX, work_dir=tmp_path) == 5


def test_ocr_number_cell_never_guesses_an_unreadable_number(tmp_path: Path) -> None:
    # PRD R8.4 / EC-4 — an invented compound number corrupts the join.
    blank = tmp_path / "blank-number.png"
    Image.new("L", (120, 90), color=255).save(blank)
    assert ocr_number_cell(blank, None, work_dir=tmp_path) is None


def test_temporary_crops_are_written_inside_the_work_dir(
    upright_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PRD §17.5 / decisions.md S4 — tesseract cannot read `/tmp` under some sandboxes.
    seen: list[str] = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(cmd[1])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr("sarmine.ocr.tesseract.subprocess.run", spy)
    ocr_region(upright_page, NAME_BBOX, work_dir=tmp_path)

    assert seen
    for image_arg in seen:
        assert Path(image_arg).is_relative_to(tmp_path)


def test_number_cell_uses_isolated_glyph_psms_and_a_digit_whitelist(
    upright_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD §8.2 — number cells are digits-only regions.

    They are NOT read with `--psm 7` first. Measured on the reference patent: a
    Table 1 number cell is a ~40x49 px digit inside a ~190x510 px ruled box, and
    psm 7 returns punctuation noise on it while psm 8 and 13 read it correctly.
    All modes are tried and the longest reading wins, because they disagree on
    two-digit numbers — compound 41 reads as "4" under psm 8 but "41" under 10.
    """
    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr("sarmine.ocr.tesseract.subprocess.run", spy)
    ocr_number_cell(upright_page, NUMBER_BBOX, work_dir=tmp_path)

    assert calls, "number-cell OCR never invoked tesseract"
    modes = {cmd[cmd.index("--psm") + 1] for cmd in calls if "--psm" in cmd}
    assert {"8", "10"} <= modes, f"isolated-glyph modes not attempted: {modes}"
    for cmd in calls:
        whitelist = [a for a in cmd if a.startswith("tessedit_char_whitelist=")]
        assert whitelist and whitelist[0].endswith("0123456789")


def test_decoding_survives_non_utf8_tesseract_output(
    upright_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Plan 5.1 — tesseract emits invalid UTF-8 on some crops; decoding must not raise.
    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 0, stdout=b"iso\xffindoline", stderr=b"")

    monkeypatch.setattr("sarmine.ocr.tesseract.subprocess.run", fake_run)
    assert "iso" in ocr_region(upright_page, None, work_dir=tmp_path)
