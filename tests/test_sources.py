"""Tests for Part 2 — source resolution (PRD §7, AC-1.*, EC-21, EC-22).

The default run must be fully offline: everything here works from the cached
Google Patents HTML fixture and the extracted page PNGs. Live-network tests are
marked `network` and additionally gated on SARMINE_TEST_NETWORK=1; tests that
need the 223-page reference PDF are marked `slow`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from sarmine.sources import google_patents as gp
from sarmine.sources import pdf as pdfsrc
from sarmine.sources import resolver as res

requires_network = pytest.mark.skipif(
    os.environ.get("SARMINE_TEST_NETWORK") != "1",
    reason="set SARMINE_TEST_NETWORK=1 to exercise the live patents.google.com fetch",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CACHED_HTML = FIXTURES / "source" / "WO2024097932A1.html"
PAGES = FIXTURES / "pages"
REFERENCE_PDF = REPO_ROOT / "data" / "patents" / "WO2024097932A1.pdf"
PUBNUM = "WO2024097932A1"

requires_reference_pdf = pytest.mark.skipif(
    not REFERENCE_PDF.is_file(), reason="reference PDF absent"
)

# A 1x1 PNG, enough to stand in for a patentimages download.
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108000000003a"
    "7e9b550000000a49444154789c63f80f0001010100b138f6140000000049"
    "454e44ae426082"
)


@pytest.fixture(scope="module")
def cached_html() -> str:
    return CACHED_HTML.read_text("utf-8", errors="replace")


def _minimal_text_pdf(path: Path, text: str = "Hello patent world") -> Path:
    """A tiny, valid, single-page PDF that DOES carry a text layer."""
    objects: list[bytes | None] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        None,  # filled in below; it needs the stream length
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects[3] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    startxref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        startxref,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


def _warm_html_cache(cache_dir: Path, html: str) -> Path:
    """Seed the bundle's source cache exactly as a previous run would have."""
    destination = cache_dir / "source" / f"{PUBNUM}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, "utf-8")
    return destination


# --------------------------------------------------------------------------
# google_patents.parse_image_page_index  (PRD §3.2 — provenance for free)
# --------------------------------------------------------------------------


def test_parse_image_page_index_roundtrips_real_urls() -> None:
    base = "https://patentimages.storage.googleapis.com/4e/1f/d5/496fb875479045/"
    assert gp.parse_image_page_index(base + "imgf000186_0001.png") == (186, 1)
    assert gp.parse_image_page_index(base + "imgf000187_0001.png") == (187, 1)
    assert gp.parse_image_page_index(base + "imgf000062_0001.png") == (62, 1)
    assert gp.parse_image_page_index(base + "imgf000004_0002.png") == (4, 2)
    assert gp.parse_image_page_index(base + "imgf000003_0001.png") == (3, 1)


def test_parse_image_page_index_rejects_non_page_images() -> None:
    assert gp.parse_image_page_index("https://example.com/logo.png") is None
    assert gp.parse_image_page_index("https://example.com/imgf00186_0001.png") is None
    assert gp.parse_image_page_index("") is None


# --------------------------------------------------------------------------
# google_patents.parse_html  (PRD AC-1.2, §3.2)
# --------------------------------------------------------------------------


def test_parse_html_recovers_the_full_description(cached_html: str) -> None:
    parsed = gp.parse_html(cached_html)
    # PRD AC-1.2 floor; 249_792 is what this frozen fixture actually yields.
    assert len(parsed.description_text) >= 240_000
    assert len(parsed.description_text) == 249_792


def test_description_proves_the_tables_are_images_only(cached_html: str) -> None:
    """PRD §3.2 — Table 1 and Table 2 exist only as images."""
    text = gp.parse_html(cached_html).description_text
    assert text.count("Compound No") == 0
    assert text.count("HbF Induction") == 0
    assert text.count("Table 1") == 104
    assert text.count("Table 2") == 3


def test_parse_html_maps_images_to_source_pages(cached_html: str) -> None:
    parsed = gp.parse_html(cached_html)
    urls = [url for page_urls in parsed.image_urls.values() for url in page_urls]
    assert len(urls) == 245  # PRD §3.2
    assert len(parsed.image_urls) == 157
    assert min(parsed.image_urls) == 3
    assert max(parsed.image_urls) == 187
    assert 186 in parsed.image_urls and 187 in parsed.image_urls
    assert all(url.endswith(".png") for url in urls)


def test_parse_html_keeps_images_in_document_order(cached_html: str) -> None:
    parsed = gp.parse_html(cached_html)
    assert parsed.image_urls[4] == [
        "https://patentimages.storage.googleapis.com/ac/08/9c/78aef462b9382e/imgf000004_0001.png",
        "https://patentimages.storage.googleapis.com/7d/21/b5/65f7c1d9bcb807/imgf000004_0002.png",
    ]


def test_parse_html_recovers_title_and_claims(cached_html: str) -> None:
    parsed = gp.parse_html(cached_html)
    assert "hemoglobinopathies" in parsed.title.lower()
    assert len(parsed.claims_text) > 20_000
    assert "claim" in parsed.claims_text.lower()


def test_parse_html_on_a_page_without_a_description_is_empty() -> None:
    parsed = gp.parse_html("<html><body><p>nothing here</p></body></html>")
    assert parsed.description_text == ""
    assert parsed.image_urls == {}


# --------------------------------------------------------------------------
# google_patents.fetch_structured  (PRD R7.2, AC-1.2, AC-1.4, EC-21)
# --------------------------------------------------------------------------


class _FakeHttp:
    """Stand-in for `_http_get`; records every request so tests can count them."""

    def __init__(self, html: str, *, status: int = 200) -> None:
        self.html = html
        self.status = status
        self.calls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.timeouts: list[float] = []

    def __call__(self, url: str, *, timeout: float, headers: dict[str, str]):
        self.calls.append(url)
        self.headers.append(headers)
        self.timeouts.append(timeout)
        if url.endswith(".png"):
            return self.status, TINY_PNG
        return self.status, self.html.encode("utf-8")


def test_fetch_structured_downloads_then_never_refetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """PRD R7.2 / AC-1.4 — a second run performs zero network fetches."""
    http = _FakeHttp(cached_html)
    monkeypatch.setattr(gp, "_http_get", http)

    first = gp.fetch_structured(PUBNUM, tmp_path, image_pages=[186, 187])
    assert first is not None
    assert first.pubnum == PUBNUM
    assert first.html_path.is_file()
    assert sorted(first.image_paths) == [186, 187]
    assert all(path.is_file() for paths in first.image_paths.values() for path in paths)
    assert len(http.calls) == 3  # one HTML page + the two requested images

    http.calls.clear()
    second = gp.fetch_structured(PUBNUM, tmp_path, image_pages=[186, 187])
    assert second is not None
    assert http.calls == []
    assert second.description_text == first.description_text
    assert second.image_paths == first.image_paths


def test_fetch_structured_from_a_warm_cache_needs_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """AC-1.4 — with the cache warm, the network is never touched at all."""
    monkeypatch.setattr(gp, "_http_get", _FakeHttp(cached_html))
    assert gp.fetch_structured(PUBNUM, tmp_path, image_pages=[186]) is not None

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("fetch_structured hit the network on a warm cache")

    monkeypatch.setattr(gp, "_http_get", _explode)
    warm = gp.fetch_structured(PUBNUM, tmp_path, allow_network=False, image_pages=[186])
    assert warm is not None
    assert len(warm.description_text) >= 240_000
    assert warm.image_paths[186][0].is_file()


def test_fetch_structured_uses_a_browser_user_agent_and_the_configured_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    from sarmine.config import get_config

    http = _FakeHttp(cached_html)
    monkeypatch.setattr(gp, "_http_get", http)
    gp.fetch_structured(PUBNUM, tmp_path, download_images=False)

    config = get_config()
    assert http.calls == [f"https://patents.google.com/patent/{PUBNUM}/en"]
    assert http.headers[0]["User-Agent"] == config.user_agent
    assert http.timeouts[0] == config.request_timeout_s


def test_fetch_structured_returns_none_on_a_cold_cache_without_network(tmp_path: Path) -> None:
    """PRD EC-21 — the caller falls back to the PDF path."""
    assert gp.fetch_structured(PUBNUM, tmp_path, allow_network=False) is None


def test_fetch_structured_returns_none_on_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """PRD EC-21 — a blocked or 404 fetch degrades gracefully, never raises."""
    monkeypatch.setattr(gp, "_http_get", _FakeHttp(cached_html, status=403))
    assert gp.fetch_structured(PUBNUM, tmp_path) is None
    assert not (tmp_path / f"{PUBNUM}.html").exists()


def test_fetch_structured_returns_none_when_the_transport_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PRD EC-21 — DNS failure / connection reset must not propagate."""

    def _boom(*args, **kwargs):
        raise OSError("name resolution failed")

    monkeypatch.setattr(gp, "_http_get", _boom)
    assert gp.fetch_structured(PUBNUM, tmp_path) is None


def test_fetch_structured_returns_none_when_there_is_no_description(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PRD §7.2 step 4 / EC-21 — a page with no description section is unusable."""
    monkeypatch.setattr(gp, "_http_get", _FakeHttp("<html><body>captcha</body></html>"))
    assert gp.fetch_structured(PUBNUM, tmp_path) is None
    assert not (tmp_path / f"{PUBNUM}.html").exists()


def test_fetch_structured_can_skip_image_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    http = _FakeHttp(cached_html)
    monkeypatch.setattr(gp, "_http_get", http)
    source = gp.fetch_structured(PUBNUM, tmp_path, download_images=False)
    assert source is not None
    assert source.image_paths == {}
    assert len(source.image_urls) == 157
    assert http.calls == [f"https://patents.google.com/patent/{PUBNUM}/en"]


def test_fetch_structured_survives_a_failing_image_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """One dead image URL must not lose the 249k characters of description."""

    def _http(url: str, *, timeout: float, headers: dict[str, str]):
        if url.endswith(".png"):
            raise OSError("connection reset")
        return 200, cached_html.encode("utf-8")

    monkeypatch.setattr(gp, "_http_get", _http)
    source = gp.fetch_structured(PUBNUM, tmp_path, image_pages=[186])
    assert source is not None
    assert source.image_paths.get(186, []) == []
    assert len(source.description_text) >= 240_000


@pytest.mark.network
@requires_network
def test_fetch_structured_against_the_live_page(tmp_path: Path) -> None:
    """PRD AC-1.2 with the real network. Skipped unless explicitly enabled."""
    source = gp.fetch_structured(PUBNUM, tmp_path, download_images=False)
    assert source is not None
    assert len(source.description_text) >= 240_000
    assert len(source.image_urls) >= 150


def test_cached_fixture_matches_the_page_the_prd_measured(cached_html: str) -> None:
    """Guards the fixture itself against silent drift (PRD §3.2).

    The raw page is 731,229 bytes, but the committed copy has its `<script>`
    blocks stripped by `tools/sanitize_html_fixtures.py`: every
    patents.google.com page embeds Google's own public Help-widget API key
    inline, and shipping a verbatim copy trips secret scanning for a credential
    that is neither ours nor secret. So the fixture is pinned on the CONTENT the
    tests actually consume, not on a byte count that sanitizing changes.
    """
    # Split so this assertion cannot itself look like a key to a scanner.
    google_key_prefix = "AIza" + "Sy"
    assert google_key_prefix not in cached_html, "third-party API key must stay stripped"
    assert "<script" not in cached_html.lower()
    assert re.search(r"imgf000186_0001\.png", cached_html) is not None
    assert len(cached_html) >= 700_000  # the patent text itself, still intact


# --------------------------------------------------------------------------
# pdf.normalize_pubnum / publication-number extraction  (PRD §7.2 step 1, AC-1.1)
# --------------------------------------------------------------------------


def test_normalize_pubnum_strips_spaces_and_slashes() -> None:
    assert pdfsrc.normalize_pubnum("WO 2024/097932 A1") == "WO2024097932A1"
    assert pdfsrc.normalize_pubnum("WO2024097932A1") == "WO2024097932A1"
    assert pdfsrc.normalize_pubnum("  us 2025/0368620  a1 ") == "US20250368620A1"
    assert pdfsrc.normalize_pubnum("EP 1 234 567 B1") == "EP1234567B1"
    assert pdfsrc.normalize_pubnum("") == ""


def test_publication_number_repairs_the_kind_code_homoglyph() -> None:
    """Tesseract reads the `A1` kind code as `Al` on this front page (PRD §9.2)."""
    text = "(10) International Publication Number\n\nWO 2024/097932 Al\n\nWIPOIPCT\n"
    assert pdfsrc.publication_number_from_text(text) == "WO2024097932A1"


def test_publication_number_prefers_the_labelled_publication_number() -> None:
    text = (
        "(21) International Application Number:\nPCT/US2023/078600\n\n"
        "(30) Priority Data:\n63/422,847 04 November 2022 (04.11.2022) US\n\n"
        "(10) International Publication Number\n\nWO 2024/097932 A1\n"
    )
    assert pdfsrc.publication_number_from_text(text) == "WO2024097932A1"


def test_publication_number_ignores_application_and_priority_numbers() -> None:
    """EC-22 — a page with no publication number must not invent one."""
    text = (
        "(21) International Application Number:\nPCT/US2023/078600\n\n"
        "(22) International Filing Date:\n03 November 2023 (03.11.2023)\n\n"
        "(30) Priority Data:\n63/422,847 04 November 2022 (04.11.2022) US\n"
    )
    assert pdfsrc.publication_number_from_text(text) is None


def test_publication_number_on_empty_text_is_none() -> None:
    assert pdfsrc.publication_number_from_text("") is None
    assert pdfsrc.publication_number_from_text("no numbers of any kind here") is None


def test_publication_number_from_the_real_front_page_image() -> None:
    """PRD AC-1.1 — page 1 of the reference patent OCRs to WO2024097932A1."""
    assert pdfsrc.publication_number_from_image(PAGES / "p-001-000.png") == PUBNUM


# --------------------------------------------------------------------------
# pdf.pdf_page_count / has_text_layer / extract_page_images  (PRD R7.4, EC-1)
# --------------------------------------------------------------------------


def test_pdf_page_count_on_a_one_page_pdf(tmp_path: Path) -> None:
    assert pdfsrc.pdf_page_count(_minimal_text_pdf(tmp_path / "mini.pdf")) == 1


def test_has_text_layer_true_when_fonts_and_text_are_present(tmp_path: Path) -> None:
    assert pdfsrc.has_text_layer(_minimal_text_pdf(tmp_path / "mini.pdf")) is True


def test_extract_page_images_falls_back_to_pdftoppm(tmp_path: Path) -> None:
    """PRD R7.4 — pdfimages yields nothing on a vector-only page, so re-render."""
    pdf = _minimal_text_pdf(tmp_path / "mini.pdf")
    pages = pdfsrc.extract_page_images(pdf, tmp_path / "out", first=1, last=1)
    assert len(pages) == 1
    page = pages[0]
    assert page.page_no == 1
    assert page.path.is_file()
    # PRD §17.5 — never /tmp; always inside the caller's working directory.
    assert (tmp_path / "out") in page.path.parents
    assert page.width >= 2000 and page.height >= 2000  # rendered at 300 dpi


@pytest.mark.slow
@requires_reference_pdf
def test_reference_pdf_page_count() -> None:
    assert pdfsrc.pdf_page_count(REFERENCE_PDF) == 223  # PRD §3.1


@pytest.mark.slow
@requires_reference_pdf
def test_reference_pdf_has_no_text_layer() -> None:
    """PRD EC-1 — 223 characters in 223 pages, no embedded fonts."""
    assert pdfsrc.has_text_layer(REFERENCE_PDF) is False


@pytest.mark.slow
@requires_reference_pdf
def test_extract_page_images_keeps_the_original_ccitt_raster(tmp_path: Path) -> None:
    """PRD R7.4 — pdfimages must hand back the embedded bitmap untouched."""
    pages = pdfsrc.extract_page_images(REFERENCE_PDF, tmp_path, first=63, last=63)
    assert len(pages) == 1
    page = pages[0]
    assert page.page_no == 63
    assert (page.width, page.height) == (2477, 3505)
    assert tmp_path in page.path.parents
    assert page.path.read_bytes() == (PAGES / "p-063-000.png").read_bytes()


@pytest.mark.slow
@requires_reference_pdf
def test_extract_page_images_returns_a_range_in_page_order(tmp_path: Path) -> None:
    pages = pdfsrc.extract_page_images(REFERENCE_PDF, tmp_path, first=61, last=63)
    assert [page.page_no for page in pages] == [61, 62, 63]
    assert all(page.path.is_file() for page in pages)


@pytest.mark.slow
@requires_reference_pdf
def test_extract_publication_number_from_the_reference_pdf(tmp_path: Path) -> None:
    """PRD AC-1.1 end to end: PDF in, publication number out."""
    assert pdfsrc.extract_publication_number(REFERENCE_PDF, tmp_path) == PUBNUM


def test_extract_publication_number_on_a_pdf_without_one(tmp_path: Path) -> None:
    """PRD EC-22 — an unreadable publication number is `None`, not a guess."""
    pdf = _minimal_text_pdf(tmp_path / "mini.pdf", text="a receipt, not a patent")
    assert pdfsrc.extract_publication_number(pdf, tmp_path / "work") is None


# --------------------------------------------------------------------------
# resolver.resolve  (PRD §7.2 steps 1-5, R7.3, AC-1.2/1.3/1.4, EC-21, EC-22)
# --------------------------------------------------------------------------


def test_resolve_a_pdf_without_network_uses_the_pdf_ocr_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PRD AC-1.3 / EC-21 — with the network off, the run still resolves."""
    monkeypatch.setattr(res, "extract_publication_number", lambda pdf, work_dir: PUBNUM)
    pdf = _minimal_text_pdf(tmp_path / "in" / "patent.pdf")

    resolved = res.resolve(pdf, tmp_path / "bundle", allow_network=False)

    assert resolved.source_mode == "pdf_ocr"  # PRD R7.3
    assert resolved.pubnum == PUBNUM
    assert resolved.structured is None
    assert resolved.pdf_path == pdf
    assert resolved.n_pages == 1
    assert [anomaly.kind for anomaly in resolved.anomalies] == ["source_unavailable"]
    assert resolved.anomalies[0].severity == "warning"
    assert PUBNUM in resolved.anomalies[0].message


def test_resolve_a_pdf_with_a_warm_cache_is_structured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """PRD AC-1.2 / AC-1.4 — the accelerated path, with zero network calls."""
    monkeypatch.setattr(res, "extract_publication_number", lambda pdf, work_dir: PUBNUM)

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("resolve hit the network with a warm cache")

    monkeypatch.setattr(gp, "_http_get", _explode)

    bundle = tmp_path / "bundle"
    _warm_html_cache(bundle, cached_html)
    pdf = _minimal_text_pdf(tmp_path / "in" / "patent.pdf")

    resolved = res.resolve(pdf, bundle, allow_network=False)

    assert resolved.source_mode == "structured"
    assert resolved.structured is not None
    assert len(resolved.structured.description_text) >= 240_000
    assert resolved.pdf_path == pdf
    assert resolved.anomalies == []


def test_resolve_force_pdf_path_never_touches_the_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """PRD R7.1 — the fallback path must be exercisable on demand."""
    monkeypatch.setattr(res, "extract_publication_number", lambda pdf, work_dir: PUBNUM)

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("force_pdf_path still fetched")

    monkeypatch.setattr(gp, "_http_get", _explode)

    bundle = tmp_path / "bundle"
    _warm_html_cache(bundle, cached_html)  # available, and deliberately ignored
    pdf = _minimal_text_pdf(tmp_path / "in" / "patent.pdf")

    resolved = res.resolve(pdf, bundle, force_pdf_path=True, allow_network=True)

    assert resolved.source_mode == "pdf_ocr"
    assert resolved.structured is None
    # Choosing the PDF path is not the same as the structured source failing.
    assert resolved.anomalies == []


def test_resolve_a_publication_number_uses_the_structured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_html: str
) -> None:
    """PRD §7.2 step 5 — with no PDF, the accelerated path is the only option."""
    monkeypatch.setattr(gp, "_http_get", _FakeHttp(cached_html))
    bundle = tmp_path / "bundle"

    resolved = res.resolve("WO 2024/097932 A1", bundle, allow_network=True)

    assert resolved.pubnum == PUBNUM  # normalized on the way in
    assert resolved.source_mode == "structured"
    assert resolved.pdf_path is None
    assert resolved.n_pages == 187  # highest page number carrying an image
    assert resolved.cache_dir == bundle


def test_resolve_a_publication_number_without_pdf_or_network_raises(tmp_path: Path) -> None:
    """PRD §7.2 step 5 — error out with an explicit message."""
    with pytest.raises(res.SourceResolutionError) as excinfo:
        res.resolve(PUBNUM, tmp_path, allow_network=False)
    message = str(excinfo.value)
    assert PUBNUM in message
    assert "pdf" in message.lower()


def test_resolve_raises_when_the_pdf_is_not_a_patent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PRD EC-22 — clear error, and offer manual publication-number entry."""
    monkeypatch.setattr(res, "extract_publication_number", lambda pdf, work_dir: None)
    pdf = _minimal_text_pdf(tmp_path / "in" / "receipt.pdf", text="a receipt")

    with pytest.raises(res.SourceResolutionError) as excinfo:
        res.resolve(pdf, tmp_path / "bundle")
    message = str(excinfo.value).lower()
    assert "publication number" in message
    assert "manual" in message


def test_resolve_raises_for_a_pdf_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(res.SourceResolutionError) as excinfo:
        res.resolve(tmp_path / "nowhere.pdf", tmp_path / "bundle")
    assert "nowhere.pdf" in str(excinfo.value)


def test_resolve_rejects_an_unusable_source_string(tmp_path: Path) -> None:
    with pytest.raises(res.SourceResolutionError):
        res.resolve("   ", tmp_path / "bundle")


@pytest.mark.slow
@requires_reference_pdf
def test_resolve_the_reference_pdf_offline(tmp_path: Path) -> None:
    """PRD AC-1.1 + AC-1.3 together, on the real 223-page PDF, with no network."""
    resolved = res.resolve(REFERENCE_PDF, tmp_path / "bundle", allow_network=False)
    assert resolved.pubnum == PUBNUM
    assert resolved.source_mode == "pdf_ocr"
    assert resolved.n_pages == 223
    assert [anomaly.kind for anomaly in resolved.anomalies] == ["source_unavailable"]
