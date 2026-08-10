"""The accelerated source path — patents.google.com (PRD §3.2, §7.2 step 3, R7.2).

For the reference patent this page carries ~245k characters of machine-readable
description plus 245 pre-cropped page images whose filenames encode the source
page number, which is where free provenance comes from (PRD §3.2).

It is an accelerator, never a dependency (PRD R7.1): every failure mode returns
`None` so the caller can fall back to full PDF OCR (PRD EC-21). Nothing runs at
import time and nothing touches the network unless `fetch_structured` is called
against a cold cache.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests
from lxml import html as lxml_html

from ..config import Config, get_config

BASE_URL = "https://patents.google.com/patent/{pubnum}/en"

# `imgf{PAGE:06d}_{INDEX:04d}.png` — the page number lives in the filename (PRD §3.2).
_IMAGE_NAME_RE = re.compile(r"^imgf(\d{6})_(\d{4})\.(?:png|jpe?g|tiff?|gif)$", re.IGNORECASE)


@dataclass
class ParsedPatentHtml:
    """What one Google Patents page yields once parsed. Pure data, no I/O."""

    title: str = ""
    description_text: str = ""
    claims_text: str = ""
    abstract_text: str = ""
    image_urls: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class StructuredSource:
    """The accelerated path's output for one publication."""

    pubnum: str
    title: str
    description_text: str
    claims_text: str
    html_path: Path
    image_urls: dict[int, list[str]] = field(default_factory=dict)
    image_paths: dict[int, list[Path]] = field(default_factory=dict)


def parse_image_page_index(url: str) -> tuple[int, int] | None:
    """`.../imgf000186_0001.png` -> `(186, 1)`; `None` when it is not a page image."""
    if not url:
        return None
    name = url.split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    match = _IMAGE_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_html(html: str | bytes) -> ParsedPatentHtml:
    """Extract the description, claims, title and the page -> image-URL map.

    An empty `description_text` is the signal PRD §7.2 step 4 keys off: a page
    that parses but carries no description section is a fetch failure.
    """
    if isinstance(html, str):
        html = html.encode("utf-8", "replace")
    doc = lxml_html.fromstring(html)

    sections = doc.xpath('//section[@itemprop="description"]')
    if not sections:
        return ParsedPatentHtml(title=_title(doc))

    image_urls: dict[int, list[str]] = {}
    for src in sections[0].xpath(".//img/@src"):
        page_index = parse_image_page_index(str(src))
        if page_index is None:
            continue
        bucket = image_urls.setdefault(page_index[0], [])
        url = _absolute(str(src))
        if url not in bucket:
            bucket.append(url)

    return ParsedPatentHtml(
        title=_title(doc),
        description_text=sections[0].text_content(),
        claims_text=_section_text(doc, "claims"),
        abstract_text=_section_text(doc, "abstract"),
        image_urls=image_urls,
    )


def fetch_structured(
    pubnum: str,
    cache_dir: Path,
    *,
    allow_network: bool = True,
    download_images: bool = True,
    image_pages: Iterable[int] | None = None,
) -> StructuredSource | None:
    """Fetch — or re-read from cache — the Google Patents page for `pubnum`.

    Returns `None` and never raises when the page cannot be fetched or carries
    no description section, so the caller can fall back to the PDF path
    (PRD EC-21). A warm `cache_dir` performs zero network requests
    (PRD R7.2 / AC-1.4). `image_pages` restricts which pages are downloaded.
    """
    config = get_config()
    cache_dir = Path(cache_dir)
    html_path = cache_dir / f"{pubnum}.html"

    cached = html_path.is_file()
    if cached:
        raw = html_path.read_bytes()
    else:
        if not allow_network:
            return None
        try:
            status, raw = _http_get(
                BASE_URL.format(pubnum=pubnum),
                timeout=config.request_timeout_s,
                headers={"User-Agent": config.user_agent},
            )
        except Exception:
            return None
        if status != 200 or not raw:
            return None

    try:
        parsed = parse_html(raw)
    except Exception:
        return None
    if not parsed.description_text:
        # Never cache a captcha interstitial or a blocked page — caching it would
        # make the failure permanent for every later run (PRD R7.2).
        return None

    if not cached:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_bytes(raw)

    image_paths: dict[int, list[Path]] = {}
    if download_images:
        wanted = None if image_pages is None else set(image_pages)
        for page_no in sorted(parsed.image_urls):
            if wanted is not None and page_no not in wanted:
                continue
            image_paths[page_no] = _cache_images(
                parsed.image_urls[page_no],
                cache_dir / "images",
                allow_network=allow_network,
                config=config,
            )

    return StructuredSource(
        pubnum=pubnum,
        title=parsed.title,
        description_text=parsed.description_text,
        claims_text=parsed.claims_text,
        html_path=html_path,
        image_urls=parsed.image_urls,
        image_paths=image_paths,
    )


def _http_get(url: str, *, timeout: float, headers: dict[str, str]) -> tuple[int, bytes]:
    """The single network seam in this module; tests replace it wholesale."""
    response = requests.get(url, headers=headers, timeout=timeout)
    return response.status_code, response.content


def _cache_images(
    urls: list[str], image_dir: Path, *, allow_network: bool, config: Config
) -> list[Path]:
    kept: list[Path] = []
    for url in urls:
        name = url.split("?", 1)[0].rsplit("/", 1)[-1]
        if not _IMAGE_NAME_RE.match(name):
            continue
        destination = image_dir / name
        if not destination.is_file():
            if not allow_network:
                continue
            try:
                status, payload = _http_get(
                    url,
                    timeout=config.request_timeout_s,
                    headers={"User-Agent": config.user_agent},
                )
            except Exception:
                continue
            if status != 200 or not payload:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        kept.append(destination)
    return kept


def _absolute(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http"):
        return url
    return urljoin("https://patents.google.com/", url)


def _section_text(doc: lxml_html.HtmlElement, itemprop: str) -> str:
    nodes = doc.xpath(f'//section[@itemprop="{itemprop}"]')
    return nodes[0].text_content() if nodes else ""


def _title(doc: lxml_html.HtmlElement) -> str:
    for expression in ('//meta[@name="DC.title"]/@content', "//h1[@id='title']/text()"):
        values = doc.xpath(expression)
        if values:
            title = " ".join(str(values[0]).split())
            if title:
                return title
    return ""
