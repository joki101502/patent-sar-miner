"""Source resolution — PRD §7.2, in the normative order (Plan Part 2.3).

    1. a PDF input has page 1 rendered at 300 dpi and OCR'd for the publication
       number;
    2. that number is looked up on patents.google.com;
    3. a page carrying a description section wins — `source_mode="structured"`;
    4. anything else falls back to full PDF OCR — `source_mode="pdf_ocr"`;
    5. a publication number with no PDF has no fallback, so it errors clearly.

`force_pdf_path=True` skips steps 2-3 entirely; that is the hook PRD R7.1
demands, so the fallback path can be exercised with the network disabled.

Resolution deliberately rasterizes only page 1: the pipeline decides which of
the remaining pages it needs, and PRD §17.1 makes page-at-a-time processing a
memory requirement rather than a style preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..artifacts.schema import DocumentAnomaly
from .google_patents import StructuredSource, fetch_structured
from .pdf import extract_publication_number, normalize_pubnum, pdf_page_count

_MANUAL_ENTRY_HINT = (
    "Enter the publication number manually (for example WO2024097932A1) "
    "or supply the patent PDF."
)


class SourceResolutionError(Exception):
    """Nothing usable could be resolved from the input (PRD §7.2 step 5, EC-22)."""


@dataclass
class ResolvedSource:
    """What the rest of the pipeline needs to start work on one publication."""

    pubnum: str
    source_mode: Literal["structured", "pdf_ocr"]  # PRD R7.3
    structured: StructuredSource | None
    pdf_path: Path | None
    cache_dir: Path
    n_pages: int
    anomalies: list[DocumentAnomaly] = field(default_factory=list)


def resolve(
    source: str | Path,
    cache_dir: Path,
    *,
    force_pdf_path: bool = False,
    allow_network: bool = True,
) -> ResolvedSource:
    """Resolve a PDF path or a publication number into a usable source."""
    cache_dir = Path(cache_dir)
    pdf_path = _pdf_input(source)

    if pdf_path is not None:
        pubnum = extract_publication_number(pdf_path, cache_dir / "pages")
        if pubnum is None:
            # PRD EC-22 — the publication number keys everything downstream, so
            # guessing one would be worse than refusing.
            raise SourceResolutionError(
                f"Could not read a publication number from page 1 of {pdf_path.name}. "
                f"{_MANUAL_ENTRY_HINT}"
            )
        n_pages = pdf_page_count(pdf_path)
    else:
        pubnum = normalize_pubnum(str(source))
        if not pubnum:
            raise SourceResolutionError(
                f"{source!r} is neither a PDF path nor a publication number. "
                f"{_MANUAL_ENTRY_HINT}"
            )
        n_pages = 0

    structured = None
    if not force_pdf_path:
        structured = fetch_structured(
            pubnum, cache_dir / "source", allow_network=allow_network
        )

    if structured is not None:
        return ResolvedSource(
            pubnum=pubnum,
            source_mode="structured",
            structured=structured,
            pdf_path=pdf_path,
            cache_dir=cache_dir,
            # Without a PDF the highest image-bearing page is the only page
            # count available, and it is a lower bound (PRD §3.2: images run 3-187).
            n_pages=n_pages or max(structured.image_urls, default=0),
        )

    if pdf_path is None:
        raise SourceResolutionError(
            f"Could not retrieve {pubnum} from patents.google.com and no PDF was "
            f"supplied to fall back to. {_MANUAL_ENTRY_HINT}"
        )

    anomalies: list[DocumentAnomaly] = []
    if not force_pdf_path:
        anomalies.append(
            DocumentAnomaly(
                kind="source_unavailable",
                severity="warning",
                message=(
                    f"patents.google.com returned no usable description for {pubnum}; "
                    "falling back to full PDF OCR (source_mode=pdf_ocr, PRD EC-21)."
                ),
            )
        )
    return ResolvedSource(
        pubnum=pubnum,
        source_mode="pdf_ocr",
        structured=None,
        pdf_path=pdf_path,
        cache_dir=cache_dir,
        n_pages=n_pages,
        anomalies=anomalies,
    )


def _pdf_input(source: str | Path) -> Path | None:
    path = Path(source)
    if path.suffix.lower() != ".pdf":
        return None
    if not path.is_file():
        raise SourceResolutionError(f"No such PDF: {path}. {_MANUAL_ENTRY_HINT}")
    return path
