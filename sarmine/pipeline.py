"""The staged orchestrator (PRD §17.1, Plan Part 12).

The stage order is a **memory requirement, not a style preference**: peak memory
must equal the largest stage, not the sum of stages, because the deploy target
has 2.7 GB and MolScribe alone resides at ~1.3 GB. So all OCR and the OPSIN JVM
finish and exit before torch is imported, MolScribe is loaded last and freed
immediately (R17.1, R17.4), page images are streamed rather than retained
(R17.3), and RSS is recorded at every stage boundary (R17.6).
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

from sarmine.artifacts.schema import (
    Compound,
    DocumentAnomaly,
    Measurement,
    Provenance,
    RunManifest,
)
from sarmine.artifacts.writer import ensure_bundle_dir, write_bundle
from sarmine.assay.lexicon import HeaderMatch, load_lexicon, reconstruct_header
from sarmine.assay.legend import detect_cross_reference_error, parse_legends
from sarmine.assay.normalize import build_measurement
from sarmine.assay.qc import assay_group_key, detect_transcription_errors
from sarmine.config import get_config
from sarmine.join.linker import ActivityRow, ExampleEntry, join
from sarmine.ocr.homoglyph import clean_ocr_name, repair_batch
from sarmine.ocr.tesseract import ocr_number_cell, ocr_region
from sarmine.rank.scorer import (
    apply_investment_signal,
    compute_properties,
    detect_in_vivo,
    rank_compounds,
)
from sarmine.resources import STAGE_ORDER, ProgressCallback, StageRecorder, release
from sarmine.segment import tatr
from sarmine.segment.crops import write_crop
from sarmine.segment.reconcile import reconcile
from sarmine.segment.roles import assign_column_roles
from sarmine.segment.rulings import Grid, detect_grid, load_gray
from sarmine.segment.stitch import TablePage, columns_match, looks_like_header, stitch_tables
from sarmine.sources.pdf import PdfPage, extract_page_images, ocr_image, resolve_rotation
from sarmine.sources.resolver import ResolvedSource, resolve
from sarmine.structure.crosscheck import crosscheck
from sarmine.structure.opsin import parse_names, smiles_to_inchikey
from sarmine.structure.standardize import rdkit_version, standardize_smiles

TableKind = Literal["compound_table", "activity_table"]

# The compound-number column of the activity table. Anything else is an assay.
_NUMBER_HEADER = re.compile(r"comp(oun)?d\s*(no|number|#)", re.IGNORECASE)
_EXAMPLE_HEADING = re.compile(
    r"\[?\d{0,6}\]?\s*Example\s+(?P<id>\d+[A-Za-z]?)\s*[:.\u2013-]\s*(?P<name>[^\n]{20,400})",
    re.IGNORECASE,
)
_NMR = re.compile(r"\bNMR\b", re.IGNORECASE)
_MS = re.compile(r"MS\s*\(?ESI", re.IGNORECASE)

# Ink coverage above this in the widest column means a structure drawing rather
# than dense text; used only to help role assignment, never to route on its own.
_CELL_INSET_PX = 8
# Measured dark-pixel counts on the reference patent: a blank activity cell has
# exactly 0, while a compound-number cell has 148-450. An absolute floor is the
# right shape — as a *fraction* of its cell a compound number is only 0.0007 to
# 0.0019, below any fraction threshold that would also reject a blank cell,
# because the number column is narrow and tall with one small glyph in it.
BLANK_CELL_MIN_DARK_PIXELS = 20
_MIN_TABLE_ROWS = 1
_MIN_TABLE_COLS = 2


@dataclass
class TablePageInfo:
    page_no: int
    kind: TableKind | None
    grid: Grid
    image_path: Path
    rotation: object
    roles: dict[int, str]
    column_text: dict[int, str]
    is_continuation: bool = False


@dataclass
class PipelineResult:
    bundle_dir: Path
    manifest: RunManifest
    compounds: list[Compound] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    anomalies: list[DocumentAnomaly] = field(default_factory=list)


# --------------------------------------------------------------------------
# page classification
# --------------------------------------------------------------------------


def classify_table_page(
    grid: Grid, column_text: dict[int, str], column_ink: dict[int, float] | None = None
) -> TableKind | None:
    """A grid plus its column text decides which router a page feeds.

    PRD R8.1 makes this the safety-critical decision: a page misread as a
    compound table would send an activity cell to OCSR, and one misread the
    other way would send a structure drawing to Tesseract.
    """
    if grid.n_rows < _MIN_TABLE_ROWS or grid.n_cols < _MIN_TABLE_COLS:
        return None

    # The activity signature is the specific one — a compound-number header
    # beside a header the assay lexicon recognizes — so it is tested first.
    # `assign_column_roles` always names a structure column on a 3+ column grid,
    # so the compound-table test alone would claim the activity table too.
    lexicon = load_lexicon()
    has_number_header = any(_NUMBER_HEADER.search(text) for text in column_text.values())
    has_assay_header = any(
        lexicon.match(_first_line(text)) is not None
        for text in column_text.values()
        if _first_line(text)
    )
    if has_number_header and has_assay_header:
        return "activity_table"

    roles = assign_column_roles(grid, column_text)
    name_col = next((c for c, role in roles.items() if role == "name"), None)
    if "structure" in roles.values() and name_col is not None:
        if _looks_like_a_chemical_name(column_text.get(name_col, "")):
            return "compound_table"
    return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


# A locant — `2-(`, `-3-yl`, `1,3-dione`. Systematic names always carry them;
# running prose and table headers do not.
_LOCANT = re.compile(r"\d\s*[-,(]|[-(]\s*\d")


def _looks_like_a_chemical_name(text: str) -> bool:
    """A name cell holds long lowercase chemical words; an activity cell a letter.

    The length test alone is not enough: it is applied to lowercased text, so a
    header strip reading `Compound No. Structure Name` passes on "structure".
    Requiring a locant as well separates a systematic name from ordinary words.
    """
    if not any(len(token) >= 8 for token in re.findall(r"[a-z]{4,}", text.lower())):
        return False
    return bool(_LOCANT.search(text))


def _column_text(image_path: Path, grid: Grid, col: int, *, rows: int = 3) -> str:
    """OCR the first few cells of one column, enough to type the column."""
    chunks: list[str] = []
    for row in range(min(rows, grid.n_rows)):
        cell = grid.cell(row, col)
        if cell is None:
            continue
        chunks.append(ocr_region(image_path, cell.bbox, psm=6))
    return "\n".join(chunks)


def _column_ink(image_path: Path, grid: Grid) -> dict[int, float]:
    """Per-column ink coverage, reported for the manifest but NOT used to route.

    Measured on the reference patent's Table 1: a dense block of IUPAC text
    covers ~3.3% of its cell while the thin-line structure drawing beside it
    covers ~2.6%. Ink therefore picks the *name* column as the structure column,
    which is exactly the swap R8.1 forbids. Column width is the reliable signal
    (1574 px of drawing against 923 px of name), so role assignment uses that.
    """
    gray = load_gray(image_path)
    ink: dict[int, float] = {}
    for col in range(grid.n_cols):
        cell = grid.cell(0, col) or grid.cell(max(grid.n_rows - 1, 0), col)
        if cell is None:
            continue
        x0, y0, x1, y1 = cell.bbox
        patch = gray[y0:y1, x0:x1]
        ink[col] = float((patch < 128).mean()) if patch.size else 0.0
    return ink


def _probe_looks_ruled(probe: Grid) -> bool:
    """Cheap pre-filter, run before orientation is known.

    A page printed 90° rotated has its rows and columns transposed, so a probe
    that demanded `n_cols >= 2` would discard exactly the Table 1 pages this
    system exists to read. The test is therefore orientation-agnostic.
    """
    return max(probe.n_rows, probe.n_cols) >= 2 and probe.n_rows + probe.n_cols >= 3


def _morphology_grid(image_path: Path) -> Grid | None:
    morph = detect_grid(image_path)
    if morph.n_rows < _MIN_TABLE_ROWS or morph.n_cols < _MIN_TABLE_COLS:
        return None
    return morph


def _wants_second_opinion(morph: Grid) -> bool:
    """R8.3 — morphology alone recovered only 37 of 54 cells in the measured spike."""
    return morph.completeness < 1.0 or morph.n_rows < 2


def find_table_pages(
    page_images: dict[int, Path], work_dir: Path
) -> dict[int, TablePageInfo]:
    """Locate and orient the table pages, streaming one page at a time (R17.3)."""
    work_dir = Path(work_dir)
    rotated_dir = work_dir / "rotated"
    rotated_dir.mkdir(parents=True, exist_ok=True)

    oriented: list[tuple[int, object, Path, Grid]] = []
    for page_no, path in sorted(page_images.items()):
        if not _probe_looks_ruled(detect_grid(path)):
            continue

        # Orientation is only resolved for pages that look ruled: OSD plus two
        # verification OCR passes is far too expensive to run on 223 pages.
        rotation = resolve_rotation(path, rotated_dir, page_no=page_no)
        morph = _morphology_grid(rotation.path)
        if morph is None:
            continue
        oriented.append((page_no, rotation, rotation.path, morph))

    # One subprocess for every page that needs a second opinion (R17.1): TATR
    # inference peaks around 1.1 GB, and out of process the OS reclaims all of
    # it instead of it sitting under MolScribe for the rest of the run.
    needs_tatr = [p for _, _, p, morph in oriented if _wants_second_opinion(morph)]
    tatr_grids = tatr.detect_grids_isolated(needs_tatr) if needs_tatr else {}

    candidates: dict[int, TablePageInfo] = {}
    for page_no, rotation, image_path, morph in oriented:
        second = tatr_grids.get(str(image_path))
        grid = reconcile(morph, second).grid if second is not None else morph

        column_text = {c: _column_text(image_path, grid, c) for c in range(grid.n_cols)}
        column_ink = _column_ink(image_path, grid)
        kind = classify_table_page(grid, column_text, column_ink)

        candidates[page_no] = TablePageInfo(
            page_no=page_no,
            kind=kind,
            grid=grid,
            image_path=image_path,
            rotation=rotation,
            roles=assign_column_roles(grid, column_text),
            column_text=column_text,
        )

    _inherit_continuations(candidates)
    return {n: p for n, p in candidates.items() if p.kind is not None}


def _inherit_continuations(candidates: dict[int, TablePageInfo]) -> None:
    """EC-3 — a continuation page carries no header, so it cannot classify itself.

    Table 2 spans pages 186-187 and page 187 opens directly at compound 32. No
    surveyed tool emits a continuation flag (R8.5), so the page inherits its
    predecessor's kind when their column geometry matches and its first row does
    not parse as a header.
    """
    for page_no in sorted(candidates):
        page = candidates[page_no]
        if page.kind is not None:
            continue
        previous = candidates.get(page_no - 1)
        if previous is None or previous.kind is None:
            continue
        if not columns_match(previous.grid, page.grid):
            continue
        if looks_like_header([_first_line(t) for t in page.column_text.values()]):
            continue
        page.kind = previous.kind
        page.is_continuation = True


# --------------------------------------------------------------------------
# compound table -> names, numbers, structure crops
# --------------------------------------------------------------------------


@dataclass
class CompoundCell:
    page_no: int
    compound_number: int | None
    name_raw: str
    structure_crop: Path | None
    provenance: list[Provenance]


def _read_compound_table(
    pages: Sequence[TablePageInfo], crops_dir: Path, source_mode: str
) -> list[CompoundCell]:
    from sarmine.ocr.tesseract import ocr_number_cell

    cells: list[CompoundCell] = []
    for page in pages:
        role_by_col = page.roles
        for row in range(page.grid.n_rows):
            number = None
            name_raw = ""
            structure_crop: Path | None = None
            provenance: list[Provenance] = []

            for col, role in role_by_col.items():
                cell = page.grid.cell(row, col)
                if cell is None or role == "unknown":
                    continue
                prov = write_crop(
                    page.image_path,
                    cell.bbox,
                    crops_dir,
                    page_no=page.page_no,
                    kind=role,
                    idx=row,
                    source=source_mode,
                    extractor="tesseract",
                    rotation_applied=getattr(page.rotation, "applied", 0),
                )
                provenance.append(prov)
                crop_path = Path(prov.crop_path)
                if not crop_path.is_absolute():
                    crop_path = crops_dir.parent / prov.crop_path

                if role == "number":
                    number = ocr_number_cell(crop_path)
                elif role == "name":
                    # R8.1 — only the name sub-cell reaches Tesseract, which is
                    # what took OPSIN's parse rate from 0/61 to 33/37 (PRD §8.1).
                    name_raw = ocr_region(crop_path, psm=6)
                elif role == "structure":
                    structure_crop = crop_path

            if not name_raw.strip() and structure_crop is None:
                continue
            cells.append(
                CompoundCell(
                    page_no=page.page_no,
                    compound_number=number,
                    name_raw=name_raw,
                    structure_crop=structure_crop,
                    provenance=provenance,
                )
            )
    return cells


# --------------------------------------------------------------------------
# activity table
# --------------------------------------------------------------------------


def _inset(bbox: tuple[int, int, int, int], margin: int = _CELL_INSET_PX) -> tuple[int, int, int, int]:
    """Shrink a cell away from its ruling lines before OCR.

    Measured on Table 2: without the inset the ruling bleeds into the crop and
    every single-letter cell reads as two characters — `A` becomes `pA`, `D`
    becomes `DO`, and the compound number reads as nothing at all. With it the
    same cells read `1 A E G`, matching the hand-checked values exactly.
    """
    x0, y0, x1, y1 = bbox
    if x1 - x0 <= 2 * margin or y1 - y0 <= 2 * margin:
        return bbox
    return (x0 + margin, y0 + margin, x1 - margin, y1 - margin)


def cell_is_blank(image_path: Path, bbox: tuple[int, int, int, int]) -> bool:
    """Decide emptiness from pixels, because OCR hallucinates on empty crops.

    EC-7 — measured on page 187: the blank HbF cells contain exactly zero dark
    pixels and Tesseract returns `Be` for every one of them. A fabricated value
    is worse than a missing one, because ranking would then treat an untested
    compound as a tested one.
    """
    x0, y0, x1, y1 = _inset(bbox)
    if x1 <= x0 or y1 <= y0:
        return True
    patch = load_gray(image_path)[y0:y1, x0:x1]
    if patch.size == 0:
        return True
    return int((patch < 128).sum()) < BLANK_CELL_MIN_DARK_PIXELS


def _cell_text(image_path: Path, bbox: tuple[int, int, int, int]) -> str:
    if cell_is_blank(image_path, bbox):
        return ""
    bbox = _inset(bbox)
    for psm in (7, 6, 10):
        text = ocr_region(image_path, bbox, psm=psm).strip()
        if text:
            return text
    return ""


def _read_activity_pages(pages: Sequence[TablePageInfo]) -> list[TablePage]:
    table_pages: list[TablePage] = []
    for page in pages:
        rows: list[list[str]] = []
        for row in range(page.grid.n_rows):
            texts = []
            for col in range(page.grid.n_cols):
                cell = page.grid.cell(row, col)
                texts.append(_cell_text(page.image_path, cell.bbox) if cell else "")
            rows.append(texts)
        table_pages.append(
            TablePage(
                page_no=page.page_no,
                grid=page.grid,
                first_row_text=rows[0] if rows else [],
                rows=rows,
            )
        )
    return table_pages


def _match_headers(header: Sequence[str]) -> list[HeaderMatch | None]:
    lexicon = load_lexicon()
    return [lexicon.match(reconstruct_header([h])) if h.strip() else None for h in header]


# --------------------------------------------------------------------------
# prose
# --------------------------------------------------------------------------


def _prose_from_pdf(page_images: dict[int, Path], pages: Iterable[int]) -> str:
    chunks = []
    for page_no in sorted(set(pages)):
        path = page_images.get(page_no)
        if path is not None:
            chunks.append(ocr_image(path, psm=4))
    return "\n".join(chunks)


def _examples_from_prose(prose: str) -> list[ExampleEntry]:
    entries: list[ExampleEntry] = []
    matches = list(_EXAMPLE_HEADING.finditer(prose))
    for idx, match in enumerate(matches):
        body = prose[match.end() : matches[idx + 1].start() if idx + 1 < len(matches) else len(prose)]
        entries.append(
            ExampleEntry(
                local_id=match.group("id"),
                name=" ".join(match.group("name").split()),
                inchikey=None,
                smiles=None,
                page_no=0,
                has_nmr=bool(_NMR.search(body)),
                has_ms=bool(_MS.search(body)),
            )
        )
    return entries


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _expand_ranges(ranges: Sequence[tuple[int, int]] | None, n_pages: int) -> list[int] | None:
    if not ranges:
        return None
    pages: list[int] = []
    for first, last in ranges:
        pages.extend(range(max(1, first), min(n_pages, last) + 1))
    return sorted(set(pages))


def run_pipeline(
    source: str | Path,
    out_root: Path | None = None,
    *,
    force_pdf_path: bool = False,
    target_assay: str | None = None,
    off_target_assay: str | None = None,
    run_ocsr: bool = True,
    max_pages: int | None = None,
    page_ranges: Sequence[tuple[int, int]] | None = None,
    allow_network: bool = True,
    on_progress: ProgressCallback | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    cfg = get_config()
    out_root = Path(out_root or cfg.artifact_root)
    target_assay = target_assay or cfg.target_assay
    off_target_assay = off_target_assay if off_target_assay is not None else cfg.off_target_assay
    run_id = run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")

    recorder = StageRecorder(on_progress=on_progress, stages=STAGE_ORDER)
    anomalies: list[DocumentAnomaly] = []

    # ---- 1. resolve --------------------------------------------------------
    with recorder.stage("resolve"):
        staging = out_root / "_staging" / run_id
        staging.mkdir(parents=True, exist_ok=True)
        resolved: ResolvedSource = resolve(
            source, staging, force_pdf_path=force_pdf_path, allow_network=allow_network
        )
        anomalies.extend(resolved.anomalies)
        bundle_dir = ensure_bundle_dir(out_root, resolved.pubnum, run_id)
        crops_dir = bundle_dir / "crops"

    # ---- 2. pages ----------------------------------------------------------
    with recorder.stage("pages"):
        page_images, prose = _collect_pages(
            resolved, bundle_dir, max_pages=max_pages, page_ranges=page_ranges
        )

    # ---- 3. segment --------------------------------------------------------
    with recorder.stage("segment"):
        table_pages = find_table_pages(page_images, bundle_dir / "source")
        compound_pages = [p for p in table_pages.values() if p.kind == "compound_table"]
        activity_pages = [p for p in table_pages.values() if p.kind == "activity_table"]
        # R17.1/R17.4 — TATR must not still be resident when MolScribe loads, or
        # peak RSS becomes the sum of the two models instead of the maximum.
        tatr.free()
        # R17.1/R17.4 — the detector must not still be resident when MolScribe
        # loads, or peak RSS becomes the sum of the two models.
        release(tatr)

    # ---- 4. ocr ------------------------------------------------------------
    with recorder.stage("ocr"):
        compound_cells = [
            cell
            for cell in _read_compound_table(compound_pages, crops_dir, resolved.source_mode)
            # Reconciling the detectors takes the union of their row boundaries,
            # which recovers real rows but also emits bands that are not compound
            # rows at all. Dropped here, before anything downstream pairs cells
            # with compounds by index.
            if is_compound_row(cell.compound_number, cell.name_raw)
        ]
        compound_cells = dedupe_compound_rows(compound_cells)
        activity_table_pages = _read_activity_pages(activity_pages)
        if not prose:
            prose = _prose_from_pdf(page_images, page_images)

    # ---- 5. name channel (batched OPSIN, JVM exits before torch loads) -----
    with recorder.stage("name_channel"):
        compounds, names_by_id = _build_compounds(
            compound_cells, resolved.pubnum, resolved.source_mode
        )

    # ---- 6. assay ----------------------------------------------------------
    with recorder.stage("assay"):
        legends, legend_anomalies = parse_legends(prose)
        anomalies.extend(legend_anomalies)
        anomalies.extend(detect_cross_reference_error(prose))
        measurements, activity_rows, assay_anomalies = _build_measurements(
            activity_table_pages,
            compounds,
            resolved.pubnum,
            legends,
            resolved.source_mode,
            off_target_assay,
        )
        anomalies.extend(assay_anomalies)
        anomalies.extend(detect_transcription_errors(measurements))

    # ---- 7. image channel (loaded last, freed immediately — R17.4) ---------
    with recorder.stage("image_channel"):
        if run_ocsr:
            _run_image_channel(compounds, compound_cells)

    # ---- 8. crosscheck -----------------------------------------------------
    with recorder.stage("crosscheck"):
        _apply_crosscheck_and_properties(compounds)

    # ---- 9. rank -----------------------------------------------------------
    with recorder.stage("rank"):
        examples = _examples_from_prose(prose)
        joined = join(compounds, activity_rows, examples, names=names_by_id)
        compounds = joined.compounds
        anomalies.extend(joined.anomalies)

        in_vivo_terms = detect_in_vivo(prose)
        claims_text = resolved.structured.claims_text if resolved.structured else ""
        for compound in compounds:
            apply_investment_signal(
                compound,
                in_examples=compound.in_examples,
                in_claims=_mentions(claims_text, compound.compound_local_id),
                in_prose=_mentions(prose, compound.compound_local_id),
                in_vivo_terms=in_vivo_terms,
            )
        rank_compounds(
            compounds, measurements, target=target_assay, off_target=off_target_assay
        )

    # ---- 10. write ---------------------------------------------------------
    with recorder.stage("write"):
        manifest = RunManifest(
            pubnum=resolved.pubnum,
            run_id=run_id,
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            source_mode=resolved.source_mode,
            n_pages=resolved.n_pages,
            n_compounds=len(compounds),
            n_measurements=len(measurements),
            target_assay=target_assay,
            off_target_assay=off_target_assay,
            legends=legends,
            anomalies=anomalies,
            versions=_versions(),
            page_rotations={
                str(p.page_no): getattr(p.rotation, "applied", 0) for p in table_pages.values()
            },
            **recorder.as_manifest_fields(),
        )
        write_bundle(manifest, compounds, measurements, anomalies, out_root)

    return PipelineResult(
        bundle_dir=bundle_dir,
        manifest=manifest,
        compounds=compounds,
        measurements=measurements,
        anomalies=anomalies,
    )


def _mentions(text: str, local_id: str) -> bool:
    if not text or not local_id:
        return False
    return re.search(rf"\bcompound\s+{re.escape(local_id)}\b", text, re.IGNORECASE) is not None


def _versions() -> dict[str, str]:
    from sarmine import __version__
    from sarmine.ocr.tesseract import tesseract_version

    versions = {"sarmine": __version__, "rdkit": rdkit_version()}
    try:
        versions["tesseract"] = tesseract_version()
    except Exception:
        pass
    return versions


def _collect_pages(
    resolved: ResolvedSource,
    bundle_dir: Path,
    *,
    max_pages: int | None,
    page_ranges: Sequence[tuple[int, int]] | None,
) -> tuple[dict[int, Path], str]:
    """Page rasters plus whatever prose the source already gives us for free."""
    pages_dir = bundle_dir / "source" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    wanted = _expand_ranges(page_ranges, resolved.n_pages or 10_000)
    prose = resolved.structured.description_text if resolved.structured else ""

    # PRD §7.2 — the PDF is the contract and the structured source is an
    # accelerator. Measured: the accelerator's value is its *text* (the legends,
    # the Examples and the claims arrive with no OCR at all), not its images.
    # Its `imgf` files are pre-cropped figures — up to three per page — so
    # treating the first one as a page raster fabricates table rows. When a PDF
    # is in hand the rasters come from it; the structured images are the page
    # source only when there is no PDF at all.
    if resolved.pdf_path is None and resolved.structured is not None and resolved.structured.image_paths:
        images = {
            page_no: paths[0]
            for page_no, paths in resolved.structured.image_paths.items()
            if paths
        }
        if wanted is not None:
            images = {p: path for p, path in images.items() if p in set(wanted)}
        elif max_pages is not None:
            images = dict(sorted(images.items())[:max_pages])
        # PRD §3.2 — the structured path also hands us the description prose,
        # so ~180 pages of synthesis-prose OCR simply disappears.
        return images, resolved.structured.description_text

    if resolved.pdf_path is None:
        return {}, prose

    if wanted is not None:
        pages: list[PdfPage] = []
        for page_no in wanted:
            pages.extend(extract_page_images(resolved.pdf_path, pages_dir, first=page_no, last=page_no))
    else:
        last = min(resolved.n_pages, max_pages) if max_pages else None
        pages = extract_page_images(resolved.pdf_path, pages_dir, last=last)
    return {p.page_no: p.path for p in pages}, prose


def is_compound_row(compound_number: int | None, name_raw: str) -> bool:
    """Is this detected band an actual compound row?

    Reconciling the two detectors takes the UNION of their row boundaries, which
    is what lifts compound-number recovery from 35/54 to 51/54 (AC-2.2) but also
    yields bands that are not compound rows. Without this filter a real run
    emitted 111 rows for a 54-compound patent.

    Asymmetric on purpose: a band with a name but no readable number is a real
    compound whose number failed to OCR, and PRD R11.4/R11.5 require keeping it
    and flagging the gap rather than dropping it or inventing a number.
    """
    if compound_number is not None:
        return True
    return _looks_like_a_chemical_name(name_raw or "")


def normalize_bin_cell(raw: str, legend_labels: Sequence[str]) -> str:
    """Read a letter-bin cell against its own legend's alphabet.

    Only collapses a cell that resolves to exactly one legend label; anything
    ambiguous or off-legend is returned verbatim so it reaches the review queue
    (§13.2) instead of being guessed at. Columns with no letter legend — numeric
    endpoints in other patents — are never touched.
    """
    text = (raw or "").strip()
    if not text or not legend_labels:
        return text

    labels = {label.upper() for label in legend_labels}
    if any(ch.isdigit() for ch in text):
        return text

    letters = [ch for ch in text if ch.isalpha()]
    found = {ch.upper() for ch in letters if ch.upper() in labels}
    if len(found) != 1:
        return text

    label = found.pop()
    for ch in letters:
        if ch.upper() == label:
            continue
        # A second UPPERCASE letter is a genuine competing reading and belongs in
        # the review queue. A stray lowercase one is the ruling line or a smudge:
        # measured on this document as `Cc` for `C` and `pA` for `A`.
        if ch.isupper():
            return text
    return label


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def dedupe_compound_rows(cells: Sequence[CompoundCell]) -> list[CompoundCell]:
    """Collapse the bands that the row-boundary union splits one row into.

    The compound number is the row identity here: two bands carrying the same
    number are one row seen twice, not two compounds. A numberless band whose
    name is a prefix of a kept row's name is a sliver of that row; a numberless
    band with its own distinct name is a real compound whose number failed to
    OCR, and R11.5 says to keep it and flag the gap rather than drop it.

    The surviving band takes the fullest name and whichever crop exists, so a
    split that put the name in one band and the drawing in another does not
    silently lose the image channel for that compound.
    """
    # Keyed by (page, number), never number alone: a split band is always within
    # one page, whereas the same number on two pages means a digit was dropped.
    # Merging those deletes a real compound — measured, when page 66's compound
    # 11 read as `1` and was absorbed into compound 1 from page 61. A duplicate
    # number reaching the review queue is the far cheaper failure (EC-4, EC-24).
    winners: dict[tuple[int, int], CompoundCell] = {}
    order: list[tuple[int, int]] = []
    numberless: list[CompoundCell] = []

    for cell in cells:
        number = cell.compound_number
        if number is None:
            numberless.append(cell)
            continue
        key = (cell.page_no, number)
        existing = winners.get(key)
        if existing is None:
            winners[key] = cell
            order.append(key)
            continue
        winners[key] = _merge_rows(existing, cell)

    kept = [winners[k] for k in order]
    kept_names = [_name_key(c.name_raw) for c in kept]
    for cell in numberless:
        key = _name_key(cell.name_raw)
        if key and any(key in other or other in key for other in kept_names if other):
            continue
        kept.append(cell)
        kept_names.append(key)
    return kept


def _merge_rows(a: CompoundCell, b: CompoundCell) -> CompoundCell:
    winner, loser = (a, b) if len(a.name_raw or "") >= len(b.name_raw or "") else (b, a)
    return CompoundCell(
        page_no=winner.page_no,
        compound_number=winner.compound_number,
        name_raw=winner.name_raw,
        structure_crop=winner.structure_crop or loser.structure_crop,
        provenance=[*winner.provenance, *loser.provenance],
    )


def _build_compounds(
    cells: Sequence[CompoundCell], pubnum: str, source_mode: str
) -> tuple[list[Compound], dict[str, str]]:
    """R9.2 — one batched OPSIN call for every name, then homoglyph repair.

    Returns exactly one `Compound` per input cell, in order: the caller pairs the
    two lists by index when attaching OCSR results, so this must not filter.
    """
    raw_names = [clean_ocr_name(c.name_raw) for c in cells]
    results = parse_names(raw_names)

    failed_idx = [i for i, r in enumerate(results) if r.smiles is None and raw_names[i].strip()]
    repairs: dict[int, object] = {}
    if failed_idx:
        repaired = repair_batch(
            [raw_names[i] for i in failed_idx],
            lambda batch: [r.smiles for r in parse_names(batch)],
        )
        repairs = dict(zip(failed_idx, repaired))

    compounds: list[Compound] = []
    names_by_id: dict[str, str] = {}
    for idx, cell in enumerate(cells):
        result = results[idx]
        smiles = result.smiles
        repair_note = None
        repair = repairs.get(idx)
        if smiles is None and repair is not None and getattr(repair, "smiles", None):
            smiles = repair.smiles
            repair_note = repair.substitution

        local_id = str(cell.compound_number) if cell.compound_number is not None else f"p{cell.page_no}r{idx}"
        compound = Compound(
            compound_id=f"{pubnum}:{local_id}",
            compound_local_id=local_id,
            compound_number=cell.compound_number,
            smiles_from_name=smiles,
            inchikey_from_name=smiles_to_inchikey(smiles) if smiles else None,
            opsin_status=result.status,
            opsin_ambiguous=result.ambiguous,
            homoglyph_repair_applied=repair_note,
            provenance=cell.provenance,
            rdkit_version=rdkit_version(),
        )
        compounds.append(compound)
        if raw_names[idx]:
            names_by_id[compound.compound_id] = raw_names[idx]
    return compounds, names_by_id


def _build_measurements(
    table_pages: Sequence[TablePage],
    compounds: Sequence[Compound],
    pubnum: str,
    legends: dict,
    source_mode: str,
    off_target_assay: str | None,
) -> tuple[list[Measurement], list[ActivityRow], list[DocumentAnomaly]]:
    """R8.5 / EC-3 — stitch the continuation page before reading any values."""
    measurements: list[Measurement] = []
    activity_rows: list[ActivityRow] = []
    anomalies: list[DocumentAnomaly] = []
    by_number = {c.compound_number: c for c in compounds if c.compound_number is not None}

    for table in stitch_tables(list(table_pages)):
        anomalies.extend(table.anomalies)
        matches = _match_headers(table.header)
        number_col = next(
            (i for i, h in enumerate(table.header) if _NUMBER_HEADER.search(h or "")), 0
        )

        seen_numbers: set[int] = set()
        last_number: int | None = None

        for row in table.rows:
            texts = list(row.texts)
            number = _as_int(texts[number_col]) if number_col < len(texts) else None
            if number is not None and not accept_compound_number(number, seen_numbers, last_number):
                anomalies.append(
                    DocumentAnomaly(
                        kind="compound_number_gap",
                        severity="warning",
                        message=(
                            f"activity row on page {row.page_no} read compound number "
                            f"{number}, which repeats or precedes {last_number}; the row is "
                            "flagged for review and its values are not attributed"
                        ),
                    )
                )
                number = None
            if number is not None:
                seen_numbers.add(number)
                last_number = number

            values = {
                table.header[i]: texts[i]
                for i in range(min(len(table.header), len(texts)))
                if i != number_col
            }
            activity_rows.append(
                ActivityRow(compound_number=number, values=values, page_no=row.page_no)
            )
            compound = by_number.get(number) if number is not None else None
            if compound is None:
                continue

            for col, header_match in enumerate(matches):
                if header_match is None or col == number_col or col >= len(texts):
                    continue
                cell_text = normalize_bin_cell(
                    texts[col], _legend_labels(legends, header_match)
                )
                if not cell_text.strip():
                    # EC-7 — a blank cell is a gap, never a low value.
                    continue
                cell = row.cells[col] if col < len(row.cells) else None
                provenance = Provenance(
                    page_no=row.page_no,
                    bbox=cell.bbox if cell is not None else (0, 0, 0, 0),
                    raster_width=0,
                    raster_height=0,
                    crop_path=f"crops/p{row.page_no:03d}_activity_{col}.png",
                    source=source_mode,
                    extractor="tesseract",
                )
                measurement = build_measurement(
                    compound_id=compound.compound_id,
                    header=header_match,
                    raw_cell=cell_text,
                    provenance=provenance,
                    assay_group_key=assay_group_key(pubnum, header_match.published_type),
                    legends=legends,
                    is_off_target=bool(
                        off_target_assay
                        and header_match.target
                        and header_match.target.upper() == off_target_assay.upper()
                    ),
                )
                if measurement is None:
                    anomalies.append(
                        DocumentAnomaly(
                            kind="unit_missing",
                            severity="warning",
                            message=(
                                f"{header_match.published_type!r} value {cell_text!r} for "
                                f"compound {number} could not be standardized; queued for review"
                            ),
                            provenance=provenance,
                        )
                    )
                    continue
                measurement.measurement_id = (
                    f"{compound.compound_id}:{header_match.published_type}"
                )
                measurements.append(measurement)
    return measurements, activity_rows, anomalies


def accept_compound_number(number: int | None, seen: set[int], last: int | None) -> bool:
    """Is this row's compound number trustworthy within its logical table?

    PRD R11.5 / EC-4 — a compound number is flagged, never repaired. A row on
    page 187 of the reference patent mis-reads its number as `4`, which already
    appeared on page 186; accepting it silently overwrote compound 4's real
    activity values with another compound's. Both the repeat and the backwards
    jump are refused, and the row goes to review instead.
    """
    if number is None:
        return False
    if number in seen:
        return False
    return last is None or number > last


def _legend_labels(legends: dict, header: HeaderMatch) -> list[str]:
    """The legal bin labels for one assay column, or [] if it is not binned.

    Legend keys come from prose ("WIZ EC50") and headers from the table
    ("WIZ ECS50 (uM)" after OCR), so they are matched loosely on the target.
    """
    if not legends:
        return []
    target = (header.target or "").upper()
    published = (header.published_type or "").upper()
    for assay, definitions in legends.items():
        key = assay.upper()
        if (target and target in key) or key in published:
            return [d.label for d in definitions if getattr(d, "label", None)]
    return []


def _as_int(text: str) -> int | None:
    digits = re.findall(r"\d+", text or "")
    return int(digits[0]) if digits else None


def _run_image_channel(compounds: list[Compound], cells: Sequence[CompoundCell]) -> None:
    """R8.6/R8.7 then R9.7–R9.12: classify, then OCSR only the clean crops."""
    from sarmine.segment.classify import classify_crops, route

    # Carry the compound object alongside its crop rather than an index into a
    # list built elsewhere. An index that outlived its list cost a whole run.
    pairs = [
        (compound, Path(cell.structure_crop))
        for compound, cell in zip(compounds, cells)
        if cell.structure_crop is not None and Path(cell.structure_crop).is_file()
    ]
    if not pairs:
        return

    labels = classify_crops([p for _, p in pairs])
    to_ocsr: list[tuple[Compound, Path]] = []
    for (compound, path), (label, _conf) in zip(pairs, labels):
        if route(label) == "ocsr":
            to_ocsr.append((compound, path))
        elif label == "markush":
            # R8.7 — no OCSR and no cross-check: OCSR would hallucinate a concrete
            # structure and attach it to a real activity value.
            compound.markush_detected = True

    if not to_ocsr:
        return

    from sarmine.structure.molscribe import MolScribeRunner

    runner = MolScribeRunner()
    try:
        results = runner.predict([p for _, p in to_ocsr])
    except Exception:
        results = []
    finally:
        release(runner)

    for (compound, _path), result in zip(to_ocsr, results):
        compound.smiles_from_image = result.smiles
        compound.inchikey_from_image = result.inchikey
        compound.ocsr_confidence_molecule = result.confidence_molecule
        compound.ocsr_confidence_min_atom = result.confidence_min_atom
        compound.ocsr_confidence_min_bond = result.confidence_min_bond


def _apply_crosscheck_and_properties(compounds: Sequence[Compound]) -> None:
    for compound in compounds:
        standardized_name = standardize_smiles(compound.smiles_from_name)
        standardized_image = (
            standardize_smiles(compound.smiles_from_image)
            if compound.smiles_from_image and not compound.markush_detected
            else None
        )
        key_name = standardized_name.inchikey_full or compound.inchikey_from_name
        key_image = standardized_image.inchikey_full if standardized_image else None
        compound.inchikey_from_name = key_name
        compound.inchikey_from_image = key_image

        result = crosscheck(
            key_name,
            None if compound.markush_detected else key_image,
            standardized_name.smiles or compound.smiles_from_name,
            None if compound.markush_detected else compound.smiles_from_image,
        )
        compound.crosscheck_tier = result.tier
        compound.smiles_final = result.smiles_final
        compound.structure_source = result.structure_source
        compound.inchikey_full = result.inchikey_full
        compound.inchikey_skeleton = result.inchikey_skeleton
        compound.smiles_tautomer_canonical = standardized_name.smiles_tautomer_canonical
        compound.has_undefined_stereocenters = standardized_name.has_undefined_stereocenters
        compound.standardization_skipped = standardized_name.standardization_skipped

        for key, value in compute_properties(compound.smiles_final).items():
            setattr(compound, key, value)


if __name__ == "__main__":  # PRD R9.10 — MolScribe's DataLoader re-executes this file
    import sys

    from sarmine.cli import main

    sys.exit(main())
