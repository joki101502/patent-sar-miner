"""Morphological ruling-line detection and cell construction (PRD R8.2, Plan 4.1).

This is the highest-leverage module in the system. PRD §8.1: whole-page OCR of
Table 1 made OPSIN parse **0 of 61** names, because atom labels from the
structure drawing interleave into the name text. OCR'ing only the name sub-cell
made it parse **33 of 37**. Same page, same OCR engine, same parser.

Also implements column-role assignment (Plan 4.2) and the compound-number
sequence check (PRD R8.4, EC-4): gaps are flagged and never interpolated.

Import-safe: no model loading and no side effects at import time (PRD §17.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from sarmine.artifacts.schema import DocumentAnomaly

Detector = Literal["morphology", "tatr", "reconciled"]
ColumnRole = Literal["number", "structure", "name"]

# PRD R8.2 — measured on this document; not tuning knobs. Morphology's misses
# are the second detector's job (`tatr.py` + `reconcile.py`), not a reason to
# move these values.
PROJECTION_THRESHOLD = 0.4
GROUPING_GAP_PX = 8
MIN_KERNEL_PX = 30
KERNEL_DIVISOR = 12


@dataclass(frozen=True)
class Cell:
    """One table cell in the image's pixel space."""

    row: int
    col: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Grid:
    """A detected table structure, with the detector that produced it.

    `n_rows` / `n_cols` may be passed explicitly or left to be derived from the
    rulings, so callers can build a grid either way.
    """

    n_rows: int = 0
    n_cols: int = 0
    cells: list[Cell] = field(default_factory=list)
    y_rulings: list[int] = field(default_factory=list)
    x_rulings: list[int] = field(default_factory=list)
    width: int = 0
    height: int = 0
    detector: Detector = "morphology"

    def __post_init__(self) -> None:
        if not self.n_rows:
            self.n_rows = max(len(self.y_rulings) - 1, 0)
        if not self.n_cols:
            self.n_cols = max(len(self.x_rulings) - 1, 0)

    def cell(self, row: int, col: int) -> Cell | None:
        for candidate in self.cells:
            if candidate.row == row and candidate.col == col:
                return candidate
        return None

    def row_cells(self, row: int) -> list[Cell]:
        return sorted((c for c in self.cells if c.row == row), key=lambda c: c.col)

    def column_widths(self) -> list[int]:
        return [b - a for a, b in zip(self.x_rulings, self.x_rulings[1:])]

    def row_heights(self) -> list[int]:
        return [b - a for a, b in zip(self.y_rulings, self.y_rulings[1:])]

    @property
    def completeness(self) -> float:
        """Fraction of the row × column lattice that actually produced a cell."""
        expected = self.n_rows * self.n_cols
        return len(self.cells) / expected if expected else 0.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == bool:
        image = np.where(image, 255, 0)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _group(indices: list[int], gap: int = GROUPING_GAP_PX) -> list[int]:
    """Collapse runs of adjacent indices into one coordinate each.

    Truncating the run mean rather than rounding it is what reproduces spike
    S2's recorded rulings for page 63 exactly; a 2-px-thick line at 200 dpi
    otherwise lands a pixel high.
    """
    if not indices:
        return []
    groups: list[int] = []
    current = [indices[0]]
    for value in indices[1:]:
        if value - current[-1] <= gap:
            current.append(value)
        else:
            groups.append(int(sum(current) / len(current)))
            current = [value]
    groups.append(int(sum(current) / len(current)))
    return groups


def find_rulings(image: np.ndarray) -> tuple[list[int], list[int]]:
    """Detect horizontal and vertical ruling lines (PRD R8.2).

    Returns `(y_rulings, x_rulings)`, both sorted ascending, in the image's
    pixel space. Input may be grayscale, colour or binary; dark pixels are ink.
    """
    gray = _to_gray(image)
    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return [], []

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(MIN_KERNEL_PX, width // KERNEL_DIVISOR), 1)
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(MIN_KERNEL_PX, height // KERNEL_DIVISOR))
    )

    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)

    def peaks(projection: np.ndarray) -> list[int]:
        peak = float(projection.max())
        if peak <= 0:
            return []
        hits = [int(i) for i in np.flatnonzero(projection > PROJECTION_THRESHOLD * peak)]
        return _group(hits)

    y_rulings = peaks(horizontal.sum(axis=1) / 255.0)
    x_rulings = peaks(vertical.sum(axis=0) / 255.0)
    return sorted(y_rulings), sorted(x_rulings)


def cells_from_rulings(
    y_rulings: list[int], x_rulings: list[int], *, min_px: int = 2
) -> list[Cell]:
    """Intersect row bands with column bands to produce addressable cells."""
    cells: list[Cell] = []
    for row, (y0, y1) in enumerate(zip(y_rulings, y_rulings[1:])):
        for col, (x0, x1) in enumerate(zip(x_rulings, x_rulings[1:])):
            if (y1 - y0) < min_px or (x1 - x0) < min_px:
                continue
            cells.append(Cell(row=row, col=col, bbox=(int(x0), int(y0), int(x1), int(y1))))
    return cells


# The same function under the name used by the segmentation contract.
build_cells = cells_from_rulings


def load_gray(image_path: Path | str) -> np.ndarray:
    """Read a page as grayscale. Raises if the file is missing or unreadable."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def detect_grid(image_path: Path | str) -> Grid:
    """Full morphological pipeline for one page: load, binarize, rule, cell.

    Takes and returns paths and coordinates only — the decoded page is dropped
    on return, because 223 retained pages would be ~1.9 GB (PRD R17.3).
    """
    gray = load_gray(image_path)
    height, width = gray.shape[:2]
    y_rulings, x_rulings = find_rulings(gray)
    return Grid(
        y_rulings=y_rulings,
        x_rulings=x_rulings,
        cells=cells_from_rulings(y_rulings, x_rulings),
        width=width,
        height=height,
        detector="morphology",
    )


def _alpha_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def assign_column_roles(
    grid: Grid,
    ocr_by_col: dict[int, str],
    column_ink: dict[int, float] | None = None,
) -> dict[int, ColumnRole]:
    """Re-export of `sarmine.segment.roles.assign_column_roles` (Plan 4.2).

    Kept as an alias so both import paths work. The single implementation lives
    in `roles.py`; imported lazily because `roles` imports `Grid` from here.
    """
    from sarmine.segment.roles import assign_column_roles as _assign

    return _assign(grid, ocr_by_col, column_ink)


# Digit homoglyphs seen in Tesseract output on this document; the name-channel
# form of the same failure is PRD R9.4.
_DIGIT_HOMOGLYPHS = str.maketrans({"l": "1", "I": "1", "|": "1", "O": "0", "o": "0"})


def parse_compound_numbers(texts: list[str]) -> list[int | None]:
    """Parse digit-only OCR from number cells; `None` when unreadable."""
    numbers: list[int | None] = []
    for text in texts:
        cleaned = (text or "").strip().translate(_DIGIT_HOMOGLYPHS)
        match = re.search(r"\d+", cleaned)
        numbers.append(int(match.group()) if match else None)
    return numbers


def _gap_anomaly(message: str) -> DocumentAnomaly:
    return DocumentAnomaly(kind="compound_number_gap", severity="warning", message=message)


def validate_number_sequence(numbers: list[int | None]) -> list[DocumentAnomaly]:
    """Check the compound-number sequence (PRD R8.4, EC-4, AC-5.4).

    Returns anomalies only. It never returns a repaired sequence and never
    mutates its argument: an invented compound number silently corrupts the
    join, which is the worst failure this component can produce.
    """
    anomalies: list[DocumentAnomaly] = []

    for index, value in enumerate(numbers):
        if value is None:
            anomalies.append(
                _gap_anomaly(
                    f"Compound number at position {index} is unreadable; left unknown "
                    "rather than interpolated (PRD EC-4)."
                )
            )

    readable = [(i, v) for i, v in enumerate(numbers) if v is not None]
    for (prev_index, previous), (index, current) in zip(readable, readable[1:]):
        unreadable_between = index - prev_index - 1
        if current == previous:
            anomalies.append(
                _gap_anomaly(f"Duplicate compound number {current} at position {index}.")
            )
        elif current < previous:
            anomalies.append(
                _gap_anomaly(
                    f"Compound numbers are not monotonic at position {index}: "
                    f"{previous} is followed by {current}."
                )
            )
        elif current - previous - 1 > unreadable_between:
            # A jump already explained by intervening unreadable cells is
            # reported once, as the unreadable cell — not twice.
            missing = ", ".join(str(n) for n in range(previous + 1, current))
            anomalies.append(
                _gap_anomaly(
                    f"Gap in the compound number sequence between {previous} and "
                    f"{current} (missing {missing}); flagged, not interpolated (PRD EC-4)."
                )
            )

    return anomalies


# Alias kept so both call sites in the pipeline keep resolving.
validate_monotonic = validate_number_sequence
