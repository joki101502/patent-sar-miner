"""Reconcile the two table-structure detectors (PRD R8.3, EC-26, Plan 4.4).

Morphology found only 37 of 54 compound-table cells on the reference patent
(PRD §8.1), so a second detector runs and the two are compared here.

**Disagreement between the detectors is a review-queue trigger, and is a more
reliable uncertainty signal than either model's self-reported confidence**
(PRD R8.3). When they disagree, EC-26 says to prefer the detector with the more
complete grid rather than to guess a merge.

Import-safe: pure geometry, no models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from sarmine.artifacts.schema import DocumentAnomaly
from sarmine.segment.rulings import GROUPING_GAP_PX, Cell, Grid, cells_from_rulings

# Fraction of cells that must match across detectors before the grids count as
# agreeing. Below this the page goes to the review queue (PRD §13.2, medium).
AGREEMENT_THRESHOLD = 0.9


@dataclass
class ReconcileResult:
    """The chosen grid plus everything the review queue needs to judge it.

    `cells` is the chosen grid's cell list, and `n_morph` / `n_tatr` are the two
    detectors' cell counts, so a caller can see how far apart they were without
    holding on to both grids.
    """

    grid: Grid
    disagreement: bool = False
    notes: list[str] = field(default_factory=list)
    anomalies: list[DocumentAnomaly] = field(default_factory=list)
    n_morph: int = 0
    n_tatr: int = 0

    @property
    def cells(self) -> list[Cell]:
        return self.grid.cells

    @property
    def detail(self) -> str:
        return "; ".join(self.notes)


def cell_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two `(x0, y0, x1, y1)` boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    area_a = max(ax1 - ax0, 0) * max(ay1 - ay0, 0)
    area_b = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = inter_w * inter_h
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def merge_row_boundaries(
    columns_from: Grid,
    rows_from: Grid | None,
    *,
    gap_px: int = GROUPING_GAP_PX,
    min_band_px: int = 0,
) -> Grid:
    """Keep one detector's columns and take the union of both detectors' rows.

    Measured on the reference patent's Table 1 (PDF pages 61-88): the two
    detectors fail in opposite directions. Morphology recovers the three column
    rules on every page but misses row separators, finding 37 of 54 compound
    cells — spike S2's number. TATR finds the missing row separators but
    over-segments the columns into five to ten. Taking rows from both and
    columns from morphology recovers all 54.

    Boundaries closer than `gap_px` are the same rule seen twice; bands thinner
    than `min_band_px` are detector noise, not a table row.
    """
    if rows_from is None:
        return columns_from

    merged: list[int] = []
    for edge in sorted(list(columns_from.y_rulings) + list(rows_from.y_rulings)):
        if merged and edge - merged[-1] <= gap_px:
            merged[-1] = (merged[-1] + edge) // 2
        else:
            merged.append(edge)

    if min_band_px > 0:
        kept = [merged[0]] if merged else []
        for edge in merged[1:]:
            if edge - kept[-1] >= min_band_px:
                kept.append(edge)
        merged = kept

    return Grid(
        y_rulings=merged,
        x_rulings=list(columns_from.x_rulings),
        cells=cells_from_rulings(merged, list(columns_from.x_rulings)),
        width=columns_from.width,
        height=columns_from.height,
        detector="reconciled",
    )


def _agreement(morph: Grid, tatr: Grid, iou_threshold: float) -> float:
    """Fraction of the larger grid's cells that have a counterpart."""
    denominator = max(len(morph.cells), len(tatr.cells))
    if denominator == 0:
        return 0.0
    matched = sum(
        1
        for cell in morph.cells
        if any(cell_iou(cell.bbox, other.bbox) >= iou_threshold for other in tatr.cells)
    )
    return matched / denominator


def _prefer(morph: Grid, tatr: Grid, expected_rows: int | None) -> tuple[Grid, str]:
    """EC-26 — pick the detector with the more complete grid."""
    if expected_rows is not None:
        morph_error = abs(morph.n_rows - expected_rows)
        tatr_error = abs(tatr.n_rows - expected_rows)
        if morph_error != tatr_error:
            winner = morph if morph_error < tatr_error else tatr
            return winner, (
                f"preferred {winner.detector}: {winner.n_rows} rows is closest to the "
                f"expected {expected_rows}"
            )

    if len(morph.cells) != len(tatr.cells):
        winner = morph if len(morph.cells) > len(tatr.cells) else tatr
        return winner, (
            f"preferred {winner.detector}: more complete grid "
            f"({len(morph.cells)} morphology cells vs {len(tatr.cells)} tatr cells)"
        )

    return morph, "detectors are equally complete; kept the morphology grid"


def reconcile(
    morph: Grid | None,
    tatr: Grid | None,
    *,
    iou_threshold: float = 0.5,
    expected_rows: int | None = None,
    merge_rows: bool = True,
) -> ReconcileResult:
    """Combine the morphological and TATR grids into one decision.

    Either input may be None — a missing second opinion is normal (TATR is an
    optional dependency) and is not itself a disagreement.

    With `merge_rows`, a TATR grid that over-segments columns still contributes
    its row separators instead of being discarded outright (see below).
    """
    if morph is None and tatr is None:
        raise ValueError("reconcile() needs at least one grid")

    if tatr is None:
        assert morph is not None
        return ReconcileResult(
            grid=morph,
            notes=["no TATR grid; used morphology alone"],
            n_morph=len(morph.cells),
        )

    if morph is None:
        return ReconcileResult(
            grid=tatr,
            notes=["no morphology grid; used TATR alone"],
            n_tatr=len(tatr.cells),
        )

    notes: list[str] = []
    anomalies: list[DocumentAnomaly] = []

    shape_matches = (morph.n_rows, morph.n_cols) == (tatr.n_rows, tatr.n_cols)
    agreement = _agreement(morph, tatr, iou_threshold)
    disagreement = not shape_matches or agreement < AGREEMENT_THRESHOLD

    if merge_rows and morph.n_cols >= 2 and morph.n_cols < tatr.n_cols:
        # Measured on Table 1 (structured-source pages 61-88): the detectors fail
        # in OPPOSITE directions, so picking a winner throws away the half that
        # each one got right. Morphology reads the column rules on every page but
        # misses faint row separators (35 of 54 compound numbers recovered);
        # TATR finds those separators but over-segments columns (3 -> 4+). Taking
        # morphology's columns with the union of both row sets recovers 51 of 54.
        chosen = merge_row_boundaries(morph, tatr)
        reason = (
            f"merged: morphology columns ({morph.n_cols}) with the union of both row "
            f"sets ({morph.n_rows} + {tatr.n_rows} -> {chosen.n_rows})"
        )
    else:
        chosen, reason = _prefer(morph, tatr, expected_rows)
    notes.append(reason)
    notes.append(
        f"cell agreement {agreement:.2f} at IoU>={iou_threshold:.2f}; "
        f"morphology {morph.n_rows}x{morph.n_cols}, tatr {tatr.n_rows}x{tatr.n_cols}"
    )

    if disagreement:
        anomalies.append(
            DocumentAnomaly(
                kind="detector_disagreement",
                severity="warning",
                message=(
                    "Table structure detectors disagree: morphology found "
                    f"{morph.n_rows}x{morph.n_cols} ({len(morph.cells)} cells), TATR found "
                    f"{tatr.n_rows}x{tatr.n_cols} ({len(tatr.cells)} cells); cell agreement "
                    f"{agreement:.2f}. {reason} (PRD R8.3 / EC-26)."
                ),
            )
        )

    return ReconcileResult(
        grid=replace(chosen, detector="reconciled"),
        disagreement=disagreement,
        notes=notes,
        anomalies=anomalies,
        n_morph=len(morph.cells),
        n_tatr=len(tatr.cells),
    )
