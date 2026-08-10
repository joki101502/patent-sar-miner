"""Stage timing and peak-RSS instrumentation (Plan Part 12.2, PRD R17.6).

PRD §17.1's pipeline order is a MEMORY requirement, not a style preference: peak
memory must equal the maximum stage, not the sum of stages. Sequential peak is
~1.3 GB (MolScribe) + ~0.3 GB (Streamlit) ~= 1.6 GB, against ~2.5 GB+ if the
stages ran concurrently, on a host with 2.7 GB.

R17.6 makes that budget a MEASURED value rather than an assumption, by recording
RSS at every stage boundary into `manifest.stage_peak_rss_mb` (AC-9.2, AC-9.3).
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator

# PRD AC-9.3 — peak RSS must stay under 2.4 GB on the reference patent.
PEAK_RSS_BUDGET_MB = 2400.0

# PRD §17.1 — the normative stage order. Named here so the pipeline and the UI
# progress bar agree on what the stages are and in what order they run.
STAGE_ORDER = (
    "resolve",
    "pages",
    "segment",
    "ocr",
    "name_channel",
    "assay",
    "image_channel",
    "crosscheck",
    "rank",
    "write",
)

ProgressCallback = Callable[[str, str, int, int], None]


def rss_mb() -> float:
    """Peak resident set size in megabytes.

    `ru_maxrss` is BYTES on macOS but KILOBYTES on Linux. Using one conversion on
    both platforms misreports the memory budget by 1000x, which would make the
    whole R17.6 verification worthless.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


class StageRecorder:
    """Records wall time and peak RSS per pipeline stage, and drives the UI's
    per-stage progress (PRD R14.1 — the run is blocking with a progress bar)."""

    def __init__(
        self,
        on_progress: ProgressCallback | None = None,
        stages: tuple[str, ...] = STAGE_ORDER,
    ) -> None:
        self.timings: dict[str, float] = {}
        self.peak_rss: dict[str, float] = {}
        self.stages = stages
        self._on_progress = on_progress
        self._index = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._index += 1
        total = max(len(self.stages), self._index)
        self._emit(name, "start", self._index, total)
        started = time.perf_counter()
        try:
            yield
        finally:
            # Recorded in `finally` because a stage that blew up is precisely the
            # one whose memory and duration you want in the manifest.
            self.timings[name] = time.perf_counter() - started
            self.peak_rss[name] = rss_mb()
            self._emit(name, "end", self._index, total)

    def _emit(self, name: str, phase: str, index: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(name, phase, index, total)

    def peak_overall(self) -> float:
        return max(self.peak_rss.values(), default=0.0)

    def over_budget(self, budget_mb: float = PEAK_RSS_BUDGET_MB) -> bool:
        return self.peak_overall() > budget_mb

    def total_seconds(self) -> float:
        return sum(self.timings.values())

    def as_manifest_fields(self) -> dict[str, dict[str, float]]:
        return {
            "stage_timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "stage_peak_rss_mb": {k: round(v, 1) for k, v in self.peak_rss.items()},
        }


def release(*objects: object) -> None:
    """Drop references and collect, between stages.

    PRD R17.4 — MolScribe is loaded last and freed immediately, so RDKit does not
    stack on top of ~1.3 GB of resident torch.
    """
    for obj in objects:
        free = getattr(obj, "free", None)
        if callable(free):
            free()
    gc.collect()
