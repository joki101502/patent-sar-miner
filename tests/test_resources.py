"""Part 12.2 — stage instrumentation (PRD R17.6, AC-9.2, AC-9.3).

The memory budget must be a MEASURED value in the logs, not an assumption. Peak
memory must equal the maximum stage, not the sum of stages (PRD §17.1), which is
only checkable if every stage boundary is recorded.
"""

from __future__ import annotations

import sys

import pytest

from sarmine.resources import (
    PEAK_RSS_BUDGET_MB,
    StageRecorder,
    rss_mb,
)


def test_rss_is_reported_in_megabytes_on_this_platform():
    """The units differ by platform: macOS ru_maxrss is BYTES, Linux is KILOBYTES.
    Getting this wrong misreports the budget by 1000x in either direction."""
    value = rss_mb()
    assert value > 0
    # Any live Python process is over 5 MB and a test run is nowhere near 100 GB.
    assert 5 < value < 100_000, f"implausible RSS {value} MB on {sys.platform}"


def test_recorder_captures_timing_and_peak_rss_per_stage():
    rec = StageRecorder()
    with rec.stage("resolve"):
        pass
    with rec.stage("ocr"):
        pass

    assert set(rec.timings) == {"resolve", "ocr"}
    assert set(rec.peak_rss) == {"resolve", "ocr"}
    assert all(v >= 0 for v in rec.timings.values())
    assert all(v > 0 for v in rec.peak_rss.values())


def test_stage_is_recorded_even_when_it_raises():
    """A stage that blows up is exactly the one whose memory you want to see."""
    rec = StageRecorder()
    with pytest.raises(RuntimeError):
        with rec.stage("ocsr"):
            raise RuntimeError("boom")

    assert "ocsr" in rec.peak_rss
    assert "ocsr" in rec.timings


def test_peak_is_the_max_stage_not_the_sum():
    """PRD §17.1 — the pipeline order is a MEMORY requirement, not a style
    preference. Sequential peak ~1.6 GB vs ~2.5 GB+ if stages ran concurrently."""
    rec = StageRecorder()
    rec.peak_rss.update({"ocr": 400.0, "segment": 300.0, "ocsr": 1300.0, "rank": 200.0})

    assert rec.peak_overall() == 1300.0
    assert rec.peak_overall() < sum(rec.peak_rss.values())


def test_budget_check_flags_an_over_budget_run():
    """PRD AC-9.3 — peak RSS must stay under 2.4 GB."""
    rec = StageRecorder()
    rec.peak_rss.update({"ocsr": PEAK_RSS_BUDGET_MB + 1})
    assert rec.over_budget() is True

    ok = StageRecorder()
    ok.peak_rss.update({"ocsr": PEAK_RSS_BUDGET_MB - 1})
    assert ok.over_budget() is False


def test_recorder_serializes_into_the_manifest_fields():
    """PRD §15.4 — these land in manifest.stage_timings_s / stage_peak_rss_mb."""
    rec = StageRecorder()
    with rec.stage("resolve"):
        pass

    assert isinstance(rec.as_manifest_fields()["stage_timings_s"]["resolve"], float)
    assert isinstance(rec.as_manifest_fields()["stage_peak_rss_mb"]["resolve"], float)


def test_progress_callback_receives_each_stage():
    """PRD R14.1 — the Streamlit run is blocking with per-stage status, so the
    recorder is also what drives the progress bar."""
    seen: list[tuple[str, str]] = []
    rec = StageRecorder(on_progress=lambda name, phase, _i, _n: seen.append((name, phase)))

    with rec.stage("resolve"):
        pass
    with rec.stage("ocr"):
        pass

    assert seen == [
        ("resolve", "start"),
        ("resolve", "end"),
        ("ocr", "start"),
        ("ocr", "end"),
    ]
