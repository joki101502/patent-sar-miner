"""Full extraction of the reference patent, for measurement (PRD §19, AC-9.4).

Run as a FILE, never piped through stdin: MolScribe's post-processing used to
re-import `__main__`, and a `<stdin>` entry point cannot be re-imported.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    from sarmine.pipeline import run_pipeline

    started = time.time()

    def progress(stage: str, phase: str, index: int, total: int) -> None:
        if phase == "start":
            print(f"[{time.time() - started:6.1f}s] stage {index}/{total}: {stage}", flush=True)

    result = run_pipeline(
        REPO / "data" / "patents" / "WO2024097932A1.pdf",
        out_root=REPO / "artifacts",
        allow_network="--offline" not in sys.argv,
        run_ocsr="--no-ocsr" not in sys.argv,
        page_ranges=[(61, 88), (182, 187)],
        on_progress=progress,
        run_id="fullrun_structured" if "--offline" not in sys.argv else "fullrun",
    )

    print(f"\nTOTAL {time.time() - started:.1f}s -> {result.bundle_dir}")
    print(f"compounds={len(result.compounds)} measurements={len(result.measurements)}")
    print("peak rss mb:", result.manifest.stage_peak_rss_mb)
    print("timings s:", {k: round(v, 1) for k, v in result.manifest.stage_timings_s.items()})

    numbers = sorted(c.compound_number for c in result.compounds if c.compound_number)
    print(f"compound numbers ({len(numbers)}):", numbers)
    print("name channel:", sum(1 for c in result.compounds if c.smiles_from_name))
    print("image channel:", sum(1 for c in result.compounds if c.smiles_from_image))
    print("tiers:", dict(Counter(c.crosscheck_tier for c in result.compounds)))
    print("anomalies:", dict(Counter(a.kind for a in result.anomalies)))
    return 0


if __name__ == "__main__":  # PRD R9.10
    sys.exit(main())
