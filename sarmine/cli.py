"""Command-line surface (PRD §16). There is no REST API.

    sarmine run <pdf-or-pubnum> [--out artifacts/] [--force-pdf-path]
                                [--target WIZ] [--off-target ZBTB7A]
    sarmine evaluate <run-dir> --gold gold/WO2024097932A1.gold.json
    sarmine slim-checkpoint

⚠️ Every entry point is guarded by `if __name__ == "__main__":` (PRD R9.10).
MolScribe's DataLoader re-executes the entry script; without the guard the
process forks repeatedly and never completes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sarmine.config import get_config


def build_parser() -> argparse.ArgumentParser:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="sarmine",
        description="Mine a chemistry patent into a joined SAR table with provenance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="extract one patent into an artifact bundle")
    run.add_argument("source", help="path to a patent PDF, or a publication number")
    run.add_argument("--out", default=str(cfg.artifact_root), help="artifact root directory")
    run.add_argument(
        "--force-pdf-path",
        action="store_true",
        help="skip patents.google.com and use the full PDF OCR path (PRD R7.1)",
    )
    run.add_argument("--target", default=cfg.target_assay, help="intended target assay")
    run.add_argument(
        "--off-target", default=cfg.off_target_assay, help="off-target assay, or empty for none"
    )
    run.add_argument(
        "--no-ocsr", action="store_true", help="skip the MolScribe image channel"
    )
    run.add_argument("--max-pages", type=int, default=None, help="limit pages (debugging)")

    ev = sub.add_parser("evaluate", help="score a run against a committed gold set")
    ev.add_argument("run_dir", help="artifact bundle directory")
    ev.add_argument("--gold", required=True, help="path to a gold JSON file")
    ev.add_argument("--json", action="store_true", help="emit the report as JSON")

    slim = sub.add_parser(
        "slim-checkpoint",
        help="build step: strip optimizer state from the MolScribe checkpoint (1134MB -> 384MB)",
    )
    slim.add_argument("--out", default=str(cfg.molscribe_ckpt))
    slim.add_argument("--src", default=None, help="full checkpoint path (downloads if omitted)")

    cal = sub.add_parser(
        "calibrate", help="sweep the OCSR confidence threshold against a gold set (PRD R13.2)"
    )
    cal.add_argument("run_dir")
    cal.add_argument("--gold", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        from sarmine.pipeline import run_pipeline

        bundle_path = run_pipeline(
            source=args.source,
            out_root=Path(args.out),
            force_pdf_path=args.force_pdf_path,
            target_assay=args.target,
            off_target_assay=args.off_target or None,
            run_ocsr=not args.no_ocsr,
            max_pages=args.max_pages,
        )
        print(bundle_path)
        return 0

    if args.command == "evaluate":
        from sarmine.evaluate import evaluate_run, format_report

        report = evaluate_run(Path(args.run_dir), Path(args.gold))
        print(json.dumps(report, indent=2) if args.json else format_report(report))
        return 0

    if args.command == "slim-checkpoint":
        from sarmine.structure.molscribe import slim_checkpoint

        out = slim_checkpoint(Path(args.out), src=Path(args.src) if args.src else None)
        print(out)
        return 0

    if args.command == "calibrate":
        from sarmine.evaluate import calibrate_threshold

        result = calibrate_threshold(Path(args.run_dir), Path(args.gold))
        print(json.dumps(result, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
