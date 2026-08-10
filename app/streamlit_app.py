"""Patent SAR Miner — the reviewer's instrument (PRD §14).

At ~60% end-to-end extraction accuracy the review interface *is* the product
(PRD §13.1): a chemist cannot use a table they cannot trace, so every value on
screen is one click from the crop it came from.

Memory discipline (PRD R17.5): `st.session_state` holds artifact **paths**, never
decoded images and never model objects. Nothing here is wrapped in
`@st.cache_resource` — pinning MolScribe would hold ~1.3 GB for the lifetime of
the process on a host with 2.7 GB.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sarmine.artifacts.writer import list_runs, read_bundle  # noqa: E402
from sarmine.config import get_config  # noqa: E402
from sarmine.rank.scorer import rank_compounds, shortlist  # noqa: E402
from sarmine.resources import PEAK_RSS_BUDGET_MB, STAGE_ORDER  # noqa: E402
from sarmine.review.edits import CorrectionStore, missing_provenance  # noqa: E402
from sarmine.review.queue import build_queue  # noqa: E402
from sarmine.review.render import render_crop_with_bbox, render_structure_png  # noqa: E402

SCREENS = (
    "Ingest",
    "SAR Table",
    "Shortlist",
    "Review Queue",
    "Anomalies",
    "Export",
    "About / Accuracy",
)

TIER_HELP = {
    "AGREE_FULL": "Both channels produced the identical InChIKey — near-certain.",
    "AGREE_SKELETON": "Same skeleton, different stereochemistry. Accepted, stereo flagged.",
    "CONFLICT": "The channels disagree on connectivity. The name wins the stored value; unresolved.",
    "SINGLE_SOURCE": "Only one channel produced a structure. Neutral, NOT low confidence.",
    "NONE": "Neither channel produced a usable structure.",
}


# --------------------------------------------------------------------------
# session state — paths and plain data only
# --------------------------------------------------------------------------


def _state() -> None:
    st.session_state.setdefault("run_dir", None)
    st.session_state.setdefault("corrections", CorrectionStore())
    st.session_state.setdefault("screen", "Ingest")


def _bundle():
    run_dir = st.session_state.get("run_dir")
    if not run_dir:
        return None
    try:
        return read_bundle(run_dir)
    except Exception as exc:  # a half-written bundle must not crash the app
        st.error(f"Could not read the artifact bundle at {run_dir}: {exc}")
        return None


def _apply_corrections(bundle):
    """Replay session corrections onto the bundle and re-rank (PRD R13.5, R13.6).

    The bundle on disk is never rewritten: the original extraction is always what
    is stored, and the audit trail is what turns it into the corrected view.
    """
    store: CorrectionStore = st.session_state["corrections"]
    entries = store.entries
    if not entries:
        return bundle

    replay = CorrectionStore()  # replaying must not duplicate the audit trail
    compounds = {c.compound_id: i for i, c in enumerate(bundle.compounds)}
    measurements = {m.measurement_id: i for i, m in enumerate(bundle.measurements)}

    for entry in entries:
        if entry.target_kind == "compound" and entry.target_id in compounds:
            index = compounds[entry.target_id]
            value = entry.corrected
            if entry.field == "compound_number":
                value = int(value) if value and value.isdigit() else None
            bundle.compounds[index] = replay.correct_compound(
                bundle.compounds[index], entry.field, value
            )
        elif entry.target_kind == "measurement" and entry.target_id in measurements:
            index = measurements[entry.target_id]
            bundle.measurements[index] = replay.correct_measurement(
                bundle.measurements[index], entry.field, entry.corrected
            )

    rank_compounds(
        bundle.compounds,
        bundle.measurements,
        target=bundle.manifest.target_assay,
        off_target=bundle.manifest.off_target_assay,
    )
    return bundle


def _resolve(bundle, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = Path(relative)
    return path if path.is_absolute() else Path(bundle.root) / relative


# --------------------------------------------------------------------------
# screens
# --------------------------------------------------------------------------


def screen_ingest() -> None:
    st.header("Ingest")
    st.caption(
        "Upload a chemistry patent PDF, or enter a publication number. "
        "Extraction is genuine and live — nothing here is pre-computed."
    )

    cfg = get_config()
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Patent PDF", type=["pdf"])
    with right:
        pubnum = st.text_input("…or a publication number", placeholder="WO2024097932A1")

    with st.expander("Options"):
        target = st.text_input("Target assay", value=cfg.target_assay)
        off_target = st.text_input("Off-target assay", value=cfg.off_target_assay or "")
        st.caption(
            "These two set the sign of the selectivity score (PRD R12.4). "
            "Getting them backwards ranks the wrong compounds first."
        )
        force_pdf = st.checkbox(
            "Force the full PDF OCR path (skip patents.google.com)", value=False
        )
        run_ocsr = st.checkbox("Run the image channel (MolScribe)", value=True)
        pages = st.text_input(
            "Page ranges (optional, e.g. 61-88,182-187)",
            help="Limits the run to the given pages. Leave empty to read the whole document.",
        )

    if st.button("Run extraction", type="primary", width="stretch"):
        source = _stage_source(uploaded, pubnum)
        if source is None:
            st.warning("Upload a PDF or enter a publication number first.")
            return
        _run_extraction(source, target, off_target, force_pdf, run_ocsr, pages)

    _existing_runs()


def _stage_source(uploaded, pubnum: str) -> str | None:
    if uploaded is not None:
        staged = get_config().artifact_root / "_uploads"
        staged.mkdir(parents=True, exist_ok=True)
        target = staged / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        return str(target)
    return pubnum.strip() or None


def _parse_ranges(text: str):
    ranges = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        first, _, last = chunk.partition("-")
        try:
            ranges.append((int(first), int(last or first)))
        except ValueError:
            continue
    return ranges or None


def _run_extraction(source, target, off_target, force_pdf, run_ocsr, pages) -> None:
    from sarmine.pipeline import run_pipeline

    progress = st.progress(0.0)
    status = st.empty()
    log = st.container()
    seen: list[str] = []

    def on_progress(stage: str, phase: str, index: int, total: int) -> None:
        # PRD R14.1 — a blocking run with a progress bar and per-stage status.
        progress.progress(min(index / max(total, 1), 1.0))
        if phase == "start":
            status.info(f"Stage {index}/{total}: {stage}")
        elif stage not in seen:
            seen.append(stage)
            log.write(f"done: {stage}")

    try:
        with st.spinner("Extracting — this takes a few minutes on a laptop CPU."):
            result = run_pipeline(
                source,
                force_pdf_path=force_pdf,
                target_assay=target or None,
                off_target_assay=off_target or None,
                run_ocsr=run_ocsr,
                page_ranges=_parse_ranges(pages),
                on_progress=on_progress,
            )
    except Exception as exc:
        status.empty()
        st.error(str(exc))
        return

    progress.progress(1.0)
    status.success(f"Extracted {result.manifest.n_compounds} compounds.")
    st.session_state["run_dir"] = str(result.bundle_dir)
    st.session_state["corrections"] = CorrectionStore()
    st.session_state["screen"] = "SAR Table"
    st.rerun()


def _existing_runs() -> None:
    runs = list_runs(get_config().artifact_root)
    if not runs:
        return
    st.divider()
    st.subheader("Previous runs")
    st.caption("Artifacts are on disk, so a completed run reopens instantly.")
    choice = st.selectbox("Run", [str(p) for p in runs], index=len(runs) - 1)
    if st.button("Open this run"):
        st.session_state["run_dir"] = choice
        st.session_state["corrections"] = CorrectionStore()
        st.session_state["screen"] = "SAR Table"
        st.rerun()


def _sar_frame(bundle) -> pd.DataFrame:
    by_compound: dict[str, dict[str, str]] = {}
    assays: list[str] = []
    for measurement in bundle.measurements:
        label = measurement.published_type
        if label not in assays:
            assays.append(label)
        cell = measurement.published_value
        if measurement.bin_lower_nM is not None or measurement.bin_upper_nM is not None:
            cell = f"{cell} ({_interval(measurement)})"
        by_compound.setdefault(measurement.compound_id, {})[label] = cell

    rows = []
    for compound in bundle.compounds:
        row = {
            "rank": compound.rank,
            "tie group": compound.rank_tie_group,
            "compound": compound.compound_local_id,
            "potency": compound.potency_score,
            "selectivity": compound.selectivity_score,
        }
        row.update({assay: by_compound.get(compound.compound_id, {}).get(assay, "") for assay in assays})
        row.update(
            {
                "SMILES": compound.smiles_final,
                "InChIKey": compound.inchikey_full,
                "confidence": compound.crosscheck_tier,
                "Markush": "yes" if compound.markush_detected else "",
                "in Examples": "yes" if compound.in_examples else "",
                "MW": _round(compound.mw),
                "cLogP": _round(compound.clogp),
                "TPSA": _round(compound.tpsa),
                "QED": _round(compound.qed),
                "pages": ", ".join(str(p.page_no) for p in compound.provenance[:4]),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "rank" in frame and frame["rank"].notna().any():
        frame = frame.sort_values("rank", na_position="last")
    return frame


def _interval(measurement) -> str:
    units = measurement.standard_units or ""
    low, high = measurement.bin_lower_nM, measurement.bin_upper_nM
    if low is not None and high is not None:
        return f"{_round(low)}–{_round(high)} {units}".strip()
    if high is not None:
        return f"< {_round(high)} {units}".strip()
    if low is not None:
        return f"> {_round(low)} {units}".strip()
    return ""


def _round(value):
    return None if value is None else round(float(value), 2)


def screen_sar_table(bundle) -> None:
    st.header("SAR Table")
    st.caption(
        "One row per compound: structure, every assay column with its decoded interval, "
        "computed properties, confidence tier and the pages every field came from. "
        "Sorted by rank."
    )
    frame = _sar_frame(bundle)
    if frame.empty:
        st.info("This run produced no compounds.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)

    st.subheader("Inspect a compound")
    labels = {c.compound_local_id: c for c in bundle.compounds}
    choice = st.selectbox("Compound", list(labels))
    _compound_detail(bundle, labels[choice])


def _compound_detail(bundle, compound) -> None:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Rendered from the extracted SMILES**")
        png = _render_structure(bundle, compound)
        if png:
            st.image(str(png))
        else:
            st.info("No structure was extracted for this compound.")
        st.markdown(f"**Confidence:** `{compound.crosscheck_tier}`")
        st.caption(TIER_HELP.get(compound.crosscheck_tier, ""))
        if compound.rank_rationale:
            st.markdown("**Why it ranks here**")
            for reason in compound.rank_rationale:
                st.markdown(f"- {reason}")
        if compound.investment_reasons:
            st.markdown("**Investment signal**")
            for reason in compound.investment_reasons:
                st.markdown(f"- {reason}")

    with right:
        st.markdown("**Provenance** — every field traced to its page (PRD R14.2)")
        if not compound.provenance:
            st.info("No provenance recorded for this compound.")
        for prov in compound.provenance:
            with st.expander(f"page {prov.page_no} · {Path(prov.crop_path).stem}"):
                crop = _resolve(bundle, prov.crop_path)
                if crop and crop.is_file():
                    out = Path(bundle.root) / "svg" / f"overlay_{Path(prov.crop_path).stem}.png"
                    try:
                        render_crop_with_bbox(
                            crop, prov.bbox, out, label=f"page {prov.page_no}",
                            crop_origin=(prov.bbox[0], prov.bbox[1]),
                        )
                        st.image(str(out))
                    except Exception:
                        st.image(str(crop))
                else:
                    st.caption(f"crop not on disk: {prov.crop_path}")
                st.caption(
                    f"bbox {prov.bbox} · rotation {prov.rotation_applied}° · "
                    f"{prov.extractor} · {prov.source}"
                )

    st.markdown("**Both channels**")
    st.code(
        f"name  (OPSIN)     -> {compound.smiles_from_name}\n"
        f"image (MolScribe) -> {compound.smiles_from_image}\n"
        f"InChIKey (name)   -> {compound.inchikey_from_name}\n"
        f"InChIKey (image)  -> {compound.inchikey_from_image}",
        language="text",
    )


def _render_structure(bundle, compound) -> Path | None:
    if not compound.smiles_final:
        return None
    out = Path(bundle.root) / "svg" / f"{compound.compound_local_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        return render_structure_png(compound.smiles_final, out)
    except Exception:
        return None


def screen_shortlist(bundle) -> None:
    st.header("Shortlist")
    st.caption(
        "The ten compounds to look at first. The ranking is tie-heavy by construction — "
        "three bins per assay over 54 compounds — so tie groups are shown rather than hidden."
    )
    picks = shortlist(bundle.compounds, 10)
    if not picks:
        st.info("Nothing has been ranked yet.")
        return

    for compound in picks:
        tie = compound.rank_tie_group
        siblings = [
            c.compound_local_id
            for c in bundle.compounds
            if tie is not None and c.rank_tie_group == tie and c.compound_id != compound.compound_id
        ]
        with st.container(border=True):
            head, body = st.columns([1, 4])
            with head:
                st.metric(f"#{compound.rank}", compound.compound_local_id)
                st.caption(f"potency {compound.potency_score} · selectivity {compound.selectivity_score}")
            with body:
                for reason in compound.rank_rationale or ["no rationale recorded"]:
                    st.markdown(f"- {reason}")
                if siblings:
                    st.warning(
                        "Indistinguishable from "
                        + ", ".join(siblings)
                        + " — same scores, so the order between them is arbitrary."
                    )


def screen_review(bundle) -> None:
    st.header("Review Queue")
    st.caption(
        "Extractions the system is not confident about, highest priority first. "
        "Corrections are session-scoped and re-rank the table immediately."
    )
    items = build_queue(bundle.compounds, bundle.measurements, bundle.anomalies)
    if not items:
        st.success("Nothing is queued for review.")
        return

    counts = {}
    for item in items:
        counts[item.priority] = counts.get(item.priority, 0) + 1
    st.write(" · ".join(f"**{p}**: {n}" for p, n in counts.items()))

    priority = st.selectbox("Priority", ["all", *counts])
    shown = [i for i in items if priority == "all" or i.priority == priority]

    by_id = {c.compound_id: c for c in bundle.compounds}
    for index, item in enumerate(shown[:40]):
        compound = by_id.get(item.compound_id)
        with st.expander(
            f"[{item.priority}] {item.compound_id} — {item.trigger}", expanded=index == 0
        ):
            st.write(item.reason)
            left, right = st.columns(2)
            with left:
                st.markdown("**Source crop**")
                crop = _resolve(bundle, item.crop_path)
                if crop and crop.is_file():
                    out = Path(bundle.root) / "svg" / f"review_{index}.png"
                    try:
                        render_crop_with_bbox(
                            crop, item.bbox, out, label=f"page {item.page_no}",
                            crop_origin=(item.bbox[0], item.bbox[1]) if item.bbox else None,
                        )
                        st.image(str(out))
                    except Exception:
                        st.image(str(crop))
                else:
                    st.caption("no crop recorded")
            with right:
                st.markdown("**Extraction**")
                if compound and compound.smiles_final:
                    png = _render_structure(bundle, compound)
                    if png:
                        st.image(str(png))
                st.code(
                    f"name  -> {item.smiles_from_name}\n"
                    f"image -> {item.smiles_from_image}\n"
                    f"key(name)  -> {item.inchikey_from_name}\n"
                    f"key(image) -> {item.inchikey_from_image}\n"
                    f"tier  -> {item.crosscheck_tier}",
                    language="text",
                )
            _correction_form(bundle, item, compound, index)


def _correction_form(bundle, item, compound, index: int) -> None:
    store: CorrectionStore = st.session_state["corrections"]
    with st.form(f"fix-{index}"):
        smiles = st.text_input(
            "Corrected SMILES",
            value=(compound.smiles_final if compound else "") or "",
            help="The original extraction is always retained, with an audit entry.",
        )
        number = st.text_input(
            "Corrected compound number",
            value=str(compound.compound_number) if compound and compound.compound_number else "",
        )
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Accept correction"):
            if compound and smiles and smiles != (compound.smiles_final or ""):
                store.correct_compound(compound, "smiles_final", smiles, note=note or None)
            if compound and number and number != str(compound.compound_number or ""):
                store.correct_compound(compound, "compound_number", number, note=note or None)
            st.success("Recorded. Ranking has been re-run.")
            st.rerun()


def screen_anomalies(bundle) -> None:
    st.header("Anomalies")
    st.caption(
        "Document-level issues. These are non-blocking: they describe the patent, not a failure "
        "of the extraction. The reference patent contradicts its own legend three times."
    )
    if not bundle.anomalies:
        st.success("No document-level anomalies were recorded.")
        return

    frame = pd.DataFrame(
        [
            {
                "kind": a.kind,
                "severity": a.severity,
                "message": a.message,
                "page": a.provenance.page_no if a.provenance else None,
            }
            for a in bundle.anomalies
        ]
    )
    for kind, group in frame.groupby("kind"):
        with st.expander(f"{kind} ({len(group)})", expanded=kind == "legend_contradiction"):
            st.dataframe(group.drop(columns=["kind"]), width="stretch", hide_index=True)

    gaps = missing_provenance(bundle.compounds, bundle.measurements)
    if gaps:
        st.subheader("Values that cannot be traced to a page")
        st.caption("Provenance is the product (PRD §2.3), so gaps in it are reported, not hidden.")
        st.write(gaps[:50])


def screen_export(bundle) -> None:
    st.header("Export")
    st.caption("Wide for the chemist, long for the record. Corrections travel with the export.")

    sar = _sar_frame(bundle)
    measurements = pd.DataFrame([m.model_dump(mode="json") for m in bundle.measurements])
    corrections = pd.DataFrame(st.session_state["corrections"].to_rows())

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        sar.to_excel(writer, sheet_name="SAR", index=False)
        if not measurements.empty:
            measurements.to_excel(writer, sheet_name="Measurements", index=False)
        if not corrections.empty:
            corrections.to_excel(writer, sheet_name="Corrections", index=False)
        pd.DataFrame(
            [{"kind": a.kind, "severity": a.severity, "message": a.message} for a in bundle.anomalies]
        ).to_excel(writer, sheet_name="Anomalies", index=False)

    stem = f"{bundle.manifest.pubnum}_{bundle.manifest.run_id}"
    st.download_button(
        "Download XLSX",
        buffer.getvalue(),
        file_name=f"{stem}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.download_button(
        "Download SAR table as CSV",
        sar.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}.csv",
        mime="text/csv",
        width="stretch",
    )
    if not corrections.empty:
        st.download_button(
            "Download corrections as CSV",
            corrections.to_csv(index=False).encode("utf-8"),
            file_name=f"{stem}_corrections.csv",
            mime="text/csv",
            width="stretch",
        )
    st.dataframe(sar.head(30), width="stretch", hide_index=True)


def screen_about(bundle) -> None:
    st.header("About / Accuracy")
    st.markdown(
        "This tool reports its own error rate, because at the accuracy the published "
        "literature achieves on whole patent documents a table you cannot audit is worse "
        "than no table at all. The most reliable published end-to-end figure is IBM's "
        "PatCID study: **63.0%** on recent US patents and **57.1%** on older ones. The "
        "~93% numbers quoted for structure recognition are measured on pre-cropped, "
        "ground-truth-segmented single molecules and do not transfer to whole documents."
    )

    if bundle is not None:
        gold = get_config().gold_dir / f"{bundle.manifest.pubnum}.gold.json"
        if gold.is_file():
            from sarmine.evaluate import evaluate_run

            try:
                report = evaluate_run(bundle.root, gold)
            except Exception as exc:
                st.warning(f"Could not score this run against the gold set: {exc}")
            else:
                st.subheader("Measured against the committed gold set")
                triplet = report["triplet"]
                columns = st.columns(4)
                columns[0].metric("End-to-end triplet", _pct(triplet.get("f1")))
                columns[1].metric("Activity cells", _pct(report["activity_cells"].get("accuracy")))
                columns[2].metric("Structures", _pct(report["structures"].get("accuracy")))
                columns[3].metric("Join", _pct(report["join"].get("accuracy")))
                st.caption(
                    "The headline number is end-to-end triplet accuracy — compound, assay and "
                    "value all correct together. Per-component scores always look better and "
                    "are misleading."
                )
                with st.expander("Full report"):
                    st.json(report)
        else:
            st.info(f"No gold set is committed for {bundle.manifest.pubnum}.")

        st.subheader("This run")
        st.write(
            {
                "publication number": bundle.manifest.pubnum,
                "source mode": bundle.manifest.source_mode,
                "pages": bundle.manifest.n_pages,
                "compounds": bundle.manifest.n_compounds,
                "measurements": bundle.manifest.n_measurements,
                "target / off-target": f"{bundle.manifest.target_assay} / {bundle.manifest.off_target_assay}",
                "versions": bundle.manifest.versions,
            }
        )
        peak = bundle.manifest.stage_peak_rss_mb
        if peak:
            st.subheader("Measured resource use")
            st.caption(
                "Peak RSS is recorded at every stage boundary so the memory budget is a "
                "measured value rather than an assumption (PRD R17.6)."
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "stage": list(peak),
                        "peak RSS (MB)": list(peak.values()),
                        "seconds": [
                            bundle.manifest.stage_timings_s.get(s) for s in peak
                        ],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            worst = max(peak.values())
            (st.success if worst <= PEAK_RSS_BUDGET_MB else st.warning)(
                f"Peak RSS {worst:.0f} MB against a {PEAK_RSS_BUDGET_MB:.0f} MB budget."
            )

    st.subheader("Scope")
    st.markdown(
        "- Markush (generic) structures are **detected and flagged, never enumerated** — the "
        "state of the art reaches ~13% exact match on real patent images.\n"
        "- Letter bins are stored as intervals. No midpoint is ever imputed, because that "
        "would fabricate precision the patent does not contain.\n"
        "- Units are never inferred from magnitude. A missing unit goes to review instead.\n"
        "- Output is decision support, not legal advice."
    )


def _pct(value) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Patent SAR Miner", page_icon="⌬", layout="wide")
    _state()

    st.sidebar.title("Patent SAR Miner")
    st.sidebar.caption("One chemistry patent → a joined, traceable SAR table.")

    bundle = _bundle()
    if bundle is not None:
        bundle = _apply_corrections(bundle)
        # PRD R14.4 — the reviewer must know which source they are looking at.
        st.sidebar.success(
            f"**{bundle.manifest.pubnum}**\n\n"
            f"source mode: `{bundle.manifest.source_mode}`\n\n"
            f"{bundle.manifest.n_compounds} compounds · "
            f"{bundle.manifest.n_measurements} measurements"
        )

    screen = st.sidebar.radio("Screen", SCREENS, index=SCREENS.index(st.session_state["screen"]))
    st.session_state["screen"] = screen

    # PRD R14.3 — state the limitations in the UI rather than in a README.
    st.sidebar.divider()
    st.sidebar.caption(
        "**Single user, one reviewer at a time.** There is no authentication and no "
        "concurrency control.\n\n"
        "**Edits are session-scoped.** The free host has no persistent disk, so corrections live "
        "in this browser session — export before you close the tab.\n\n"
        "Decision support, not legal advice."
    )

    if screen == "Ingest":
        screen_ingest()
        return
    if screen == "About / Accuracy":
        screen_about(bundle)
        return
    if bundle is None:
        st.info("Run an extraction first, on the Ingest screen.")
        return

    {
        "SAR Table": screen_sar_table,
        "Shortlist": screen_shortlist,
        "Review Queue": screen_review,
        "Anomalies": screen_anomalies,
        "Export": screen_export,
    }[screen](bundle)


if __name__ == "__main__":  # PRD R9.10 — never let a DataLoader re-enter this
    main()
