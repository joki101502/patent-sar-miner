"""Streamlit app smoke tests (PRD §14, Plan Part 11).

The app cannot be meaningfully unit-tested without a Streamlit runtime, but the
failure that actually bites — an import-time error or a drifted function
signature in a module the app calls — is cheap to catch here.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("sarmine_app_undertest", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_imports_without_side_effects(app_module):
    """PRD §17.5 — a module with top-level side effects re-runs on import."""
    assert callable(app_module.main)


def test_every_prd_screen_exists(app_module):
    """PRD §14.1 — Ingest, Progress, SAR Table, Shortlist, Review Queue,
    Anomalies, Export. Progress is rendered inside the ingest run (R14.1)."""
    for screen in (
        "screen_ingest",
        "screen_sar_table",
        "screen_shortlist",
        "screen_review",
        "screen_anomalies",
        "screen_export",
        "screen_about",
    ):
        assert callable(getattr(app_module, screen)), f"missing {screen}"


def test_entry_point_is_main_guarded():
    """PRD R9.10 — MolScribe's DataLoader re-executes the entry script."""
    assert '__name__ == "__main__"' in APP.read_text("utf-8")


def test_app_does_not_cache_models_or_images_in_session_state():
    """PRD R17.5 — `@st.cache_resource` on a model pins ~1.3 GB for the process
    lifetime on a 2.7 GB host, and session_state must hold artifact PATHS only."""
    import ast

    tree = ast.parse(APP.read_text("utf-8"))
    decorators = [
        ast.unparse(decorator)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
    ]
    assert not [d for d in decorators if "cache_resource" in d or "cache_data" in d], (
        f"Streamlit caching decorators found: {decorators}"
    )


def test_functions_the_app_calls_still_have_the_expected_signatures():
    """Guards against another module drifting away from what the app passes."""
    from sarmine.export import to_csv, to_wide_frame, to_xlsx
    from sarmine.rank.scorer import rank_compounds, shortlist
    from sarmine.review.queue import build_queue, sort_queue
    from sarmine.review.render import render_crop_with_bbox, render_structure_png

    assert {"compounds", "measurements"} <= set(inspect.signature(to_wide_frame).parameters)
    assert "corrections" in inspect.signature(to_xlsx).parameters
    assert {"target", "off_target"} <= set(inspect.signature(rank_compounds).parameters)
    assert "n" in inspect.signature(shortlist).parameters
    assert "ocsr_conf_threshold" in inspect.signature(build_queue).parameters
    assert "label" in inspect.signature(render_crop_with_bbox).parameters
    assert callable(sort_queue) and callable(render_structure_png) and callable(to_csv)


def test_app_states_the_single_user_and_session_scoped_limitations():
    """PRD R14.3 — both limitations must be stated in the UI, not just the README."""
    source = APP.read_text("utf-8").lower()
    assert "single user" in source
    assert "session-scoped" in source


def test_app_surfaces_the_source_mode(app_module):
    """PRD R14.4 — the reviewer must know whether this was structured or full OCR."""
    source = APP.read_text("utf-8")
    assert "source_mode" in source
    assert "full PDF OCR" in source
