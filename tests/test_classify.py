"""Part 7 — Markush detection and crop routing (PRD §8.4, Plan Part 7.3).

Covers R8.6 (classify every crop before OCSR), R8.7 + EC-12 (Markush crops get
neither OCSR nor a cross-check) and the graceful-degradation requirement: with
no MolClassifier weights the pipeline must still run, treating every crop as
clean rather than refusing to start.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sarmine.segment import classify
from sarmine.segment.classify import (
    MOLCLASSIFIER_REPO,
    classify_crops,
    is_available,
    route,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_weights_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the weights lookup at an empty directory so tests never touch the network."""
    monkeypatch.setenv("SARMINE_MOLCLASSIFIER_DIR", str(tmp_path / "absent"))


# --------------------------------------------------------------------------
# import safety and identity
# --------------------------------------------------------------------------


def test_module_import_does_not_pull_in_transformers_or_torch() -> None:
    """PRD §17.5 — no heavyweight imports at module scope."""
    code = (
        "import sys; import sarmine.segment.classify; "
        "print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert proc.stdout.strip() == "[]"


def test_weights_repo_matches_the_prd() -> None:
    """PRD R8.6 — MolClassifier, IBM, MIT, 93.4% precision on PatCID."""
    assert MOLCLASSIFIER_REPO == "ds4sd/MolClassifier"


# --------------------------------------------------------------------------
# R8.7 — routing
# --------------------------------------------------------------------------


def test_route_sends_clean_crops_to_ocsr() -> None:
    assert route("clean") == "ocsr"


def test_route_skips_trash() -> None:
    assert route("trash") == "skip"


def test_route_keeps_markush_away_from_ocsr() -> None:
    """PRD R8.7/EC-12 — OCSR would hallucinate a concrete structure from a Markush."""
    assert route("markush") == "markush"
    assert route("markush") != "ocsr"


def test_route_covers_every_crop_class_and_nothing_else() -> None:
    assert {cls: route(cls) for cls in ("clean", "markush", "trash")} == {
        "clean": "ocsr",
        "markush": "markush",
        "trash": "skip",
    }
    with pytest.raises(ValueError):
        route("probably-a-structure")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# graceful degradation — the default state of this repo
# --------------------------------------------------------------------------


def test_is_available_is_false_without_weights() -> None:
    """Missing weights are a configuration state, not an error."""
    assert is_available() is False


def test_is_available_is_false_when_transformers_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "molclassifier"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", "utf-8")
    monkeypatch.setenv("SARMINE_MOLCLASSIFIER_DIR", str(model_dir))
    monkeypatch.setattr(classify, "_transformers_installed", lambda: False)

    assert is_available() is False


def test_is_available_is_true_with_local_weights_and_transformers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "molclassifier"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", "utf-8")
    monkeypatch.setenv("SARMINE_MOLCLASSIFIER_DIR", str(model_dir))
    monkeypatch.setattr(classify, "_transformers_installed", lambda: True)

    assert is_available() is True


def test_unavailable_classifier_treats_every_crop_as_clean(tmp_path: Path) -> None:
    """R8.6 is a gate, not a dependency: without it every crop goes to OCSR."""
    crops = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]

    results = classify_crops(crops)

    assert len(results) == len(crops)
    assert [cls for cls, _score in results] == ["clean", "clean", "clean"]
    # A zero score says "not actually classified", so nothing downstream can
    # mistake the degraded default for a confident judgement.
    assert [score for _cls, score in results] == [0.0, 0.0, 0.0]
    assert {route(cls) for cls, _ in results} == {"ocsr"}


def test_a_failing_model_load_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "molclassifier"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", "utf-8")
    monkeypatch.setenv("SARMINE_MOLCLASSIFIER_DIR", str(model_dir))
    monkeypatch.setattr(classify, "_transformers_installed", lambda: True)

    def _boom() -> Any:
        raise RuntimeError("no such architecture")

    monkeypatch.setattr(classify, "_load_classifier", _boom)

    results = classify_crops([tmp_path / "a.png"])

    assert results == [("clean", 0.0)]


def test_classify_crops_on_an_empty_batch() -> None:
    assert classify_crops([]) == []


# --------------------------------------------------------------------------
# label mapping, with the model itself stubbed out
# --------------------------------------------------------------------------


@pytest.fixture()
def stubbed_classifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Make the classifier 'available' and replace its forward pass."""
    model_dir = tmp_path / "molclassifier"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", "utf-8")
    monkeypatch.setenv("SARMINE_MOLCLASSIFIER_DIR", str(model_dir))
    monkeypatch.setattr(classify, "_transformers_installed", lambda: True)
    monkeypatch.setattr(classify, "_load_classifier", lambda: ("model", "processor"))

    def install(labels: list[tuple[str, float]]) -> None:
        monkeypatch.setattr(
            classify,
            "_predict_labels",
            lambda paths, model, processor: labels[: len(list(paths))],
        )

    return install


def test_model_labels_map_onto_the_three_crop_classes(stubbed_classifier: Any) -> None:
    """PRD R8.6 — Clean / Markush / Trash, however the checkpoint spells them."""
    stubbed_classifier(
        [("Clean", 0.97), ("markush_structure", 0.88), ("TRASH", 0.71), ("LABEL_7", 0.42)]
    )

    results = classify_crops([Path(f"{i}.png") for i in range(4)])

    assert results == [
        ("clean", 0.97),
        ("markush", 0.88),
        ("trash", 0.71),
        ("clean", 0.42),  # an unrecognised label must not silently become Trash
    ]


def test_ec_12_a_markush_crop_is_routed_away_from_ocsr(stubbed_classifier: Any) -> None:
    """PRD EC-12 — no OCSR and no cross-check; the row is marked markush_detected."""
    stubbed_classifier([("markush", 0.93), ("clean", 0.99)])

    routes = [route(cls) for cls, _score in classify_crops([Path("m.png"), Path("c.png")])]

    assert routes == ["markush", "ocsr"]
