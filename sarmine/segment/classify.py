"""Markush detection — the gate in front of OCSR (PRD §8.4, Plan Part 7.3).

Implements R8.6 (classify every structure crop into Clean / Markush / Trash
before OCSR) and R8.7/EC-12 (drop Trash, send Clean to OCSR, and route Markush
away from both OCSR and the cross-check). A Markush image fails OPSIN outright
while OCSR happily hallucinates a concrete structure; that combination attaches
a confidently wrong molecule to a real activity value.

The gate is optional by design. When the weights or `transformers` are absent
the pipeline degrades to treating every crop as clean rather than refusing to
run, and says so by reporting a zero score.

Note for whoever wires up the real weights: `ds4sd/MolClassifier` publishes a
bare torch checkpoint whose model class lives in IBM's own package, not in
`transformers`. `_resolve_weights` therefore looks for a local
`transformers`-style model directory (`$SARMINE_MOLCLASSIFIER_DIR`, else
`models/molclassifier`), which is what a converted checkpoint would look like.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Literal, Sequence

from sarmine.config import REPO_ROOT

logger = logging.getLogger(__name__)

MOLCLASSIFIER_REPO = "ds4sd/MolClassifier"
MOLCLASSIFIER_CKPT_FILE = "models/molclassifier_model.chpt"
MOLCLASSIFIER_DIR_ENV = "SARMINE_MOLCLASSIFIER_DIR"
DEFAULT_MOLCLASSIFIER_DIR = REPO_ROOT / "models" / "molclassifier"

CropClass = Literal["clean", "markush", "trash"]
CropRoute = Literal["ocsr", "skip", "markush"]

# PRD R8.7 — the whole point of the gate is the middle row.
ROUTES: dict[CropClass, CropRoute] = {
    "clean": "ocsr",
    "markush": "markush",
    "trash": "skip",
}

# An unclassified crop is treated as clean (R8.7's default action) and scored 0
# so nothing downstream reads the degraded default as a confident judgement.
DEGRADED_CLASS: CropClass = "clean"
DEGRADED_SCORE = 0.0


def route(cls: CropClass) -> CropRoute:
    """Map a crop class onto its pipeline destination (PRD R8.7)."""
    try:
        return ROUTES[cls]
    except KeyError:
        raise ValueError(f"unknown crop class {cls!r}; expected one of {sorted(ROUTES)}") from None


def is_available() -> bool:
    """True only when a local model and `transformers` are both present."""
    return _resolve_weights() is not None and _transformers_installed()


def classify_crops(image_paths: Sequence[Path]) -> list[tuple[CropClass, float]]:
    """Classify a batch of crops, one `(class, score)` per input path (PRD R8.6)."""
    paths = list(image_paths)
    if not paths:
        return []

    degraded = [(DEGRADED_CLASS, DEGRADED_SCORE)] * len(paths)
    if not is_available():
        logger.info(
            "MolClassifier unavailable; treating %d crops as clean (PRD R8.6 degradation)",
            len(paths),
        )
        return degraded

    try:
        model, processor = _load_classifier()
        labelled = _predict_labels(paths, model, processor)
    except Exception:
        logger.exception("MolClassifier failed; treating %d crops as clean", len(paths))
        return degraded

    return [(_to_crop_class(label), float(score)) for label, score in labelled]


def _transformers_installed() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _resolve_weights() -> Path | None:
    """Local model directory, or None. Never downloads (PRD R17.9 in spirit)."""
    configured = os.environ.get(MOLCLASSIFIER_DIR_ENV, "").strip()
    candidate = Path(configured) if configured else DEFAULT_MOLCLASSIFIER_DIR
    return candidate if (candidate / "config.json").is_file() else None


def _load_classifier() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    weights = _resolve_weights()
    if weights is None:
        raise FileNotFoundError(f"no MolClassifier model directory at {DEFAULT_MOLCLASSIFIER_DIR}")
    model = AutoModelForImageClassification.from_pretrained(str(weights))
    model.eval()
    return model, AutoImageProcessor.from_pretrained(str(weights))


def _predict_labels(
    image_paths: Sequence[Path], model: Any, processor: Any
) -> list[tuple[str, float]]:
    """One forward pass over the whole batch, returning the model's own labels."""
    import torch
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=-1)
    best = scores.argmax(dim=-1)
    return [
        (str(model.config.id2label[int(index)]), float(scores[row, index]))
        for row, index in enumerate(best)
    ]


def _to_crop_class(label: str) -> CropClass:
    """Normalize whatever the checkpoint calls its classes onto PRD R8.6's three."""
    normalized = label.strip().lower()
    if "markush" in normalized:
        return "markush"
    if "trash" in normalized or "noise" in normalized:
        return "trash"
    # An unrecognised label must not become Trash: dropping a real structure
    # costs a compound, while sending a bad crop to OCSR costs one failed parse.
    return "clean"
