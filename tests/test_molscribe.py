"""Part 7 — the MolScribe image channel (PRD §9.3, Plan Part 7).

The model weights are 1.13 GB, so the default run must never download them.
Everything except `test_ac_3_4_*` runs against a synthetic checkpoint dict or a
stub model object; the one real-model test skips cleanly when
`models/molscribe_slim.pth` is absent.

Covers: R9.9 (checkpoint slimming), R9.10 (no fork storm / import safety),
R9.11 + R9.12 (batched call with confidences), R9.13/R13.1 (minimum atom and
bond confidence), R17.4 (`free()`), R17.9 (build step, never a runtime
download), EC-20 (unparseable SMILES is data, not a crash), AC-3.3, AC-3.4.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sarmine.structure.molscribe import (
    MOLSCRIBE_CKPT_FILE,
    MOLSCRIBE_REPO,
    MolScribeRunner,
    OcsrResult,
    molscribe_version,
    slim_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SLIM_CKPT = REPO_ROOT / "models" / "molscribe_slim.pth"

# PRD §9.4 / Appendix B.2 — compound 5, the first data row of source page 63.
AC_3_4_SMILES = "COc1ccc(-c2cc3ncn(C)c3cc2Nc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)cc1F"
AC_3_4_INCHIKEY = "WZPDSZGYLXZFEK-UHFFFAOYSA-N"

# PRD §9.4 quotes the crop as bbox (350,211)->(1394,715), 1044x504 px, on the
# de-rotated page. That page was rendered at 200 DPI; `pages/p-063-000.png` is
# the same page at 300 DPI, so every coordinate scales by 300/200.
AC_3_4_BBOX_200DPI = (350, 211, 1394, 715)
AC_3_4_DPI_SCALE = 1.5
TABLE1_DEROTATION = -90  # see tests/test_segment.py


# --------------------------------------------------------------------------
# stub model — stands in for a loaded MolScribe without the 1.13 GB of weights
# --------------------------------------------------------------------------


class _StubMolScribe:
    """Records how it was called and replays canned `predict_image_files` output."""

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def predict_image_files(self, image_files: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((list(image_files), dict(kwargs)))
        return self.outputs[: len(image_files)]


def _prediction(
    smiles: str,
    *,
    confidence: float = 0.9,
    atom_confidences: tuple[float, ...] = (0.99, 0.95),
    bond_confidences: tuple[float, ...] = (0.98,),
) -> dict[str, Any]:
    """One entry shaped like MolScribe's `predict_image_files` output."""
    return {
        "smiles": smiles,
        "molfile": "",
        "confidence": confidence,
        "atoms": [
            {"atom_symbol": "C", "x": 0.0, "y": 0.0, "confidence": c} for c in atom_confidences
        ],
        "bonds": [
            {"bond_type": "single", "endpoint_atoms": (0, 1), "confidence": c}
            for c in bond_confidences
        ],
    }


def _runner_with_stub(tmp_path: Path, stub: _StubMolScribe) -> MolScribeRunner:
    """A runner that believes it has a checkpoint and is already loaded."""
    ckpt = tmp_path / "molscribe_slim.pth"
    ckpt.write_bytes(b"not a real checkpoint")
    runner = MolScribeRunner(ckpt=ckpt)
    runner._model = stub  # type: ignore[attr-defined]
    return runner


# --------------------------------------------------------------------------
# import safety and identity
# --------------------------------------------------------------------------


def test_module_import_does_not_pull_in_torch_or_molscribe() -> None:
    """PRD §17.5 — importing a module must have no heavyweight side effects."""
    code = (
        "import sys; import sarmine.structure.molscribe; "
        "print(sorted(m for m in ('torch', 'molscribe', 'rdkit') if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert proc.stdout.strip() == "[]"


def test_checkpoint_coordinates_match_the_prd() -> None:
    """PRD R9.9 — the build step's source is pinned, not discovered at runtime."""
    assert MOLSCRIBE_REPO == "yujieq/MolScribe"
    assert MOLSCRIBE_CKPT_FILE == "swin_base_char_aux_1m680k.pth"


def test_molscribe_version_is_a_provenance_string() -> None:
    """PRD §15.3 — `Provenance.extractor` needs a versioned tool identifier."""
    version = molscribe_version()
    assert version.startswith("molscribe@")
    assert version.split("@", 1)[1]


# --------------------------------------------------------------------------
# availability: missing weights degrade, they do not raise on construction
# --------------------------------------------------------------------------


def test_runner_construction_does_not_load_the_model(tmp_path: Path) -> None:
    """PRD R17.1/R17.4 — the model loads last and only when asked to."""
    runner = MolScribeRunner(ckpt=tmp_path / "absent.pth")
    assert runner._model is None  # type: ignore[attr-defined]


def test_available_is_false_when_the_checkpoint_is_missing(tmp_path: Path) -> None:
    runner = MolScribeRunner(ckpt=tmp_path / "absent.pth")
    assert runner.available is False


def test_available_is_true_when_the_checkpoint_exists(tmp_path: Path) -> None:
    ckpt = tmp_path / "molscribe_slim.pth"
    ckpt.write_bytes(b"stub")
    assert MolScribeRunner(ckpt=ckpt).available is True


def test_load_without_a_checkpoint_names_the_build_step(tmp_path: Path) -> None:
    """PRD R17.9 — the fix is to run the build step, never a runtime download."""
    runner = MolScribeRunner(ckpt=tmp_path / "absent.pth")
    with pytest.raises(FileNotFoundError) as excinfo:
        runner.load()
    assert "slim-checkpoint" in str(excinfo.value)


def test_predict_without_a_checkpoint_returns_error_results(tmp_path: Path) -> None:
    """The image channel going missing must not abort the run (PRD R7.3 spirit)."""
    runner = MolScribeRunner(ckpt=tmp_path / "absent.pth")
    results = runner.predict([Path("a.png"), Path("b.png")])
    assert len(results) == 2
    for result in results:
        assert isinstance(result, OcsrResult)
        assert result.smiles is None
        assert result.inchikey is None
        assert result.rdkit_valid is False
        assert result.error and "checkpoint" in result.error.lower()


def test_predict_on_an_empty_batch_does_no_work(tmp_path: Path) -> None:
    runner = MolScribeRunner(ckpt=tmp_path / "absent.pth")
    assert runner.predict([]) == []


# --------------------------------------------------------------------------
# the prediction contract
# --------------------------------------------------------------------------


def test_predict_calls_molscribe_once_for_the_whole_batch(tmp_path: Path) -> None:
    """PRD R9.12 — 16.7 s for one image vs 5.9 s for a batch of two."""
    stub = _StubMolScribe([_prediction(AC_3_4_SMILES), _prediction("c1ccccc1")])
    runner = _runner_with_stub(tmp_path, stub)

    runner.predict([tmp_path / "one.png", tmp_path / "two.png"])

    assert len(stub.calls) == 1
    paths, kwargs = stub.calls[0]
    assert paths == [str(tmp_path / "one.png"), str(tmp_path / "two.png")]
    assert all(isinstance(p, str) for p in paths)  # cv2.imread rejects Path
    # PRD R9.11 — both confidence channels are requested.
    assert kwargs["return_atoms_bonds"] is True
    assert kwargs["return_confidence"] is True


def test_predict_reports_the_minimum_atom_and_bond_confidence(tmp_path: Path) -> None:
    """PRD R9.13/R13.1 — one wrong atom ruins a structure but barely moves the mean."""
    stub = _StubMolScribe(
        [
            _prediction(
                AC_3_4_SMILES,
                confidence=0.909,
                atom_confidences=(0.99, 0.902, 0.97),
                bond_confidences=(0.9999, 0.9999),
            )
        ]
    )
    runner = _runner_with_stub(tmp_path, stub)

    (result,) = runner.predict([tmp_path / "one.png"])

    assert result.confidence_molecule == pytest.approx(0.909)
    assert result.confidence_min_atom == pytest.approx(0.902)
    assert result.confidence_min_bond == pytest.approx(0.9999)
    assert result.n_atoms == 3
    assert result.n_bonds == 2


def test_predict_validates_smiles_through_rdkit_and_keys_it(tmp_path: Path) -> None:
    """AC-3.3 plus PRD R9.16 — the InChIKey is the identity, the SMILES is display."""
    stub = _StubMolScribe([_prediction(AC_3_4_SMILES)])
    runner = _runner_with_stub(tmp_path, stub)

    (result,) = runner.predict([tmp_path / "one.png"])

    assert result.rdkit_valid is True
    assert result.smiles == AC_3_4_SMILES
    assert result.inchikey == AC_3_4_INCHIKEY
    assert result.error is None


def test_unparseable_smiles_is_reported_not_raised(tmp_path: Path) -> None:
    """PRD EC-20 — a structure failing RDKit sanitization is a review trigger."""
    stub = _StubMolScribe([_prediction("C1CC(")])
    runner = _runner_with_stub(tmp_path, stub)

    (result,) = runner.predict([tmp_path / "one.png"])

    assert result.rdkit_valid is False
    assert result.inchikey is None
    assert result.smiles == "C1CC("  # kept verbatim so a reviewer can see it
    assert result.error


def test_a_model_level_failure_becomes_one_error_result_per_image(tmp_path: Path) -> None:
    class _Exploding:
        def predict_image_files(self, image_files: list[str], **kwargs: Any) -> None:
            raise RuntimeError("boom")

    runner = _runner_with_stub(tmp_path, _Exploding())  # type: ignore[arg-type]
    results = runner.predict([tmp_path / "a.png", tmp_path / "b.png"])

    assert len(results) == 2
    assert all(r.error and "boom" in r.error for r in results)
    assert all(r.smiles is None for r in results)


# --------------------------------------------------------------------------
# memory hygiene
# --------------------------------------------------------------------------


def test_free_drops_the_model(tmp_path: Path) -> None:
    """PRD R17.4 — `del model; gc.collect()` so RDKit does not stack on torch."""
    runner = _runner_with_stub(tmp_path, _StubMolScribe([]))
    runner.free()
    assert runner._model is None  # type: ignore[attr-defined]


def test_context_manager_frees_on_exit(tmp_path: Path) -> None:
    ckpt = tmp_path / "molscribe_slim.pth"
    ckpt.write_bytes(b"stub")
    with MolScribeRunner(ckpt=ckpt) as runner:
        runner._model = _StubMolScribe([])  # type: ignore[attr-defined]
    assert runner._model is None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# R9.9 — the slim-checkpoint build step, tested on a synthetic checkpoint
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_full_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint shaped like `swin_base_char_aux_1m680k.pth`: 2/3 dead weight."""
    import argparse

    import torch

    def block(n: int) -> dict[str, Any]:
        return {f"layer{i}.weight": torch.zeros(4096) for i in range(n)}

    states = {
        "encoder": block(20),
        "decoder": block(4),
        "encoder_optimizer": block(40),
        "decoder_optimizer": block(8),
        "encoder_scheduler": {"last_epoch": 7},
        "decoder_scheduler": {"last_epoch": 7},
        "global_step": 1_680_000,
        "args": argparse.Namespace(encoder="swin_base", decoder="transformer", input_size=384),
    }
    path = tmp_path / "full.pth"
    torch.save(states, path)
    return path


def test_slim_checkpoint_keeps_exactly_encoder_decoder_args(
    synthetic_full_checkpoint: Path, tmp_path: Path
) -> None:
    """PRD R9.9 — strip to `{encoder, decoder, args}`; the rest is training state."""
    import torch

    out = slim_checkpoint(tmp_path / "slim" / "molscribe_slim.pth", src=synthetic_full_checkpoint)

    assert out.is_file()
    slim = torch.load(out, map_location="cpu", weights_only=False)
    assert set(slim) == {"encoder", "decoder", "args"}
    assert set(slim["encoder"]) == set(
        torch.load(synthetic_full_checkpoint, map_location="cpu", weights_only=False)["encoder"]
    )
    assert slim["args"].input_size == 384


def test_slim_checkpoint_is_materially_smaller(
    synthetic_full_checkpoint: Path, tmp_path: Path
) -> None:
    """PRD R9.9 — measured 1,134 MB -> 384 MB (-66%) on the real checkpoint."""
    out = slim_checkpoint(tmp_path / "molscribe_slim.pth", src=synthetic_full_checkpoint)
    assert out.stat().st_size < 0.5 * synthetic_full_checkpoint.stat().st_size


def test_slim_checkpoint_rejects_a_checkpoint_without_weights(tmp_path: Path) -> None:
    """A silently empty checkpoint would fail later, inside the model loader."""
    import torch

    src = tmp_path / "junk.pth"
    torch.save({"global_step": 1, "args": {}}, src)
    with pytest.raises(ValueError, match="encoder"):
        slim_checkpoint(tmp_path / "slim.pth", src=src)


def test_slim_checkpoint_never_downloads_when_src_is_given(
    synthetic_full_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD R17.9 — a 1.13 GB cold-start download is unacceptable."""
    import huggingface_hub

    def _forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("slim_checkpoint downloaded despite an explicit src")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _forbidden)
    slim_checkpoint(tmp_path / "slim.pth", src=synthetic_full_checkpoint)


# --------------------------------------------------------------------------
# AC-3.4 — the real model on the real crop
# --------------------------------------------------------------------------


@pytest.fixture()
def ac_3_4_structure_crop(tmp_path: Path) -> Path:
    """The structure cell of compound 5 (PRD §9.4), cut from the de-rotated page."""
    from PIL import Image

    left, top, right, bottom = (round(v * AC_3_4_DPI_SCALE) for v in AC_3_4_BBOX_200DPI)
    page = Image.open(FIXTURES / "pages" / "p-063-000.png").convert("L")
    crop = page.rotate(TABLE1_DEROTATION, expand=True).crop((left, top, right, bottom))
    out = tmp_path / "compound-5-structure.png"
    crop.save(out)
    return out


@pytest.mark.slow
@pytest.mark.skipif(not SLIM_CKPT.is_file(), reason="run `sarmine slim-checkpoint` first")
def test_ac_3_4_reference_compound_yields_the_expected_inchikey(
    ac_3_4_structure_crop: Path,
) -> None:
    """AC-3.4 — page 63, first data row -> WZPDSZGYLXZFEK-UHFFFAOYSA-N."""
    with MolScribeRunner(ckpt=SLIM_CKPT) as runner:
        (result,) = runner.predict([ac_3_4_structure_crop])

    assert result.error is None
    assert result.rdkit_valid is True
    assert result.inchikey == AC_3_4_INCHIKEY
    # AC-3.3 — all three confidence levels are populated.
    assert result.confidence_molecule is not None
    assert result.confidence_min_atom is not None
    assert result.confidence_min_bond is not None
    assert result.n_atoms == 39
    assert result.n_bonds == 44
