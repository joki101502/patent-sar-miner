"""Channel B — image to structure via MolScribe (PRD §9.3, Plan Part 7).

Implements R9.7–R9.13 (the OCSR channel and its three confidence levels),
R9.9/R17.9 (the slim-checkpoint build step), R9.10 (the fork trap), R13.1
(gate on the minimum atom/bond confidence) and R17.4 (free the model).

Nothing heavyweight is imported at module scope: torch, molscribe and RDKit are
all pulled in inside the functions that need them, so `import sarmine` stays
cheap and a missing checkpoint degrades instead of raising (PRD §17.5).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sarmine.config import get_config

logger = logging.getLogger(__name__)

MOLSCRIBE_REPO = "yujieq/MolScribe"
MOLSCRIBE_CKPT_FILE = "swin_base_char_aux_1m680k.pth"

# PRD R9.9 — everything else in the published checkpoint is optimizer and
# scheduler state: ~751 MB of the 1,134 MB is useless for inference.
INFERENCE_KEYS = ("encoder", "decoder", "args")

# PRD R9.10 — `molscribe.chemistry.convert_graph_to_smiles` fans out over
# `multiprocessing.Pool(16)`. Under the spawn start method each worker
# re-executes the entry script (observed forking ~17 times), and 16 workers each
# importing RDKit blows the 1.3 GB budget. One worker is enough for a cell batch.
OCSR_POOL_WORKERS = 0  # PRD R9.10 - no process pool at all; see _install_serial_graph_conversion


@dataclass
class OcsrResult:
    """One structure recognised from one crop, with its three confidence levels."""

    smiles: str | None = None
    inchikey: str | None = None
    confidence_molecule: float | None = None
    confidence_min_atom: float | None = None
    confidence_min_bond: float | None = None
    n_atoms: int | None = None
    n_bonds: int | None = None
    rdkit_valid: bool = False
    error: str | None = None


def molscribe_version() -> str:
    """Tool identifier for `Provenance.extractor` (PRD §15.3)."""
    from importlib.metadata import PackageNotFoundError, version

    for name in ("MolScribe", "molscribe"):
        try:
            return f"molscribe@{version(name)}"
        except PackageNotFoundError:
            continue
    return "molscribe@unknown"


def slim_checkpoint(out: Path, src: Path | None = None) -> Path:
    """Strip the MolScribe checkpoint to `{encoder, decoder, args}` (PRD R9.9).

    A build step, never a runtime path: 1,134 MB -> 384 MB, and model load drops
    from 16.0 s to 1.5 s (PRD R17.9, SPIKE S7).
    """
    import torch

    if src is None:
        import huggingface_hub

        src = Path(huggingface_hub.hf_hub_download(MOLSCRIBE_REPO, MOLSCRIBE_CKPT_FILE))

    # `args` is an argparse.Namespace, so the checkpoint is not weights-only.
    states = torch.load(src, map_location="cpu", weights_only=False)
    missing = [key for key in ("encoder", "decoder") if not states.get(key)]
    if missing:
        raise ValueError(f"{src} carries no {' or '.join(missing)} weights; wrong checkpoint?")

    slim = {key: states[key] for key in INFERENCE_KEYS if key in states}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)
    logger.info(
        "slimmed %s (%.0f MB) -> %s (%.0f MB)",
        src,
        src.stat().st_size / 1e6,
        out,
        out.stat().st_size / 1e6,
    )
    return out


class MolScribeRunner:
    """Lazily-loaded MolScribe, batched, freeable (PRD R9.11, R9.12, R17.4)."""

    def __init__(self, ckpt: Path | None = None, *, batch_size: int | None = None) -> None:
        config = get_config()
        self.ckpt = Path(ckpt) if ckpt is not None else config.molscribe_ckpt
        self.batch_size = batch_size or config.ocsr_batch_size
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        """False rather than an exception when the build step has not been run."""
        return self.ckpt.is_file()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                f"MolScribe checkpoint not found at {self.ckpt}; "
                "run `sarmine slim-checkpoint` (a build step, PRD R17.9)"
            )
        import torch

        torch.set_num_threads(get_config().torch_threads)  # PRD R9.11

        from molscribe import MolScribe

        self._install_serial_graph_conversion()
        self._model = MolScribe(str(self.ckpt))

    @staticmethod
    def _install_serial_graph_conversion() -> None:
        """PRD R9.10 — remove MolScribe's process pool entirely.

        `convert_graph_to_smiles` fans its post-processing out over a
        `multiprocessing.Pool`. On a spawn platform each worker re-imports
        `__main__`, re-executing the entry script — observed forking ~17 times
        during the requirements jam. Capping the pool at one worker does not
        fix it, because one worker is still one re-import. Running the same
        `_convert_graph_to_smiles` serially is equivalent, and at 54 structures
        the pool was never buying anything anyway.
        """
        import numpy as np
        from molscribe import chemistry, interface

        if getattr(interface.convert_graph_to_smiles, "_sarmine_serial", False):
            return

        def serial(coords, symbols, edges, images=None, num_workers=OCSR_POOL_WORKERS):
            args = zip(coords, symbols, edges) if images is None else zip(coords, symbols, edges, images)
            results = [chemistry._convert_graph_to_smiles(*a) for a in args]
            if not results:
                return [], [], 0.0
            smiles_list, molblock_list, success = zip(*results)
            return list(smiles_list), list(molblock_list), float(np.mean(success))

        serial._sarmine_serial = True  # type: ignore[attr-defined]
        interface.convert_graph_to_smiles = serial  # type: ignore[assignment]

    def predict(self, image_paths: Sequence[Path]) -> list[OcsrResult]:
        """Recognise a whole batch in one call (PRD R9.12: ~3 s/structure amortized)."""
        paths = list(image_paths)
        if not paths:
            return []
        if self._model is None:
            if not self.available:
                return [
                    OcsrResult(error=f"MolScribe checkpoint missing at {self.ckpt}") for _ in paths
                ]
            self.load()

        results: list[OcsrResult] = []
        for start in range(0, len(paths), self.batch_size):
            chunk = paths[start : start + self.batch_size]
            try:
                outputs = self._model.predict_image_files(  # type: ignore[union-attr]
                    [str(p) for p in chunk],  # cv2.imread cannot take a Path
                    return_atoms_bonds=True,  # PRD R9.11
                    return_confidence=True,
                )
            except Exception as exc:
                # One bad chunk must not cost the run every other structure.
                logger.exception("MolScribe failed on a chunk of %d crops", len(chunk))
                results.extend(OcsrResult(error=f"molscribe failed: {exc}") for _ in chunk)
                continue
            results.extend(_result_from_prediction(pred) for pred in outputs)
            gc.collect()
        return results

    def free(self) -> None:
        """PRD R17.4 — drop the model so RDKit does not stack on torch."""
        self._model = None
        gc.collect()

    def __enter__(self) -> MolScribeRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.free()


def _result_from_prediction(pred: dict[str, Any]) -> OcsrResult:
    from sarmine.structure.standardize import inchikey_from_smiles

    atoms = pred.get("atoms") or []
    bonds = pred.get("bonds") or []
    atom_scores = [a["confidence"] for a in atoms if a.get("confidence") is not None]
    bond_scores = [b["confidence"] for b in bonds if b.get("confidence") is not None]

    smiles = pred.get("smiles") or None
    result = OcsrResult(
        smiles=smiles,
        confidence_molecule=pred.get("confidence"),
        # PRD R13.1 — the minimum is the signal; one wrong atom barely moves the mean.
        confidence_min_atom=min(atom_scores) if atom_scores else None,
        confidence_min_bond=min(bond_scores) if bond_scores else None,
        n_atoms=len(atoms) or None,
        n_bonds=len(bonds) or None,
    )

    if smiles is None:
        result.error = "molscribe returned no SMILES"
        return result

    inchikey = inchikey_from_smiles(smiles)
    if inchikey is None:
        # PRD EC-20 — reject the channel, keep the string, trigger review.
        result.error = "SMILES failed RDKit sanitization"
        return result

    result.inchikey = inchikey
    result.rdkit_valid = True
    return result
