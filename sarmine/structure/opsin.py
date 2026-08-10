"""Channel A — IUPAC name to structure via OPSIN 2.9.0 (PRD R9.1–R9.3, R9.5).

Every entry point here is batched. Per-call JVM startup is ~5 s per molecule, so a
54-compound patent parsed one name at a time would spend 4.5 minutes starting JVMs
and milliseconds parsing (PRD R9.2).

OPSIN's status and its `nameAppearsToBeAmbiguous` warning are free, uncorrelated
confidence signals (PRD R9.3). `py2opsin` does not expose them: it returns SMILES
only and forwards OPSIN's stderr to `warnings.warn`. OPSIN's CLI does not tag those
stderr lines with the input they belong to, so this module interleaves an
unparsable sentinel between inputs; the warnings between sentinel *k* and *k+1*
belong to name *k*.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from py2opsin import py2opsin

OpsinStatus = Literal["SUCCESS", "WARNING", "FAILURE"]

# Unparsable by construction, and its stderr line carries the token verbatim.
_SENTINEL = "zzsarminesentinelzz"
_AMBIGUOUS_PREFIX = "APPEARS_AMBIGUOUS"


@dataclass
class OpsinResult:
    name: str
    smiles: str | None
    status: OpsinStatus
    ambiguous: bool
    inchikey: str | None


@lru_cache(maxsize=1)
def opsin_version() -> str:
    """`Provenance.extractor` value for the name channel, read off the bundled jar."""
    import py2opsin as package

    jars = sorted(Path(package.__file__).parent.glob("opsin-cli-*-jar-with-dependencies.jar"))
    match = re.search(r"opsin-cli-([\d.]+)-jar", jars[0].name) if jars else None
    return f"opsin@{match.group(1)}" if match else "opsin@unknown"


def parse_names(names: Sequence[str], *, work_dir: Path | None = None) -> list[OpsinResult]:
    """Parse a whole batch in ONE OPSIN call (PRD R9.2). Output aligns 1:1 with input."""
    originals = list(names)
    if not originals:
        return []

    # A newline inside a name would desynchronise OPSIN's line-oriented input from
    # the caller's list; OCR'd name cells are routinely multi-line.
    flattened = [name.replace("\r", " ").replace("\n", " ").strip() for name in originals]

    payload: list[str] = [_SENTINEL]
    for name in flattened:
        payload += [name or _SENTINEL, _SENTINEL]

    owned = work_dir is None
    base = Path(tempfile.mkdtemp(prefix="sarmine-opsin-")) if owned else Path(work_dir)
    base.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = py2opsin(
                payload,
                output_format="SMILES",
                tmp_fpath=str(base / "opsin-batch-input.txt"),
            )
        messages = [str(entry.message) for entry in caught]
    finally:
        if owned:
            shutil.rmtree(base, ignore_errors=True)

    outputs = list(raw) if isinstance(raw, list) else []
    outputs += [""] * (len(payload) - len(outputs))
    warning_groups = _group_warnings(messages, len(flattened))

    results: list[OpsinResult] = []
    for i, name in enumerate(originals):
        smiles = (outputs[2 * i + 1] or "").strip() or None
        group = warning_groups[i]
        if smiles is None:
            status: OpsinStatus = "FAILURE"
        else:
            status = "WARNING" if group else "SUCCESS"
        results.append(
            OpsinResult(
                name=name,
                smiles=smiles,
                status=status,
                ambiguous=any(line.startswith(_AMBIGUOUS_PREFIX) for line in group),
                inchikey=smiles_to_inchikey(smiles) if smiles else None,
            )
        )
    return results


def smiles_to_inchikey(smiles: str) -> str | None:
    """PRD R9.3 — the InChIKey is computed by RDKit, never taken from OPSIN."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol) or None
    except Exception:
        return None
    finally:
        RDLogger.EnableLog("rdApp.*")


def _group_warnings(messages: Sequence[str], n_names: int) -> list[list[str]]:
    """Attribute OPSIN's untagged stderr lines to inputs by sentinel position."""
    groups: list[list[str]] = [[] for _ in range(n_names)]
    index = -1
    for line in _stderr_lines(messages):
        if _SENTINEL in line:
            index += 1
            continue
        if 0 <= index < n_names:
            groups[index].append(line)
    return groups


def _stderr_lines(messages: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        body = message.split("while parsing:", 1)[-1]
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(">"):
                stripped = stripped[1:].strip()
            if stripped:
                lines.append(stripped)
    return lines
