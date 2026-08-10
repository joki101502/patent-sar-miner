"""Typed configuration (Plan Part 1.4).

Every value has a code default; `.env.local` is optional and the app must run
correctly without it (Plan Part 0.4). No secrets exist anywhere in this system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON = Path(__file__).resolve().parent / "data" / "assay_lexicon.yaml"


def _load_dotenv_local() -> None:
    """Load `.env.local` if present. Absence is normal and not an error."""
    path = REPO_ROOT / ".env.local"
    if not path.is_file():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv_local()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # --- assay roles: the only settings that change results, not plumbing (PRD R12.5)
    target_assay: str = field(default_factory=lambda: _env("SARMINE_TARGET_ASSAY", "WIZ"))
    off_target_assay: str | None = field(
        default_factory=lambda: _env("SARMINE_OFF_TARGET_ASSAY", "ZBTB7A") or None
    )

    # --- paths
    artifact_root: Path = field(
        default_factory=lambda: Path(_env("SARMINE_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts")))
    )
    molscribe_ckpt: Path = field(
        default_factory=lambda: Path(
            _env("SARMINE_MOLSCRIBE_CKPT", str(REPO_ROOT / "models" / "molscribe_slim.pth"))
        )
    )
    assay_lexicon: Path = field(
        default_factory=lambda: Path(_env("SARMINE_ASSAY_LEXICON", str(DEFAULT_LEXICON)))
    )
    gold_dir: Path = field(default_factory=lambda: REPO_ROOT / "gold")

    # --- thresholds
    # TODO: calibrate against the gold set in Part 13 (PRD R13.2). Gate on the
    # MINIMUM atom/bond confidence, never the molecule mean (PRD R13.1).
    ocsr_conf_threshold: float = field(
        default_factory=lambda: _env_float("SARMINE_OCSR_CONF_THRESHOLD", 0.85)
    )
    rotation_score_threshold: float = field(
        default_factory=lambda: _env_float("SARMINE_ROTATION_SCORE_THRESHOLD", 0.30)
    )
    heavy_atom_review_threshold: int = 70  # PRD EC-13
    homoglyph_max_candidates: int = 200  # PRD R9.4
    header_fuzzy_threshold: float = field(
        default_factory=lambda: _env_float("SARMINE_HEADER_FUZZY_THRESHOLD", 0.82)
    )
    noise_floor_log_units: float = 0.3  # PRD R10.16

    # --- binaries
    tesseract_bin: str = field(default_factory=lambda: _env("SARMINE_TESSERACT_BIN", "tesseract"))
    java_bin: str = field(default_factory=lambda: _env("SARMINE_JAVA_BIN", "java"))
    poppler_bin_dir: str = field(default_factory=lambda: _env("SARMINE_POPPLER_BIN_DIR", ""))

    # --- network
    allow_network: bool = field(
        default_factory=lambda: _env("SARMINE_ALLOW_NETWORK", "1") not in {"0", "false", "no"}
    )
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    request_timeout_s: float = 30.0

    # --- torch
    torch_threads: int = 2  # PRD R9.11
    # OCSR activations are live for a whole chunk at once. Measured: ~47 crops in
    # one call peaks at 2843 MB against a 2400 MB budget (AC-9.3), so inference is
    # chunked while the model itself is still loaded only once (R9.12).
    ocsr_batch_size: int = field(
        default_factory=lambda: max(1, int(_env_float("SARMINE_OCSR_BATCH_SIZE", 4)))
    )

    def poppler(self, tool: str) -> str:
        return str(Path(self.poppler_bin_dir) / tool) if self.poppler_bin_dir else tool


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def set_config(cfg: Config) -> None:
    global _CONFIG
    _CONFIG = cfg
