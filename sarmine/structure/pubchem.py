"""PubChem PUG-REST fallback for trivial and trade names (PRD R9.6, Plan 5.4).

This is a FALLBACK, never a primary: OPSIN structurally cannot parse `aspirin`,
`pomalidomide` or `Lipitor`, and only those names come here. Answers are cached on
disk so a repeat run performs zero network requests (AC-1.4).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from sarmine.config import get_config

ENDPOINT = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/SMILES,InChIKey/JSON"
)

# PRD R9.6 — PubChem's published limits are <=5 requests/second and <=400/minute.
MIN_REQUEST_INTERVAL_S = 0.21
MAX_REQUESTS_PER_MINUTE = 400

# PubChem renamed this property; code written against older docs KeyErrors.
SMILES_KEYS = ("SMILES", "ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES")

_REQUEST_TIMES: deque[float] = deque()


def reset_throttle() -> None:
    _REQUEST_TIMES.clear()


def resolve_name(
    name: str, cache_dir: Path | None = None, *, allow_network: bool = True
) -> tuple[str | None, str | None]:
    """Return `(smiles, inchikey)` for a trivial name, or `(None, None)`."""
    key = (name or "").strip()
    if not key:
        return (None, None)

    cache_path = _cache_path(key, cache_dir)
    cached = _read_cache(cache_path)
    if cached is not None:
        return parse_property_payload(cached)

    if not allow_network:
        return (None, None)

    _throttle()
    try:
        payload = _fetch(key)
    except Exception:
        # A dead fallback must not take the run down (EC-21 posture).
        return (None, None)

    smiles, inchikey = parse_property_payload(payload)
    if smiles or inchikey:
        _write_cache(cache_path, payload)
    return (smiles, inchikey)


def parse_property_payload(payload: Any) -> tuple[str | None, str | None]:
    """Tolerate every spelling PubChem has used for the SMILES property (PRD R9.6)."""
    try:
        properties = payload["PropertyTable"]["Properties"]
    except (TypeError, KeyError):
        return (None, None)
    if not properties:
        return (None, None)

    record = properties[0]
    smiles = next((record[k] for k in SMILES_KEYS if record.get(k)), None)
    return (smiles or None, record.get("InChIKey") or None)


def _fetch(name: str) -> dict[str, Any]:
    import requests

    cfg = get_config()
    response = requests.get(
        ENDPOINT.format(name=requests.utils.quote(name, safe="")),
        headers={"User-Agent": cfg.user_agent},
        timeout=cfg.request_timeout_s,
    )
    response.raise_for_status()
    return response.json()


def _throttle() -> None:
    now = time.monotonic()
    if _REQUEST_TIMES:
        wait = MIN_REQUEST_INTERVAL_S - (now - _REQUEST_TIMES[-1])
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
    while _REQUEST_TIMES and now - _REQUEST_TIMES[0] > 60.0:
        _REQUEST_TIMES.popleft()
    if len(_REQUEST_TIMES) >= MAX_REQUESTS_PER_MINUTE:
        time.sleep(max(0.0, 60.0 - (now - _REQUEST_TIMES[0])))
        now = time.monotonic()
    _REQUEST_TIMES.append(now)


def _cache_path(name: str, cache_dir: Path | None) -> Path:
    base = Path(cache_dir) if cache_dir else get_config().artifact_root / "cache" / "pubchem"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "name"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return base / f"{slug}-{digest}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), "utf-8")
    except OSError:
        pass
