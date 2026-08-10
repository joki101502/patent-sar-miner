"""Pydantic transcription of the PRD §15 data model.

Everything else in the package writes into these objects. Provenance is the
product (PRD §2.3), so `Provenance` is required on every `Measurement` and is
accumulated on every `Compound`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SourceMode = Literal["structured", "pdf_ocr"]
CrosscheckTier = Literal["AGREE_FULL", "AGREE_SKELETON", "CONFLICT", "SINGLE_SOURCE", "NONE"]
StructureSource = Literal["name", "image", "name+image", "none"]
OpsinStatus = Literal["SUCCESS", "WARNING", "FAILURE"]
StandardRelation = Literal["=", ">", "<", ">=", "<=", "~"]
CensorDirection = Literal["upper_bound", "lower_bound"]
AnomalyKind = Literal[
    "legend_contradiction",
    "cross_reference_error",
    "table_stitch_uncertain",
    "rotation_uncertain",
    "compound_number_gap",
    "unit_missing",
    "duplicate_structure",
    "detector_disagreement",
    "transcription_error",
    "outside_typical_range",
    "source_unavailable",
]
AnomalySeverity = Literal["info", "warning", "error"]


class Provenance(BaseModel):
    """PRD §15.3 — where a single extracted field came from."""

    model_config = ConfigDict(extra="forbid")

    page_no: int
    bbox: tuple[int, int, int, int]
    raster_width: int
    raster_height: int
    crop_path: str
    source: SourceMode
    extractor: str
    rotation_applied: int = 0


class Compound(BaseModel):
    """PRD §15.1 — one row of the SAR table."""

    model_config = ConfigDict(extra="forbid")

    compound_id: str
    compound_local_id: str
    compound_number: int | None = None

    # structure, per channel
    smiles_from_name: str | None = None
    smiles_from_image: str | None = None
    smiles_final: str | None = None
    structure_source: StructureSource = "none"

    # identity keys
    inchikey_full: str | None = None
    inchikey_skeleton: str | None = None
    smiles_tautomer_canonical: str | None = None

    # cross-check
    crosscheck_tier: CrosscheckTier = "NONE"
    inchikey_from_name: str | None = None
    inchikey_from_image: str | None = None
    ocsr_confidence_molecule: float | None = None
    ocsr_confidence_min_atom: float | None = None
    ocsr_confidence_min_bond: float | None = None
    opsin_status: OpsinStatus | None = None
    opsin_ambiguous: bool | None = None
    homoglyph_repair_applied: str | None = None

    # flags
    markush_detected: bool = False
    has_undefined_stereocenters: bool = False
    standardization_skipped: bool = False
    potential_duplicate: bool = False

    # computed properties
    mw: float | None = None
    clogp: float | None = None
    tpsa: float | None = None
    qed: float | None = None
    hbd_lipinski: int | None = None
    hba_lipinski: int | None = None
    rotb_strict: int | None = None
    heavy_atoms: int | None = None
    fsp3: float | None = None
    n_aromatic_rings: int | None = None

    # investment signal (PRD R12.9)
    in_examples: bool = False
    in_claims: bool = False
    in_prose: bool = False
    has_in_vivo: bool = False
    investment_reasons: list[str] = Field(default_factory=list)

    # ranking
    potency_score: int | None = None
    selectivity_score: int | None = None
    rank: int | None = None
    rank_tie_group: int | None = None
    rank_rationale: list[str] = Field(default_factory=list)

    # join bookkeeping (PRD R11.6)
    join_method: str | None = None
    join_channels: list[str] = Field(default_factory=list)
    example_local_id: str | None = None

    # provenance + reproducibility
    provenance: list[Provenance] = Field(default_factory=list)
    rdkit_version: str = ""

    @model_validator(mode="after")
    def _derive_skeleton_key(self) -> Compound:
        """PRD §5 / R9.16 — the skeleton key is the first 14 chars of the InChIKey."""
        if self.inchikey_skeleton is None and self.inchikey_full:
            object.__setattr__(self, "inchikey_skeleton", self.inchikey_full[:14])
        return self


class Measurement(BaseModel):
    """PRD §15.2 — one measured value. Storage is long: one row per measurement."""

    model_config = ConfigDict(extra="forbid")

    measurement_id: str
    compound_id: str
    assay_group_key: str

    # assay context
    assay_name_raw: str
    target_raw: str | None = None
    is_off_target: bool = False
    cell_line: str | None = None
    timepoint_h: float | None = None

    # PUBLISHED — verbatim, immutable (PRD R10.1)
    published_type: str
    published_relation: str | None = None
    published_value: str
    published_units: str | None = None
    published_text_value: str | None = None

    # STANDARDIZED
    standard_type: str
    standard_relation: StandardRelation = "="
    standard_value: float | None = None
    standard_units: str | None = None
    standard_upper_value: float | None = None
    standard_flag: bool = True
    bao_endpoint: str | None = None
    uo_units: str | None = None
    qudt_units: str | None = None

    # censoring / bins
    is_censored: bool = False
    censor_direction: CensorDirection | None = None
    bin_label_raw: str | None = None
    bin_definition: str | None = None
    bin_lower_nM: float | None = None
    bin_upper_nM: float | None = None
    bin_score: int | None = None

    # derived — only when legal
    pchembl_value: float | None = None
    pdc50_value: float | None = None
    dmax_pct: float | None = None

    # QC
    data_validity_comment: str | None = None
    potential_duplicate: bool = False
    reduced_confidence: bool = False

    provenance: Provenance

    @field_validator("published_value", mode="before")
    @classmethod
    def _published_value_is_a_string(cls, v: Any) -> str:
        """PRD R10.1 — the raw cell string must survive verbatim, always as `str`."""
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)

    @model_validator(mode="after")
    def _degrader_endpoints_stay_separate(self) -> Measurement:
        # PRD R10.4 — pDC50 must never pool with pChEMBL.
        if self.pchembl_value is not None and self.pdc50_value is not None:
            raise ValueError(
                "pchembl_value and pdc50_value must never both be set (PRD R10.4)"
            )
        # PRD R10.4 — Dmax is a paired attribute of a DC50 on the same row.
        if self.dmax_pct is not None and self.standard_value is None and self.pdc50_value is None:
            raise ValueError(
                "dmax_pct requires a paired DC50 on the same row (PRD R10.4)"
            )
        return self


class DocumentAnomaly(BaseModel):
    """PRD §15.4 — document-level issues, surfaced non-blocking in the UI."""

    model_config = ConfigDict(extra="forbid")

    kind: AnomalyKind
    severity: AnomalySeverity
    message: str
    provenance: Provenance | None = None


class BinDefinition(BaseModel):
    """A decoded letter bin (PRD R10.7)."""

    model_config = ConfigDict(extra="forbid")

    label: str
    assay: str
    lower: float | None = None
    upper: float | None = None
    units: str
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    definition_text: str = ""
    score: int | None = None


class RunManifest(BaseModel):
    """PRD §15.4 — the record of one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    pubnum: str
    run_id: str
    created_at: str
    source_mode: SourceMode
    n_pages: int = 0
    n_compounds: int = 0
    n_measurements: int = 0
    target_assay: str = "WIZ"
    off_target_assay: str | None = "ZBTB7A"
    legends: dict[str, list[BinDefinition]] = Field(default_factory=dict)
    anomalies: list[DocumentAnomaly] = Field(default_factory=list)
    stage_timings_s: dict[str, float] = Field(default_factory=dict)
    stage_peak_rss_mb: dict[str, float] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    accuracy: dict[str, float] = Field(default_factory=dict)
    page_rotations: dict[str, int] = Field(default_factory=dict)


class Bundle(BaseModel):
    """The in-memory form of one artifact bundle (PRD §15.5)."""

    model_config = ConfigDict(extra="forbid")

    root: str
    manifest: RunManifest
    compounds: list[Compound] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    anomalies: list[DocumentAnomaly] = Field(default_factory=list)
