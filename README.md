# Patent SAR Miner

Takes one chemistry patent and produces a **joined SAR table** — one row per
compound with a machine-readable structure (SMILES), at least one bioactivity
value, and a page-level provenance pointer for every field — plus a **ranked
shortlist** of the compounds a medicinal chemist should look at first and a
**review queue** of the extractions the system is not confident about.

---

## The idea

Chemistry patents usually state each compound **twice** — once as a drawn
structure and once as an IUPAC name. Those are two completely independent
encodings of the same molecule. Convert both to InChIKeys and compare them:

```
name  (OPSIN)     -> O=C1NC(CCC1N1C(C2=CC=CC(=C2C1=O)NC=1C(=CC2=C(N(C=N2)C)C1)C1=CC(=C(C=C1)OC)F)=O)=O
image (MolScribe) -> COc1ccc(-c2cc3ncn(C)c3cc2Nc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)cc1F

InChIKey (name)  : WZPDSZGYLXZFEK-UHFFFAOYSA-N
InChIKey (image) : WZPDSZGYLXZFEK-UHFFFAOYSA-N   ← agreement from two orthogonal derivations
```

The SMILES strings are textually unrelated; only the InChIKey reveals the match.
When the two channels agree you have near-certainty; when they disagree you have
a precise, actionable review item. As far as an extensive literature search could
establish, this cross-modal `OPSIN(name) ≟ OCSR(image)` gate is unpublished.

The second load-bearing idea is much less glamorous: **segment before you read.**
OCR'ing a whole Table 1 page and feeding the text to OPSIN parsed **0 of 61**
names, because atom labels from the structure drawing interleave into the name.
Isolating the name sub-cell first parsed **33 of 37**. Same page, same OCR engine,
same parser.

---

## Measured results on the reference patent

WO 2024/097932 A1 (Bristol-Myers Squibb) — 223 pages, **no text layer at all**
(223 characters in the whole document), every page a 1-bit bitonal 300 dpi scan.

| Metric | Measured |
|---|---|
| End-to-end triplet accuracy (compound + assay + value all correct) | **F1 0.990** — precision **1.000**, recall 0.980 |
| Activity cells | **148 / 151 correct, 0 wrong, 0 spurious, 3 missing** |
| Compound numbers recovered | 54 of 54 |
| Compound–activity join | 53 of 54 |
| Legends recovered | 3 of 3, with all contradictions flagged |
| Compounds carrying an Examples synthesis | 32 of 54 |
| Wall clock, whole run | **251 s** (budget: 15 min) |
| Peak RSS | **1693 MB** (budget: 2400 MB) |
| Slim MolScribe checkpoint | **384 MB**, down from 1134 MB |

Reproduce with `sarmine run data/patents/WO2024097932A1.pdf --out artifacts/`,
then `sarmine evaluate artifacts/WO2024097932A1/<run-id> --gold gold/WO2024097932A1.gold.json`.

**Read the precision number as the important one.** Nothing the tool asserts
about an activity value is wrong; where it is unsure it declines and queues the
row for review. That is the design goal — at the accuracy the published
literature achieves on whole patent documents (IBM's PatCID measured **63.0%**
end-to-end on recent US patents), a table you cannot audit is worse than no table.

Two honest caveats:

- **The structure gold set is thin.** Only compound 5 was hand-verified against
  both channels, so the reported structure accuracy is indicative rather than a
  benchmark result. The 32 Example names come from the machine-readable
  description and are resolved deterministically by OPSIN.
- **The image channel agrees with the name channel on about 40% of compounds**
  (20 `AGREE_FULL` of 52 with both channels). That is in the band the literature
  predicts for whole-document OCSR on bitonal scans, and it is why the
  cross-check exists: every disagreement becomes a review item rather than a
  silently wrong molecule. Upsampling and blurring the bitonal crops was measured
  and changed nothing (8/16 correct for every variant tried).

---

## Install

Python **3.11 exactly**.

System binaries:

| Binary | Purpose | macOS | Linux |
|---|---|---|---|
| `pdftoppm`, `pdfimages`, `pdfinfo`, `pdffonts` | PDF rasterization | `brew install poppler` | `poppler-utils` |
| `tesseract` | OCR | `brew install tesseract` | `tesseract-ocr`, `tesseract-ocr-eng` |
| `java` | OPSIN runtime | `brew install openjdk` | `default-jre` |

> ⚠️ On macOS `/usr/bin/java` is a **stub** that prints an install prompt and
> exits. It is not a JRE, and `py2opsin` fails confusingly against it. Put a real
> JDK on `PATH` (`export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"`) or set
> `SARMINE_JAVA_BIN`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps MolScribe      # its published pin torch<2.0 is unsatisfiable today
pip install -e .
sarmine slim-checkpoint              # build step: 1134 MB -> 384 MB, load 16.0 s -> 1.5 s
```

`MolScribe` is installed separately and with `--no-deps` on purpose: it pins
`torch<2.0`, and the oldest torch on PyPI is now 2.0.0, so a normal install
cannot resolve. Two of its other pins are **real** and must not be relaxed —
`albumentations==1.1.0` and `timm==0.4.12` both provide private symbols removed
in their 2.x/1.x releases.

There are **no API keys, no database and no accounts**. Every external touchpoint
is anonymous, by design.

---

## Use

```bash
# a PDF, or a publication number
sarmine run data/patents/WO2024097932A1.pdf
sarmine run WO2024097932A1 --target WIZ --off-target ZBTB7A
sarmine run patent.pdf --force-pdf-path        # skip patents.google.com entirely

sarmine evaluate artifacts/<pubnum>/<run-id> --gold gold/<pubnum>.gold.json
sarmine slim-checkpoint

streamlit run app/streamlit_app.py
```

`--target` / `--off-target` are the only settings that change *results* rather
than plumbing: they set the sign of the selectivity score, and swapping them
ranks the wrong compounds first.

---

## How it works

```
1. Source resolve      publication number from page 1; patents.google.com if reachable
2. Pages               pdfimages -png (never a re-render — resampling destroys thin bonds)
3. Segment             ruling-line morphology + Table Transformer, reconciled; rotation
                       detected then VERIFIED by re-OCR quality
4. OCR                 per-cell, per-role: names to Tesseract, drawings never
5. Name channel        one batched OPSIN call + homoglyph repair
6. Assay               legend parsing, unit normalization, bins as intervals
7. Image channel       MolScribe on clean crops only, loaded last and freed
8. Cross-check         InChIKey tiering
9. Rank                potency and selectivity as separate axes
10. Write              JSON + crops + rendered structures
```

The order is a **memory requirement, not a style preference**: peak must equal
the largest stage, not the sum. Peak RSS is recorded at every stage boundary into
`manifest.stage_peak_rss_mb`, so the budget is a measured number in the artifacts
rather than an assumption.

### Things that are deliberately refused

- **Markush structures are detected and flagged, never enumerated.** The state of
  the art reaches ~13% exact match on real patent images, and the recognition
  models do not enumerate at all. A Markush image fails OPSIN outright while OCSR
  will happily hallucinate a concrete structure — that combination would attach a
  confidently wrong molecule to a real activity value.
- **Letter bins are stored as intervals; no midpoint is ever imputed.** A bin is
  interval-censored data, and a midpoint would fabricate precision the patent
  does not contain.
- **Units are never inferred from magnitude.** A missing unit is refused and
  queued. A silent nM/µM confusion is a 1000× error in a potency ranking.
- **Compound numbers are never interpolated.** A gap is an anomaly; an invented
  number silently corrupts the join.
- **No LLM or VLM reads a structure or a table.** On the closest published proxy
  to this document class, frontier models score in the high teens; on
  IUPAC→SMILES, GPT-4 scores 1.4% against OPSIN's 99.4%. They produce
  plausible-looking but structurally wrong SMILES, which is the single most
  dangerous failure mode for SAR work.
- **No TensorFlow anywhere**, which keeps the app inside a free tier's memory.

---

## Deployment

**Primary target: Streamlit Community Cloud (free).** `requirements.txt` plus
`packages.txt` (`tesseract-ocr`, `tesseract-ocr-eng`, `default-jre`,
`poppler-utils`, `libgl1`).

> ⚠️ Deploy a stub first. Streamlit's published limits (0.078–2 cores,
> 690 MB–2.7 GB RAM) are dated February 2024 in their own docs and could not be
> re-verified. Push a minimal app that only imports torch and loads the slim
> checkpoint, confirm it runs, and only then deploy the full app.

The slim checkpoint must be produced at **build time** and baked in. A 1.13 GB
cold-start download is not acceptable, and `models/` is gitignored.

**Fallback: Render Pro** (4 GB / 2 CPU). Render Standard is explicitly rejected —
at 2 GB it has *less* RAM than the free option.

### Productization caveat — patents.google.com

The accelerated path scrapes `patents.google.com` with a browser User-Agent and
caches the result. It is an **accelerator, not a dependency**: the full PDF OCR
path is completely implemented and tested with the network disabled, and any
fetch failure degrades to it silently and records `source_mode="pdf_ocr"`.

**The terms-of-service position for this access is unverified.** Before running
this as a product, either confirm that use is permitted or replace the accelerated
path with a licensed source (EPO OPS serves WO full text per document; note that
no EPO *bulk* product contains WO full text, and BigQuery's `description_localized`
is US-only). Nothing in the system breaks if that path is removed — it only gets
slower, because roughly 180 pages of synthesis prose then have to be OCR'd.

---

## Tests

```bash
pytest                                    # fast, hermetic
pytest -m slow                            # models + the reference PDF
SARMINE_TEST_NETWORK=1 pytest -m network  # live patents.google.com and PubChem
```

727 tests. The slow set includes the acceptance checks that matter: both channels
reproducing `WZPDSZGYLXZFEK-UHFFFAOYSA-N` for compound 5, Table 2 stitching across
pages 186–187 despite page 187 having no header row, and rotation being corrected
on a page whose OSD confidence is 13.62.

---

## Limitations

Single user, no authentication, one document at a time. Reviewer corrections live
in the browser session — the free host has no persistent disk, so export before
closing the tab. The artifact schema is designed so a database is a later drop-in.

**Output is decision support, not legal advice.**
