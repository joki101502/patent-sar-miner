"""Build gold/WO2024097932A1.gold.json (Plan Part 13.1, PRD §20.1).

Two halves, derived by two different and separately-recorded methods:

* The activity half is transcribed verbatim from PRD Appendix B.1, which was
  hand-read from the source images `imgf000186_0001.png` (rows 1-31) and
  `imgf000187_0001.png` (rows 32-54). It is hand-checked ground truth.
* The Examples half is derived by running OPSIN over the IUPAC names in the
  machine-readable description. Those names are themselves OCR output and carry
  the same l/1 corruption as the scan, so they are repaired first; every
  resulting structure is then required to be RDKit-valid. This is recorded on
  each row as `verification: "opsin_on_description_text"` — it is strong, but it
  is NOT the same standard as a chemist redrawing the structure, and
  `sarmine evaluate` must not present it as if it were.

Run:  .venv/bin/python tools/build_gold_wo.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "tests" / "fixtures" / "source" / "WO2024097932A1.html"
OUT = REPO / "gold" / "WO2024097932A1.gold.json"

PARA = r"\[\s*\d{5}\s*\]"  # paragraph markers appear as both [00203] and [00203 ]

# --- PRD Appendix B.1, verbatim. "-" denotes a blank cell in the source. -----
ACTIVITY_TABLE = """
1  A E G   19 A F I   37 - D G
2  A D G   20 B D I   38 - E G
3  A E G   21 B D H   39 B E H
4  A E H   22 B E I   40 A D G
5  A E H   23 B F I   41 A D G
6  A D H   24 B F I   42 B F H
7  A E H   25 B E I   43 A D G
8  A E H   26 C E I   44 A E H
9  A D H   27 C E I   45 A D G
10 A D I   28 C F I   46 A D G
11 A F I   29 C E I   47 A D G
12 A D H   30 C E H   48 A D G
13 A E H   31 B E I   49 - D G
14 A F I   32 A D H   50 - E G
15 A D H   33 - E H   51 - E H
16 A D I   34 - E G   52 A D I
17 A E H   35 - E H   53 - F I
18 A D H   36 - E G   54 - F I
"""

# --- PRD R10.7. The DEFINITIONAL sentence wins wherever the patent's own
# --- summary sentence contradicts it (PRD R10.6 / EC-5).
LEGENDS = {
    "HbF Induction (%)": [
        {"label": "A", "lower": 66.0, "upper": 100.0, "units": "%", "score": 3},
        {"label": "B", "lower": 33.0, "upper": 66.0, "units": "%", "score": 2},
        {"label": "C", "lower": None, "upper": 33.0, "units": "%", "score": 1},
    ],
    "WIZ EC50 (uM)": [
        {"label": "D", "lower_nM": None, "upper_nM": 10.0, "units": "nM", "score": 3},
        {"label": "E", "lower_nM": 10.0, "upper_nM": 100.0, "units": "nM", "score": 2},
        {"label": "F", "lower_nM": 100.0, "upper_nM": None, "units": "nM", "score": 1},
    ],
    "ZBTB7A EC50 (uM)": [
        {"label": "G", "lower_nM": None, "upper_nM": 30.0, "units": "nM", "score": 3},
        {"label": "H", "lower_nM": 30.0, "upper_nM": 100.0, "units": "nM", "score": 2},
        {"label": "I", "lower_nM": 100.0, "upper_nM": None, "units": "nM", "score": 1},
    ],
}

# PRD §3.5 — three genuine self-contradictions between the definitional and the
# summary sentence, plus the Table 1/Table 2 cross-reference error (EC-6).
LEGEND_CONTRADICTIONS = [
    {
        "assay": "HbF Induction (%)",
        "bin": "A",
        "definitional": "between 66% and 100%",
        "summary": "between 67% and 100%",
        "resolution": "definitional",
    },
    {
        "assay": "WIZ EC50 (uM)",
        "bin": "F",
        "definitional": "> 0.1 uM",
        "summary": "< .01 uM",
        "resolution": "definitional",
    },
    {
        "assay": "ZBTB7A EC50 (uM)",
        "bin": "G",
        "definitional": "< 0.03 uM",
        "summary": "< .01 M",
        "resolution": "definitional",
    },
]

CROSS_REFERENCE_ERRORS = [
    {
        "paragraph": "[00508]",
        "says": "Table 1",
        "means": "Table 2",
        "detail": "The HbF/viability results described as being 'in Table 1 below' are in Table 2.",
    }
]

# PRD §9.4 / Appendix B.2 — the one structure verified by hand, on both channels.
HAND_CHECKED_COMPOUNDS = {
    "5": {
        "name": (
            "2-(2,6-dioxopiperidin-3-yl)-4-((5-(3-fluoro-4-methoxyphenyl)-1-methyl-"
            "1H-benzo[d]imidazol-6-yl)amino)isoindoline-1,3-dione"
        ),
        "inchikey": "WZPDSZGYLXZFEK-UHFFFAOYSA-N",
        "smiles_opsin": (
            "O=C1NC(CCC1N1C(C2=CC=CC(=C2C1=O)NC=1C(=CC2=C(N(C=N2)C)C1)"
            "C1=CC(=C(C=C1)OC)F)=O)=O"
        ),
        "smiles_molscribe": (
            "COc1ccc(-c2cc3ncn(C)c3cc2Nc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)cc1F"
        ),
        "page_no": 63,
        "example": "14",
        "verification": "hand_checked",
    }
}


def parse_activity() -> dict[str, dict[str, str | None]]:
    rows: dict[str, dict[str, str | None]] = {}
    for line in ACTIVITY_TABLE.strip().splitlines():
        tokens = line.split()
        for i in range(0, len(tokens), 4):
            num, hbf, wiz, zbtb = tokens[i : i + 4]
            rows[num] = {
                "HbF Induction (%)": None if hbf == "-" else hbf,
                "WIZ EC50 (uM)": None if wiz == "-" else wiz,
                "ZBTB7A EC50 (uM)": None if zbtb == "-" else zbtb,
            }
    return dict(sorted(rows.items(), key=lambda kv: int(kv[0])))


def repair_ocr_name(name: str) -> str:
    """The description text is OCR output: `l` stands in for `1` throughout, and
    stereo descriptors are mangled. Measured: 0/32 names parse untouched."""
    name = re.sub(r"\(7\?\)|\(l\?\)|\(R\?\)", "(R)", name)
    name = re.sub(r"\(5\)", "(S)", name)
    name = re.sub(r"l(?=H[-\]])", "1", name)
    name = re.sub(r"(?<=[-(\[,])l(?=[,\)\]-])", "1", name)
    return name


def extract_examples() -> dict[str, dict[str, str]]:
    from lxml import html as LH
    from py2opsin import py2opsin
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")

    doc = LH.parse(str(HTML))
    flat = re.sub(r"\s+", " ", doc.xpath('//section[@itemprop="description"]')[0].text_content())

    raw: dict[int, str] = {}
    for m in re.finditer(rf"Example\s+(\d+)\s*:\s*(.+?)(?={PARA}|\s*$)", flat):
        name = re.sub(rf"{PARA}.*$", "", m.group(2)).replace("\u2013", "-").replace("\u2014", "-")
        raw[int(m.group(1))] = re.sub(r"\s+", "", name).strip()

    # `&`-joined enantiomer pairs: the patent lists both; keep the first as the
    # representative and record the pair (PRD R9.22 — never enumerate stereo).
    prepared = {n: repair_ocr_name(v) for n, v in raw.items()}
    order = sorted(prepared)
    smiles_list = py2opsin([prepared[n].split("&")[0] for n in order], output_format="SMILES")

    out: dict[str, dict[str, str]] = {}
    for n, smi in zip(order, smiles_list):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[str(n)] = {
            "name_as_published": raw[n],
            "name_repaired": prepared[n].split("&")[0],
            "is_enantiomer_pair": "&" in prepared[n],
            "smiles": smi,
            "inchikey": Chem.MolToInchiKey(mol),
            "verification": "opsin_on_description_text",
        }
    return out


def main() -> None:
    activity = parse_activity()
    examples = extract_examples()

    letters: dict[str, dict[str, int]] = {}
    for assay in ("HbF Induction (%)", "WIZ EC50 (uM)", "ZBTB7A EC50 (uM)"):
        counts: dict[str, int] = {}
        for row in activity.values():
            key = row[assay] or "blank"
            counts[key] = counts.get(key, 0) + 1
        letters[assay] = dict(sorted(counts.items()))

    blanks = [n for n, r in activity.items() if r["HbF Induction (%)"] is None]
    selective = [
        n
        for n, r in activity.items()
        if r["WIZ EC50 (uM)"] == "D" and r["ZBTB7A EC50 (uM)"] == "I"
    ]

    gold = {
        "pubnum": "WO2024097932A1",
        "title": "COMPOUNDS AND THEIR USE FOR TREATMENT OF HEMOGLOBINOPATHIES",
        "gold_version": "1.0",
        "provenance": {
            "activity": "Transcribed verbatim from PRD Appendix B.1 (hand-read from "
            "imgf000186_0001.png rows 1-31 and imgf000187_0001.png rows 32-54).",
            "legends": "PRD R10.7. Where the patent's definitional and summary sentences "
            "contradict, the definitional sentence is authoritative (PRD R10.6).",
            "compounds": "Only compound 5 is hand-checked (PRD Appendix B.2, agreed by both "
            "channels). Other Table 1 structures are NOT in this gold set.",
            "examples": "OPSIN over the machine-readable description names after l->1 "
            "homoglyph repair; every structure RDKit-valid. NOT hand-drawn.",
        },
        "assays": [
            {"name": "HbF Induction (%)", "target": "HbF", "kind": "letter_bin", "role": "phenotypic"},
            {"name": "WIZ EC50 (uM)", "target": "WIZ", "kind": "letter_bin", "role": "target"},
            {"name": "ZBTB7A EC50 (uM)", "target": "ZBTB7A", "kind": "letter_bin", "role": "off_target"},
        ],
        "cell_line": "HUDEP-2",
        "legends": LEGENDS,
        "legend_contradictions": LEGEND_CONTRADICTIONS,
        "cross_reference_errors": CROSS_REFERENCE_ERRORS,
        "activity": activity,
        "compounds": HAND_CHECKED_COMPOUNDS,
        "examples": examples,
        "compound_to_example": {"5": "14"},
        "counts": {
            "n_compounds": len(activity),
            "n_activity_cells": len(activity) * 3,
            "n_examples": len(examples),
            "blank_hbf_compounds": blanks,
            "letter_distribution": letters,
            "max_selectivity_compounds": selective,
            "has_in_vivo_data": False,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  compounds={len(activity)} cells={len(activity)*3} examples={len(examples)}")
    print(f"  blank HbF: {blanks}")
    print(f"  letters: {letters}")
    print(f"  selectivity +2: {selective}")


if __name__ == "__main__":
    main()
