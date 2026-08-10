"""Build gold/US20250368620A1.gold.json (Plan Part 13.2, PRD §3.8, §20.1).

This is the GENERALITY fixture. It is structurally the opposite of the primary
reference patent — real HTML tables instead of images, numeric DC50 values
instead of letter bins, explicit `(nM)` units in the header instead of units
buried in prose — and it is the ONLY coverage the numeric-unit normalization
path gets, because the primary patent is entirely letter bins.

Two things verified here that the implementation must handle (PRD R10.10 / EC-27):

* The activity table's header really is split across four physical <tr> rows
  with hyphenated word-wrapping: `Iso-`/`indoline`/`syn-`/`thesis`,
  `Coupl-`/`ing`/`proce-`/`dure`, `HiBiT`/`DC50`/`(nM)`.
* The compound names are hyphen-wrapped mid-token across lines
  (`isoin- doline-2-carbonyl`), so de-hyphenation must precede OPSIN.

Also recorded, because it is a trap: the activity table is sorted by POTENCY,
not by compound number, so a monotonicity check over its number column would
fire spuriously. Compound-number gaps are only meaningful in the compound table.

Run:  .venv/bin/python tools/build_gold_us.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "tests" / "fixtures" / "source" / "US20250368620A1.html"
OUT = REPO / "gold" / "US20250368620A1.gold.json"

# The four physical header rows, verbatim, and what they must reconstruct to.
SPLIT_HEADER_ROWS = [
    ["", "", "Iso-", "Coupl-", ""],
    ["", "", "indoline", "ing", "HiBiT"],
    ["Cmpd.", "", "syn-", "proce-", "DC50"],
    ["No.", "Compound Structure", "thesis", "dure", "(nM)"],
]
EXPECTED_HEADER = [
    "Cmpd. No.",
    "Compound Structure",
    "Isoindoline synthesis",
    "Coupling procedure",
    "HiBiT DC50 (nM)",
]


def cell_texts(row) -> list[str]:
    return [re.sub(r"\s+", " ", c.text_content()).strip() for c in row.xpath("./td|./th")]


def dehyphenate(text: str) -> str:
    """`isoin- doline` -> `isoindoline`. Applied before whitespace removal."""
    return re.sub(r"-\s+", "", text)


def main() -> None:
    from lxml import html as LH
    from py2opsin import py2opsin
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")

    doc = LH.parse(str(HTML))
    tables = doc.xpath('//section[@itemprop="description"]//table')

    # Compound table: number | structure image | IUPAC name.
    compounds: dict[str, dict] = {}
    for row in tables[2].xpath(".//tr"):
        cells = cell_texts(row)
        if len(cells) < 3 or not cells[0].isdigit() or not cells[2]:
            continue
        compounds[cells[0]] = {"name_as_published": cells[2]}

    # Activity table: number | structure image | synthesis | procedure | DC50 nM.
    activity: dict[str, dict] = {}
    for row in tables[4].xpath(".//tr"):
        cells = cell_texts(row)
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        try:
            dc50 = float(cells[4])
        except ValueError:
            continue
        activity[cells[0]] = {
            "isoindoline_synthesis": cells[2],
            "coupling_procedure": cells[3],
            "HiBiT DC50 (nM)": dc50,
        }

    numbers = [int(n) for n in activity]
    names = {n: re.sub(r"\s+", "", dehyphenate(c["name_as_published"])) for n, c in compounds.items()}
    order = sorted(names, key=int)
    smiles_list = py2opsin([names[n] for n in order], output_format="SMILES")

    n_parsed = 0
    for n, smi in zip(order, smiles_list):
        if not smi:
            compounds[n]["parse_status"] = "FAILURE"
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            compounds[n]["parse_status"] = "INVALID"
            continue
        n_parsed += 1
        compounds[n].update(
            {
                "name_dehyphenated": names[n],
                "smiles": smi,
                "inchikey": Chem.MolToInchiKey(mol),
                "parse_status": "SUCCESS",
                "verification": "opsin_on_html_table_name",
            }
        )

    gold = {
        "pubnum": "US20250368620A1",
        "title": "ARNT Degrading Compounds and Uses Thereof",
        "gold_version": "1.0",
        "role": "generality fixture — real HTML tables, numeric values, explicit units "
        "(PRD §3.8). Structurally opposite to WO2024097932A1.",
        "provenance": {
            "tables": "Parsed directly from the real <table> elements in the Google Patents "
            "description (no OCR involved).",
            "structures": "OPSIN over the HTML table names after de-hyphenating the "
            "line-wrapped tokens; every structure RDKit-valid.",
        },
        "assays": [
            {
                "name": "HiBiT DC50 (nM)",
                "target": "ARNT",
                "kind": "numeric",
                "standard_type": "DC50",
                "units": "nM",
                "role": "target",
            }
        ],
        "split_header": {
            "physical_rows": SPLIT_HEADER_ROWS,
            "expected_reconstruction": EXPECTED_HEADER,
            "note": "PRD R10.10 / EC-27 — join adjacent header rows, then de-hyphenate.",
        },
        "compounds": dict(sorted(compounds.items(), key=lambda kv: int(kv[0]))),
        "activity": dict(sorted(activity.items(), key=lambda kv: int(kv[0]))),
        "counts": {
            "n_compounds": len(compounds),
            "n_activity_rows": len(activity),
            "n_names_parsed": n_parsed,
            "dc50_min_nM": min(v["HiBiT DC50 (nM)"] for v in activity.values()) if activity else None,
            "dc50_max_nM": max(v["HiBiT DC50 (nM)"] for v in activity.values()) if activity else None,
            "activity_table_is_number_sorted": numbers == sorted(numbers),
        },
        "expectations": {
            "pdc50_populated": True,
            "pchembl_null": True,
            "note": "PRD R10.4 / AC-4.5 — DC50 gets pdc50_value; pchembl_value must stay null.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  compounds={len(compounds)} activity_rows={len(activity)} names_parsed={n_parsed}")
    print(f"  header reconstructs to: {EXPECTED_HEADER}")
    print(f"  activity table sorted by compound number: {numbers == sorted(numbers)}")
    print(f"  DC50 range: {gold['counts']['dc50_min_nM']} .. {gold['counts']['dc50_max_nM']} nM")


if __name__ == "__main__":
    main()
