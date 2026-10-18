"""Rebuild the gold-audit answer key from the committed sheet, after labeling is done.

Usage:
    python -m src.gold.rebuild_key [--sheet audits/gold/gold_audit_sheet.csv] [--out audits/gold]

The original key (runs/gold-audit/key.csv) was written on the GPU machine together with the
sheet. Every field of it can be recomputed from the sheet: the gold verdict is deterministic,
and a response was truncated exactly when it is max_new_tokens long (in the audit sample 20
responses have 1536 tokens and the next longest has 1419). Population counts per stratum come
from the difficulty run (docs/findings.md, Section 3.5). Writes key.csv and strata.json.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from src.gold.audit_sheet import SHEET_DIR, stratum
from src.gold.checker import check
from src.paths import ROOT, display_path

POPULATION = {"correct_plain": 34822, "correct_nontrivial": 771, "wrong_answered": 133394, "no_answer": 2002,
              "truncated": 3771}


def rebuild(sheet_rows, tokenizer, max_new_tokens):
    rows = []
    for r in sheet_rows:
        truncated = len(tokenizer(r["response"], add_special_tokens=False)["input_ids"]) >= max_new_tokens
        g = check(r["response"], r["reference"], truncated=truncated)
        s = stratum({"truncated": truncated, "no_answer": g.no_answer, "gold_correct": g.correct,
                     "gold_extracted": g.extracted, "reference": r["reference"]})
        rows.append({"audit_id": r["audit_id"], "stratum": s, "gold_correct": g.correct,
                     "gold_extracted": g.extracted, "truncated": truncated})
    return rows


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default=str(SHEET_DIR / "gold_audit_sheet.csv"))
    ap.add_argument("--out", default=str(SHEET_DIR))
    args = ap.parse_args()
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    tokenizer = AutoTokenizer.from_pretrained(data_cfg["prompt"]["model"])
    with open(args.sheet, encoding="utf-8") as f:
        rows = rebuild(list(csv.DictReader(f)), tokenizer, data_cfg["difficulty"]["max_new_tokens"])
    out = Path(args.out)
    with (out / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audit_id", "stratum", "gold_correct", "gold_extracted", "truncated"])
        w.writeheader()
        w.writerows(rows)
    sampled = dict(Counter(r["stratum"] for r in rows))
    (out / "strata.json").write_text(json.dumps({"population": POPULATION, "sampled": sampled}, indent=2))
    print(f"sampled {sampled} -> {display_path(out / 'key.csv')}")


if __name__ == "__main__":
    main()
