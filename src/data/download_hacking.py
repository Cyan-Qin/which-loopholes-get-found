"""Download the hacking set released with Huang et al. [1] and check it against HANDOFF.md.

Usage:
    .venv/bin/python -I src/data/download_hacking.py

Saves data/raw/hacking/hacking_data.parquet and prints a summary.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ID = "hkust-nlp/rl-verifier-pitfalls_hacking_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "hacking"

# Expected values from HANDOFF.md section 11, plus the row index column `idx`
# that the released file also carries. `score` is False on every row: it is the
# gold label of the hacking answer, not a judge verdict.
EXPECTED_ROWS = 6123
EXPECTED_QUESTIONS = 471
EXPECTED_COLUMNS = {"idx", "question", "gt", "pred", "hacking_type", "score"}
EXPECTED_TYPES = 13

INT_RE = re.compile(r"^-?\d+$")


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    ds = load_dataset(REPO_ID)
    print("splits:", {name: split.num_rows for name, split in ds.items()})
    if len(ds) != 1:
        print("more than one split; summary below covers all splits concatenated")
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "hacking_data.parquet"
    df.to_parquet(out_path, index=False)
    print("saved:", out_path.relative_to(OUT_DIR.parents[2]))
    print()

    results = [
        check("rows", len(df) == EXPECTED_ROWS, f"{len(df)} (expected {EXPECTED_ROWS})"),
        check("columns", set(df.columns) == EXPECTED_COLUMNS,
              f"{list(df.columns)} (expected {sorted(EXPECTED_COLUMNS)})"),
    ]
    if "question" in df:
        n_q = df["question"].nunique()
        results.append(check("unique questions", n_q == EXPECTED_QUESTIONS,
                             f"{n_q} (expected {EXPECTED_QUESTIONS})"))
    if "hacking_type" in df:
        n_t = df["hacking_type"].nunique()
        results.append(check("hacking types", n_t == EXPECTED_TYPES, f"{n_t} (expected {EXPECTED_TYPES})"))

    print("\nhacking_type counts:")
    print(df["hacking_type"].value_counts().to_string())

    print("\nscore values:")
    print(df["score"].value_counts(dropna=False).to_string())

    print("\nmean score by hacking_type:")
    print(df.groupby("hacking_type")["score"].mean().sort_values(ascending=False).round(3).to_string())

    gt = df.drop_duplicates("question")["gt"].astype(str).str.strip()
    n_int = gt.str.fullmatch(INT_RE.pattern).sum()
    print(f"\nquestions whose gt is a plain integer: {n_int} / {len(gt)}")
    print("examples of non-integer gt:", gt[~gt.str.fullmatch(INT_RE.pattern)].head(8).tolist())

    per_q = df.groupby("question").size()
    print(f"\nrows per question: min {per_q.min()}, median {int(per_q.median())}, max {per_q.max()}")
    print(f"empty or whitespace-only pred: {(df['pred'].astype(str).str.strip() == '').sum()}")

    print("\none example per hacking_type (pred truncated to 120 chars):")
    for t, g in df.groupby("hacking_type"):
        row = g.iloc[0]
        pred = repr(str(row["pred"]))[:120]
        print(f"- {t} | gt={row['gt']!r} | score={row['score']} | pred={pred}")

    print()
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
