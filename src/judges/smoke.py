"""Smoke test of one judge on a few hacking answers and normal answers.

Usage:
    .venv/bin/python -m src.judges.smoke [judge_name] [per_family]

Defaults to xverify_0p5b with 2 hacking answers per family. Uses questions whose
reference answer is an integer. Gibberish rows use real random gibberish.
This is a check that prompts, generation and parsing work end to end, not an
estimate of false positive rates.
"""

import random
import sys
from pathlib import Path

import pandas as pd

from src.detectors.synthetic import random_gibberish
from src.judges.local import LocalJudge
from src.judges.prompts import judge_config

ROOT = Path(__file__).resolve().parents[2]
HACKING = ROOT / "data" / "raw" / "hacking" / "hacking_data.parquet"


def build_items(per_family, seed=0):
    df = pd.read_parquet(HACKING)
    df = df[df["gt"].astype(str).str.strip().str.fullmatch(r"-?\d+")]
    items = []
    for label, g in df.groupby("hacking_type"):
        for i, (_, r) in enumerate(g.sample(per_family, random_state=seed).iterrows()):
            family = label.replace("tiger_", "")
            response = random_gibberish(seed * 1000 + i) if family == "gibberish" else str(r["pred"])
            items.append({"kind": family, "question": r["question"], "reference": str(r["gt"]).strip(),
                          "response": response, "should_accept": False})
    rng = random.Random(seed)
    for _, r in df.drop_duplicates("question").sample(4, random_state=seed).iterrows():
        gt = str(r["gt"]).strip()
        wrong = str(int(gt) + rng.choice([-3, -1, 1, 2]))
        items.append({"kind": "normal_correct", "question": r["question"], "reference": gt,
                      "response": f"Working through the problem gives {gt}. \\boxed{{{gt}}}", "should_accept": True})
        items.append({"kind": "normal_wrong", "question": r["question"], "reference": gt,
                      "response": f"Working through the problem gives {wrong}. \\boxed{{{wrong}}}", "should_accept": False})
    return items


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "xverify_0p5b"
    per_family = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cfg = judge_config(name)
    judge = LocalJudge(cfg)
    print(f"judge {name} ({cfg['model']}) on {judge.device}")
    items = build_items(per_family)
    results, seconds = judge.judge(items)

    rows = []
    for it, r in zip(items, results):
        rows.append({"kind": it["kind"], "should_accept": it["should_accept"], "accepted": r.accepted,
                     "parse_fail": r.parse_fail, "label": r.label, "answer": r.answer[:40],
                     "output": r.output.strip().replace("\n", " ")[:60]})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.to_string(index=False))
    hack = df[~df["kind"].str.startswith("normal")]
    normal = df[df["kind"].str.startswith("normal")]
    print(f"\n{len(df)} items in {seconds:.1f}s ({seconds / len(df):.2f}s each)")
    print(f"parse failures: {df['parse_fail'].sum()}")
    print(f"hacking answers accepted (false positives): {hack['accepted'].sum()} / {len(hack)}")
    print(f"normal answers judged as expected: {(normal['accepted'] == normal['should_accept']).sum()} / {len(normal)}")


if __name__ == "__main__":
    main()
