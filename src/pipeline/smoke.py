"""End-to-end smoke test on a few problems with small local models.

Usage:
    .venv/bin/python -m src.pipeline.smoke [--n-problems 5] [--n-samples 4]
        [--policy Qwen/Qwen2.5-0.5B-Instruct] [--judges xverify_0p5b,qwen2p5_math_1p5b_instruct]
        [--max-new-tokens N] [--seed 0]

Picks problems from the MATH part of the candidate pool (levels 1-2, so a small
policy gets some right), samples the policy, gold-checks, classifies, judges and
writes runs/smoke-<time>/records.jsonl and meta.json. Models are loaded one at a
time to keep memory low.
"""

import argparse
import gc
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.judges.local import LocalJudge
from src.judges.prompts import judge_config
from src.pipeline.records import read_jsonl, write_jsonl, write_meta
from src.pipeline.run import judge_graded, sample_and_grade
from src.policy.local import LocalPolicy

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "processed" / "pool_candidates.jsonl"


def pick_items(n, seed):
    rows = [json.loads(line) for line in POOL.open()]
    easy = [r for r in rows if r["source"] == "math_train" and r["level"] in ("Level 1", "Level 2")]
    return [{"uid": r["uid"], "source": r["source"], "problem": r["problem"], "reference": r["answer"]}
            for r in random.Random(seed).sample(easy, n)]


def free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())["difficulty"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--policy", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--judges", default="xverify_0p5b,qwen2p5_math_1p5b_instruct")
    ap.add_argument("--max-new-tokens", type=int, default=data_cfg["max_new_tokens"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_id = "smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / run_id
    sampling = {"temperature": data_cfg["temperature"], "top_p": 1.0, "max_new_tokens": args.max_new_tokens}
    write_meta(run_dir, run_id=run_id, stage="smoke", args=vars(args), sampling=sampling)
    items = pick_items(args.n_problems, args.seed)

    policy = LocalPolicy(args.policy)
    graded, policy_seconds = sample_and_grade(items, policy, args.n_samples, sampling, args.seed)
    n_tokens = sum(g.sample.n_tokens for g in graded)
    print(f"policy {args.policy} on {policy.device}: {len(graded)} responses, {n_tokens} tokens "
          f"in {policy_seconds:.0f}s ({n_tokens / policy_seconds:.0f} tok/s)")
    del policy
    free_memory()

    timings = {}
    for name in args.judges.split(","):
        judge = LocalJudge(judge_config(name))
        records, seconds = judge_graded(graded, judge, name, run_id=run_id, stage="smoke", step=None,
                                        policy_model=args.policy)
        write_jsonl(run_dir / "records.jsonl", records)
        timings[name] = seconds
        del judge
        free_memory()

    # Read back what was written: the summary is computed from the log, not from memory.
    df = pd.DataFrame(read_jsonl(run_dir / "records.jsonl"))
    first = df[df["judge"] == df["judge"].iloc[0]]
    wide = first[["uid", "sample_idx", "n_tokens", "truncated", "gold_correct", "gold_extracted", "family"]].copy()
    for name in timings:
        wide[name.split("_")[0] + "_acc"] = df[df["judge"] == name]["accepted"].values
    wide["gold_extracted"] = wide["gold_extracted"].fillna("—").str.slice(0, 14)
    print(f"\n{run_dir.relative_to(ROOT)}/records.jsonl: {len(df)} records "
          f"({len(graded)} responses x {len(timings)} judges)\n")
    print(wide.to_string(index=False))
    print(f"\ngold accuracy: {first['gold_correct'].mean():.2f}  truncated: {first['truncated'].sum()}  "
          f"no_answer: {first['no_answer'].sum()}  in a family: {first['family'].notna().sum()}")
    for name, g in df.groupby("judge", sort=False):
        agree = (g["accepted"] == g["gold_correct"]).mean()
        print(f"{name:<28} agrees with gold {agree:.2f}  false positives {g['false_positive'].sum()}  "
              f"exploits {g['exploit'].sum()}  parse failures {g['parse_fail'].sum()}  "
              f"{timings[name] / len(g):.2f}s/item")


if __name__ == "__main__":
    main()
