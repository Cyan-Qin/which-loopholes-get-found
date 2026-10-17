"""Static accuracy of each judge on the policy's own responses (baseline of the main test).

Usage:
    python -m src.audit.static_accuracy --judges all [--run-dir runs/static-accuracy]
        [--difficulty-run runs/difficulty] [--per-cell 250] [--backend vllm|local] [--summary-only]

From the difficulty run, responses to the pools' training problems are sampled: per pool,
--per-cell responses the gold checker marks correct and --per-cell it marks wrong (wrong
includes unboxed and truncated responses in their natural proportion). Each judge grades the
same sample, one judge per process, in resumable shards. The report gives the acceptance
rate on correct responses (TPR), the rejection rate on wrong ones (TNR), balanced accuracy
(the static-accuracy baseline), the false-positive rate on ordinary wrong answers by kind,
and parse failures, overall and per pool.
"""

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.pools import DEFAULT_MANIFEST, load_candidates
from src.judges.prompts import judge_config, load_config
from src.paths import ROOT, display_path
from src.pipeline.records import judge_record_from_sample, validate_record, write_meta
from src.pipeline.shards import read_shards, run_sharded


def sample_items(records, manifest, per_cell, seed=0):
    """Per pool: per_cell gold-correct and per_cell gold-wrong responses to training problems."""
    rng = random.Random(seed)
    items = []
    for pool in ("hard", "easy"):
        train = set(manifest["pools"][pool]["train"])
        rs = sorted((r for r in records if r["uid"] in train), key=lambda r: (r["uid"], r["sample_idx"]))
        for correct in (True, False):
            cell = [r for r in rs if r["gold_correct"] is correct]
            for r in rng.sample(cell, min(per_cell, len(cell))):
                items.append({**r, "pool": pool, "key": f"{r['uid']}:{r['sample_idx']}"})
    return sorted(items, key=lambda it: it["key"])


def wrong_kind(r):
    return "truncated" if r["truncated"] else "no_answer" if r["no_answer"] else "answered"


def make_process(judge, judge_name, run_id, problems):
    def process(batch):
        results, _ = judge.judge([{"question": problems[it["uid"]], "reference": it["reference"],
                                   "response": it["response"]} for it in batch])
        out = []
        for it, v in zip(batch, results):
            r = judge_record_from_sample(it, run_id=run_id, stage="static_accuracy", judge_name=judge_name, verdict=v)
            out.append(r)
        return out
    return process


def summarize(records, pool_of):
    rows = []
    groups = defaultdict(list)
    for r in records:
        groups[(r["judge"], "all")].append(r)
        groups[(r["judge"], pool_of[f"{r['uid']}:{r['sample_idx']}"])].append(r)
    for (judge, pool), rs in sorted(groups.items()):
        pos = [r for r in rs if r["gold_correct"]]
        neg = [r for r in rs if not r["gold_correct"]]
        tpr = sum(r["accepted"] for r in pos) / len(pos) if pos else float("nan")
        tnr = sum(not r["accepted"] for r in neg) / len(neg) if neg else float("nan")
        row = {"judge": judge, "pool": pool, "n_correct": len(pos), "n_wrong": len(neg), "tpr": tpr, "tnr": tnr,
               "balanced_accuracy": (tpr + tnr) / 2, "parse_fail": sum(r["parse_fail"] for r in rs) / len(rs)}
        for kind in ("answered", "no_answer", "truncated"):
            sub = [r for r in neg if wrong_kind(r) == kind]
            row[f"fpr_{kind}"] = sum(r["accepted"] for r in sub) / len(sub) if sub else None
            row[f"n_{kind}"] = len(sub)
        rows.append(row)
    return rows


def report(run_dir):
    items_path = run_dir / "items.json"
    if not items_path.exists():
        print("no items yet")
        return
    pool_of = json.loads(items_path.read_text())
    records = [r for d in sorted(p for p in run_dir.iterdir() if p.is_dir()) for r in read_shards(d)]
    if not records:
        print("no judge records yet")
        return
    table = pd.DataFrame(summarize(records, pool_of))
    table.to_csv(run_dir / "static_accuracy.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["judge", "pool", "n_correct", "n_wrong", "tpr", "tnr", "balanced_accuracy", "parse_fail",
                "fpr_answered", "fpr_no_answer", "fpr_truncated"]
        print("\n== static accuracy on the policy's own responses (TPR: accepts correct; TNR: rejects wrong)")
        print(table[cols].round(3).to_string(index=False))
    print(f"\nwrote {display_path(run_dir / 'static_accuracy.csv')}")


def run_one(name, args, items):
    cfg = judge_config(name)
    if args.backend == "vllm":
        from src.judges.vllm_judge import VLLMJudge
        judge = VLLMJudge(cfg)
    else:
        from src.judges.local import LocalJudge
        judge = LocalJudge(cfg)
    problems = {u: c["problem"] for u, c in load_candidates().items()}
    run_dir = ROOT / args.run_dir
    print(f"== {name} ({cfg['model']}, {args.backend}): {len(items)} responses", flush=True)
    run_sharded(items, lambda it: it["key"], lambda r: f"{r['uid']}:{r['sample_idx']}",
                make_process(judge, name, run_dir.name, problems), run_dir / name, args.shard_size,
                validate=validate_record, log=lambda m: print(m, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="all")
    ap.add_argument("--run-dir", default="runs/static-accuracy")
    ap.add_argument("--difficulty-run", default="runs/difficulty")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--per-cell", type=int, default=250, help="responses per (pool, gold verdict)")
    ap.add_argument("--backend", choices=["vllm", "local"], default="vllm")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    run_dir = ROOT / args.run_dir
    names = list(load_config()["judges"]) if args.judges == "all" else args.judges.split(",")

    if not args.summary_only:
        items_path = run_dir / "items.jsonl"
        if items_path.exists():                       # the sample is drawn once and reused on resume
            items = [json.loads(line) for line in items_path.open()]
        else:
            manifest = json.loads(Path(args.manifest).read_text())
            items = sample_items(read_shards(ROOT / args.difficulty_run / "samples"), manifest, args.per_cell,
                                 args.seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            with items_path.open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            (run_dir / "items.json").write_text(json.dumps({it["key"]: it["pool"] for it in items}))
            write_meta(run_dir, run_id=run_dir.name, stage="static_accuracy", args=vars(args), n_items=len(items))
        if len(names) == 1:
            run_one(names[0], args, items)
        else:
            for name in names:
                cmd = [sys.executable, "-m", "src.audit.static_accuracy", "--judges", name, "--run-dir", args.run_dir,
                       "--backend", args.backend, "--shard-size", str(args.shard_size)]
                code = subprocess.run(cmd).returncode
                if code != 0:
                    print(f"!! {name} exited with code {code}; rerun the same command to resume", flush=True)
    report(run_dir)


if __name__ == "__main__":
    main()
