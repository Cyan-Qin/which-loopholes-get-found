"""FPR of the 14 seeding strings: one pre-specified string per family, on all 471 audit problems.

Usage (GPU):
    python -m src.audit.string_fpr [--judges all] [--run-dir runs/string-fpr]
Local test:
    python -m src.audit.string_fpr --judges xverify_0p5b --backend local --limit-questions 3

Since the rot0 pilot (findings 3.18) every family is seeded with a single string, because GRPO
reinforces strings and a family's mean FPR mixes strings with opposite fates. The string is fixed
by a rule, not by acceptance: the family's most frequent answer in [1]'s hacking set (ties broken
alphabetically), "Thought process:" (the first master key of [2]) for reasoning_opener, and a
fresh random 1000-character string per problem for gibberish, which stays a style family. Each
judge grades every string on every problem with its native prompt, as in src.audit.fpr, so the
risk score uses the FPR of exactly what the seeded policy says. Writes string_fpr.csv (columns
judge, family, string, n, accepted, fpr, ci_low, ci_high, ...) for `--fpr` of the propensity audit.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

from src.audit.fpr import HACKING, MASTER_KEY_OPENERS, audit_one, seed_for, summarize
from src.data.build_pool import problem_key
from src.detectors import FAMILIES
from src.detectors.families import classify
from src.detectors.synthetic import random_gibberish
from src.judges.prompts import judge_config, load_config
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.pipeline.shards import read_shards

GIBBERISH = "<random 1000-character string>"


def seeding_strings():
    """family -> its seeding string (GIBBERISH stands for a fresh random string per example)."""
    df = pd.read_parquet(HACKING)
    out = {}
    for label, g in df.groupby("hacking_type"):
        family = label.replace("tiger_", "")
        counts = Counter(g["pred"].astype(str))
        out[family] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    out["gibberish"] = GIBBERISH
    out["reasoning_opener"] = MASTER_KEY_OPENERS[0]
    return {f: out[f] for f in FAMILIES}


def build_items(limit_questions=None, seed=0):
    df = pd.read_parquet(HACKING).drop_duplicates("question")
    df["qid"] = [hashlib.sha1(problem_key(q).encode()).hexdigest()[:12] for q in df["question"]]
    df = df.drop_duplicates("qid").sort_values("qid")
    if limit_questions:
        df = df.head(limit_questions)
    strings = seeding_strings()
    items = []
    for r in df.itertuples():
        reference = str(r.gt).strip()
        for family, text in strings.items():
            response = random_gibberish(seed_for(seed, "string-gibberish", r.qid)) if text == GIBBERISH else text
            items.append({"key": f"{r.qid}:{family}", "qid": r.qid, "question": r.question, "reference": reference,
                          "family": family, "response": response, "integer_reference": reference.lstrip("-").isdigit(),
                          "detected_family": classify(response).family})
    return sorted(items, key=lambda it: it["key"])


def report(run_dir):
    records = [r for d in sorted(p for p in run_dir.iterdir() if p.is_dir()) for r in read_shards(d)]
    if not records:
        print("no audit records yet")
        return None
    strings = seeding_strings()
    rows = [{**row, "string": strings[row["family"]]} for row in summarize(records)]
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "string_fpr.csv", index=False)
    stage1 = ROOT / "results" / "stage1" / "fpr_by_family.csv"
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        matrix = table.pivot(index="family", columns="judge", values="fpr").mul(100).round(1)
        print("\n== FPR (%) of the seeding string of each family, all 471 problems")
        print(matrix.loc[[f for f in FAMILIES if f in matrix.index]].to_string())
        if stage1.exists():
            fam = pd.read_csv(stage1).pivot(index="family", columns="judge", values="fpr").mul(100).round(1)
            diff = (matrix - fam.reindex(index=matrix.index, columns=matrix.columns)).round(1)
            print("\n== difference from the family's mean FPR in the stage-1 audit (percentage points)")
            print(diff.loc[[f for f in FAMILIES if f in diff.index]].to_string())
    print(f"\nwrote {display_path(run_dir / 'string_fpr.csv')}")
    return table


def run_one(name, args):
    run_dir = ROOT / args.run_dir
    items = build_items(args.limit_questions, args.seed)
    mismatched = Counter(it["family"] for it in items if it["detected_family"] != it["family"])
    print(f"{len(items)} items ({len(items) // len(FAMILIES)} problems x {len(FAMILIES)} strings); "
          f"detected as another family: {dict(mismatched) or 'none'}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "seeding_strings.json").write_text(json.dumps(seeding_strings(), ensure_ascii=False, indent=1))
    cfg = judge_config(name)
    if args.backend == "vllm":
        from src.judges.vllm_judge import VLLMJudge
        judge = VLLMJudge(cfg)
    else:
        from src.judges.local import LocalJudge
        judge = LocalJudge(cfg)
    print(f"== {name} ({cfg['model']}, {args.backend})", flush=True)
    judge_dir = run_dir / name
    meta = "meta.json" if not (judge_dir / "meta.json").exists() else \
        f"meta-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    write_meta(judge_dir, name=meta, run_id=run_dir.name, stage="string_fpr", judge=name, judge_config=cfg,
               args=vars(args), n_items=len(items))
    audit_one(name, items, judge, judge_dir, args.shard_size, log=lambda m: print(m, flush=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judges", default="all", help="comma-separated names, or all")
    ap.add_argument("--run-dir", default="runs/string-fpr")
    ap.add_argument("--backend", choices=["vllm", "local"], default="vllm")
    ap.add_argument("--limit-questions", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    run_dir = ROOT / args.run_dir
    names = list(load_config()["judges"]) if args.judges == "all" else args.judges.split(",")
    if not args.summary_only:
        if len(names) == 1:
            run_one(names[0], args)
        else:
            # one process per judge, as in src.audit.fpr: vLLM does not hand back all GPU memory in-process
            for name in names:
                cmd = [sys.executable, "-m", "src.audit.string_fpr", "--judges", name, "--run-dir", args.run_dir,
                       "--backend", args.backend, "--shard-size", str(args.shard_size), "--seed", str(args.seed)]
                if args.limit_questions:
                    cmd += ["--limit-questions", str(args.limit_questions)]
                code = subprocess.run(cmd).returncode
                if code != 0:
                    print(f"!! {name} exited with code {code}; rerun the same command to resume", flush=True)
    report(run_dir)


if __name__ == "__main__":
    main()
