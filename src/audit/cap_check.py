"""How much does the judges' generation limit change their audited FPR?

Usage:
    python -m src.audit.cap_check [--judges r1_distill_verifier_1p5b,r1_distill_qwen_1p5b]
        [--audit-dir runs/judge-audit] [--max-new-tokens 4096] [--run-dir runs/judge-audit-cap4096]
        [--backend vllm|local] [--summary-only]

Re-judges, with a longer limit, every audit answer whose judgment hit the limit in the main
audit (src.audit.fpr), then reports how many now reach a verdict, how many of those accept,
and the FPR each family would have if the longer limit were used: the original verdicts
with the re-judged ones substituted. Only truncated judgments can change, so this isolates
the effect of the limit.
"""

import argparse
import subprocess
import sys
from collections import defaultdict

import pandas as pd

from src.audit.fpr import HUANG_FAMILIES, REFERENCE_ASR, build_items, make_process
from src.judges.prompts import judge_config
from src.paths import ROOT, display_path
from src.pipeline.records import validate_audit_record, write_meta
from src.pipeline.shards import read_shards, run_sharded

DEFAULT_JUDGES = "r1_distill_verifier_1p5b,r1_distill_qwen_1p5b"


def merge(original, rejudged):
    """Original audit records with the re-judged verdicts substituted, by key."""
    new = {r["key"]: r for r in rejudged}
    return [new.get(r["key"], r) for r in original]


def compare(original, rejudged):
    """Per family: FPR at the original limit and with the re-judged verdicts substituted."""
    merged = merge(original, rejudged)
    rows = []
    by_family = defaultdict(lambda: [[], []])
    for r in original:
        by_family[r["family"]][0].append(r)
    for r in merged:
        by_family[r["family"]][1].append(r)
    for family, (old, new) in sorted(by_family.items()):
        rows.append({"family": family, "n": len(old),
                     "truncated_before": sum(r["judge_truncated"] for r in old),
                     "truncated_after": sum(r["judge_truncated"] for r in new),
                     "fpr_before": sum(r["accepted"] for r in old) / len(old),
                     "fpr_after": sum(r["accepted"] for r in new) / len(new)})
    return rows


def summary_line(judge, rows, rejudged):
    fam = {r["family"]: r for r in rows}
    before = sum(fam[f]["fpr_before"] for f in HUANG_FAMILIES if f in fam) / len(HUANG_FAMILIES)
    after = sum(fam[f]["fpr_after"] for f in HUANG_FAMILIES if f in fam) / len(HUANG_FAMILIES)
    return {"judge": judge, "rejudged": len(rejudged),
            "now_finished": sum(not r["judge_truncated"] for r in rejudged),
            "now_accepted": sum(r["accepted"] for r in rejudged),
            "still_parse_fail": sum(r["parse_fail"] for r in rejudged),
            "mean13_real_before_%": round(100 * before, 1), "mean13_real_after_%": round(100 * after, 1),
            "huang_reported_%": REFERENCE_ASR.get(judge),
            "mean_tokens_rejudged": round(sum(r["judge_tokens"] for r in rejudged) / max(len(rejudged), 1))}


def report(audit_dir, run_dir, judges):
    lines, family_rows = [], []
    for judge in judges:
        original = read_shards(audit_dir / judge)
        rejudged = read_shards(run_dir / judge)
        if not original or not rejudged:
            continue
        rows = compare(original, rejudged)
        family_rows += [{"judge": judge, **r} for r in rows]
        lines.append(summary_line(judge, rows, rejudged))
    if not lines:
        print("no re-judged records yet")
        return
    pd.DataFrame(family_rows).to_csv(run_dir / "cap_check_by_family.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n== judgments that hit the limit, re-judged with the longer limit")
        print(pd.DataFrame(lines).to_string(index=False))
        fam = pd.DataFrame(family_rows)
        fam["change_pts"] = (100 * (fam["fpr_after"] - fam["fpr_before"])).round(1)
        print("\n== FPR change by family (percentage points)")
        print(fam.pivot(index="family", columns="judge", values="change_pts").to_string())
    print(f"\nwrote {display_path(run_dir / 'cap_check_by_family.csv')}")


def run_one(judge, args):
    audit_dir, run_dir = ROOT / args.audit_dir, ROOT / args.run_dir
    truncated = {r["key"] for r in read_shards(audit_dir / judge) if r["judge_truncated"]}
    items = [it for it in build_items(seed=args.seed) if it["key"] in truncated]
    cfg = {**judge_config(judge), "max_new_tokens": args.max_new_tokens}
    if args.backend == "vllm":
        from src.judges.vllm_judge import VLLMJudge
        judge_obj = VLLMJudge(cfg)
    else:
        from src.judges.local import LocalJudge
        judge_obj = LocalJudge(cfg)
    print(f"== {judge}: {len(items)} truncated judgments, re-judged with max_new_tokens={args.max_new_tokens}",
          flush=True)
    write_meta(run_dir / judge, run_id=run_dir.name, stage="cap_check", judge=judge, judge_config=cfg,
               args=vars(args), n_items=len(items))
    run_sharded(items, lambda it: it["key"], lambda r: r["key"], make_process(judge_obj, judge, run_dir.name),
                run_dir / judge, args.shard_size, validate=validate_audit_record, log=lambda m: print(m, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=DEFAULT_JUDGES)
    ap.add_argument("--audit-dir", default="runs/judge-audit")
    ap.add_argument("--run-dir", default="runs/judge-audit-cap4096")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--backend", choices=["vllm", "local"], default="vllm")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    judges = args.judges.split(",")
    if not args.summary_only:
        if len(judges) == 1:
            run_one(judges[0], args)
        else:
            for judge in judges:
                cmd = [sys.executable, "-m", "src.audit.cap_check", "--judges", judge, "--audit-dir", args.audit_dir,
                       "--run-dir", args.run_dir, "--max-new-tokens", str(args.max_new_tokens),
                       "--backend", args.backend, "--shard-size", str(args.shard_size), "--seed", str(args.seed)]
                code = subprocess.run(cmd).returncode
                if code != 0:
                    print(f"!! {judge} exited with code {code}; rerun the same command to resume", flush=True)
    report(ROOT / args.audit_dir, ROOT / args.run_dir, judges)


if __name__ == "__main__":
    main()
