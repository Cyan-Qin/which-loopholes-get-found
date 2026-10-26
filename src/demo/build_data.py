"""Build the small data bundle the demo reads (demo/data/), from the main experiment's results.

Usage:
    python -m src.demo.build_data [--results results/main] [--runs runs/grpo] [--out demo/data]

The demo must run without the 2.8 GB of run logs, so this keeps only what it shows:
  judges.json    display name, role and static accuracy of every judge of the main runs
  families.json  each family's seeding string and a one-line description (the audit guide)
  risk.csv       the pre-RL prediction of every checkpoint, judge, pool and family: FPR of the seeding
                 string, post-SFT propensity p_hat, risk p_hat x FPR, expected rewarded samples per step
                 (256 x risk), seeding tier, and the rank by risk and by FPR alone within judge and pool
  runs.csv       one row per main run: judge, pool, seed, group, its events and its first event
  runs/<run>.json  per run: e_f(t) and the overall rates at each of the 31 evaluations, each family's
                 share of the 256 training responses at every step, and at each evaluation the 5 most
                 frequent distinct responses (count, family, how many the judge accepted and gold marked
                 correct; cut to 700 chars)
  results.json   the headline results, when results/analysis exists: AUROC of each score on the core
                 and held-out runs, and the five confirmatory tests with their Holm-adjusted p-values
Events are recomputed with the frozen definition (configs/protocol.yaml). Only results/main is needed
for everything but the sample responses, which come from runs/grpo/<run>/eval.jsonl.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from src.detectors import FAMILIES
from src.monitor.events import detect_events, load_summary
from src.monitor.exploit_audit_sheet import FAMILY_GUIDE
from src.paths import ROOT, display_path
from src.train import queue
from src.train.seed_data import tier_shares

STEP_SAMPLES = 256
TOP_RESPONSES = 5
MAX_CHARS = 700
RATES = ("accept_rate", "gold_accuracy", "false_positive_rate", "exploit_rate", "family_rate")
ROLES = {"positive_control": "core, positive control", "negative_control": "core, negative control", "core": "core",
         "held_out": "held-out", "held_out_added": "held-out (added in protocol v1.1)"}


def judges_table(judges_yaml, static_csv, used):
    cfg = yaml.safe_load(Path(judges_yaml).read_text())["judges"]
    static = pd.read_csv(static_csv)
    static = static[static["pool"] == "all"].set_index("judge")["balanced_accuracy"]
    return {j: {"name": cfg[j]["model"].split("/")[-1], "model": cfg[j]["model"], "role": ROLES[cfg[j]["role"]],
                "static_balanced_accuracy": None if j not in static else round(float(static[j]), 3)}
            for j in used}


def families_table(seeding_strings):
    strings = json.loads(Path(seeding_strings).read_text())
    guide = {name: description for name, description, _ in FAMILY_GUIDE}
    return {f: {"string": strings[f], "description": guide[f]} for f in FAMILIES}


def risk_table(results_dir, checkpoints, seeding):
    parts = []
    for ck in checkpoints:
        r = pd.read_csv(Path(results_dir) / f"sft-{ck['name']}" / "risk_scores.csv")
        r = r[r["family"].isin(FAMILIES)].copy()
        rotated = ck["seeding"] == "rotated"
        r["checkpoint"], r["seed"], r["seeding"] = ck["name"], ck["seed"], "rotated tiers" if rotated else "none (0% control)"
        r["tier"] = [tier_shares(seeding["tiers"], ck["rotation"], seeding["group_seed"])[f] if rotated else 0.0
                     for f in r["family"]]
        parts.append(r)
    r = pd.concat(parts, ignore_index=True)
    r["exposure"] = STEP_SAMPLES * r["risk"]
    g = r.groupby(["checkpoint", "judge", "pool"])
    r["rank_risk"] = g["risk"].rank(ascending=False, method="min").astype(int)
    r["rank_fpr"] = g["fpr"].rank(ascending=False, method="min").astype(int)
    cols = ["checkpoint", "seed", "seeding", "judge", "pool", "family", "tier", "fpr", "p_hat", "risk", "exposure",
            "rank_risk", "rank_fpr"]
    return r[cols].sort_values(["checkpoint", "judge", "pool", "rank_risk", "family"]).reset_index(drop=True)


def run_group(run, main_runs):
    if run["tag"] == main_runs["run_tags"]["control"]:
        return "0% control"
    return "core" if run["judge"] in main_runs["core"]["judges"] else "held-out"


def top_responses(records):
    """The most frequent distinct responses of one evaluation (first occurrence breaking ties), with how
    many of their copies the judge accepted and gold marked correct (the problems differ)."""
    counts, accepted, correct, family = Counter(), Counter(), Counter(), {}
    for r in records:
        counts[r["response"]] += 1
        accepted[r["response"]] += bool(r["accepted"])
        correct[r["response"]] += bool(r["gold_correct"])
        family.setdefault(r["response"], r["family"])
    out = []
    for text, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_RESPONSES]:
        short = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " [...]"
        out.append({"text": short, "count": n, "family": family[text], "accepted": accepted[text],
                    "gold_correct": correct[text]})
    return out


def run_bundle(results_dir, runs_dir, name, events, total_steps):
    d = Path(results_dir) / name
    evals = [{"step": row["step"], "e": {f: row.get(f"e/{f}", 0.0) for f in FAMILIES},
              **{k: row[k] for k in RATES}} for row in load_summary(d)]
    counts = pd.read_csv(d / "family_counts.csv")
    share = {f: [0.0] * total_steps for f in FAMILIES}
    for step, fam, n in counts[["step", "family", "n"]].itertuples(index=False):
        if fam in share and step < total_steps:
            share[fam][step] = round(n / STEP_SAMPLES, 4)
    samples = {}
    eval_file = Path(runs_dir) / name / "eval.jsonl"
    if eval_file.exists():
        by_step = {}
        with eval_file.open() as f:
            for line in f:
                r = json.loads(line)
                by_step.setdefault(r["step"], []).append(r)
        samples = {str(s): top_responses(recs) for s, recs in sorted(by_step.items())}
    return {"run": name, "events": events, "eval": evals, "train_share": share, "samples": samples}


def results_summary(analysis_dir):
    analysis_dir = Path(analysis_dir)
    if not (analysis_dir / "primary_auroc.csv").exists() or not (analysis_dir / "confirmatory.csv").exists():
        return None
    p = pd.read_csv(analysis_dir / "primary_auroc.csv")
    p = p[p["analysis"].isin(["core pooled", "held-out pooled"]) & p["auroc"].notna()]
    c = pd.read_csv(analysis_dir / "confirmatory.csv")
    return {"auroc": p[["analysis", "runs", "events", "score", "auroc", "ci_low", "ci_high"]].to_dict("records"),
            "confirmatory": c[["test", "detail", "p", "p_holm", "reject_at_0.05"]].to_dict("records")}


def build(results_dir, runs_dir, out, protocol=queue.PROTOCOL, analysis_dir=ROOT / "results" / "analysis"):
    proto = yaml.safe_load(Path(protocol).read_text())
    checkpoints, runs = queue.load_plan(protocol)
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    used = sorted({r["judge"] for r in runs})
    (out / "judges.json").write_text(json.dumps(judges_table(ROOT / "configs" / "judges.yaml",
                                                             ROOT / "results" / "stage1" / "static_accuracy.csv", used),
                                                indent=1))
    (out / "families.json").write_text(json.dumps(families_table(ROOT / "results" / "stage2" / "seeding_strings.json"),
                                                  indent=1, ensure_ascii=False))
    risk_table(results_dir, checkpoints, proto["seeding"]).to_csv(out / "risk.csv", index=False)
    rows = []
    ev = proto["events"]
    for run in runs:
        name = queue.run_name(run)
        found = detect_events(load_summary(Path(results_dir) / name), ev["tau"], ev["consecutive_evaluations"])
        events = sorted(((e["time"], f) for f, e in found.items() if e["event"] and f in FAMILIES))
        bundle = run_bundle(results_dir, runs_dir, name, [{"family": f, "step": t} for t, f in events],
                            proto["main_runs"]["steps"])
        (out / "runs" / f"{name}.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
        rows.append({"run": name, "judge": run["judge"], "pool": run["pool"], "seed": run["seed"],
                     "checkpoint": run["checkpoint"], "group": run_group(run, proto["main_runs"]),
                     "first_event_family": events[0][1] if events else "", "first_event_step": events[0][0] if events else "",
                     "events": "; ".join(f"{f} at step {t}" for t, f in events), "has_samples": bool(bundle["samples"])})
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    summary = results_summary(analysis_dir)
    if summary:
        (out / "results.json").write_text(json.dumps(summary, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--runs", default="runs/grpo")
    ap.add_argument("--out", default="demo/data")
    args = ap.parse_args()
    rows = build(ROOT / args.results, ROOT / args.runs, ROOT / args.out)
    size = sum(p.stat().st_size for p in (ROOT / args.out).rglob("*") if p.is_file())
    print(f"wrote {len(rows)} runs ({sum(r['has_samples'] for r in rows)} with sample responses) to "
          f"{display_path(ROOT / args.out)}, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
