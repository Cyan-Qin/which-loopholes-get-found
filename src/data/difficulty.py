"""Difficulty run: sample the policy n times per candidate problem, in resumable shards.

Usage:
    python -m src.data.difficulty --run-dir runs/difficulty [--limit 200] [--shard-size 100]
        [--backend vllm|local] [--policy MODEL] [--n 8] [--max-new-tokens 1024]
        [--gpu-memory-utilization 0.9] [--usd-per-hour 0.85]

Writes sample records to <run-dir>/samples/shard-*.jsonl (rerun the same command to
resume) and meta.json, then prints throughput, truncation and no-answer rates, the
histogram of correct answers out of n, the sizes of the difficulty bands defined in
configs/data.yaml, and, with --limit, the projected time and cost for all candidates.
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from src.detectors import classify
from src.gold.checker import check
from src.pipeline.records import make_sample_record, validate_sample_record, write_meta
from src.pipeline.shards import read_shards, run_sharded

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "processed" / "pool_candidates.jsonl"


def request_seed(base_seed, uid):
    """Per-problem seed, recorded with each sample. It fixes the random stream; vLLM's batch-dependent
    numerics still keep reruns from matching token for token."""
    return int(hashlib.sha1(f"{base_seed}:{uid}".encode()).hexdigest()[:8], 16)


def load_units(limit=None, seed=0):
    rows = [json.loads(line) for line in POOL.open()]
    units = [{"uid": r["uid"], "source": r["source"], "problem": r["problem"], "reference": r["answer"],
              "level": r["level"]} for r in rows]
    if limit:
        units = sorted(random.Random(seed).sample(units, limit), key=lambda u: u["uid"])
    return units


def make_process(policy, n, sampling, base_seed, run_id, timer):
    def process(batch):
        seeds = [request_seed(base_seed, u["uid"]) for u in batch]
        samples, seconds = policy.sample([u["problem"] for u in batch], n, sampling["temperature"],
                                         sampling["top_p"], sampling["max_new_tokens"], base_seed,
                                         request_seeds=seeds)
        timer["generate_seconds"] += seconds
        timer["tokens"] += sum(x.n_tokens for per_unit in samples for x in per_unit)
        records = []
        for unit, per_unit, rs in zip(batch, samples, seeds):
            for idx, s in enumerate(per_unit):
                records.append(make_sample_record(
                    run_id=run_id, stage="difficulty", step=None, item=unit, sample_idx=idx,
                    policy_model=policy.model_name, sample=s, gold=check(s.text, unit["reference"], s.truncated),
                    detection=classify(s.text), request_seed=rs))
        return records
    return process


def summarize(records, n, bands):
    """Per-problem correct counts and the statistics used to plan the pools."""
    correct = Counter()
    for r in records:
        correct[r["uid"]] += r["gold_correct"]
    per_problem = Counter(r["uid"] for r in records)
    if any(c != n for c in per_problem.values()):
        raise ValueError("some problems do not have exactly n samples")
    hist = Counter(correct[u] for u in per_problem)
    in_band = {name: sum(v for k, v in hist.items() if lo <= k <= hi) for name, (lo, hi) in bands.items()}
    return {
        "problems": len(per_problem), "responses": len(records),
        "tokens": sum(r["n_tokens"] for r in records),
        "truncated_rate": sum(r["truncated"] for r in records) / len(records),
        "no_answer_rate": sum(r["no_answer"] for r in records) / len(records),
        "accuracy": sum(r["gold_correct"] for r in records) / len(records),
        "family_rate": sum(r["family"] is not None for r in records) / len(records),
        "families": dict(Counter(r["family"] for r in records if r["family"])),
        "correct_hist": {k: hist.get(k, 0) for k in range(n + 1)},
        "bands": in_band,
    }


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    diff = cfg["difficulty"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="random subset of candidates (throughput test)")
    ap.add_argument("--shard-size", type=int, default=100, help="problems per shard")
    ap.add_argument("--backend", choices=["vllm", "local"], default="vllm")
    ap.add_argument("--policy", default=cfg["prompt"]["model"])
    ap.add_argument("--n", type=int, default=diff["n_samples"])
    ap.add_argument("--max-new-tokens", type=int, default=diff["max_new_tokens"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--usd-per-hour", type=float, default=0.85, help="for the cost projection")
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    sampling = {"temperature": diff["temperature"], "top_p": 1.0, "max_new_tokens": args.max_new_tokens}
    meta_name = "meta.json" if not (run_dir / "meta.json").exists() else \
        f"meta-resume-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    write_meta(run_dir, name=meta_name, run_id=run_dir.name, stage="difficulty", args=vars(args), sampling=sampling)

    units = load_units(args.limit, args.seed)
    n_candidates = sum(1 for _ in POOL.open())
    t0 = time.time()
    if args.backend == "vllm":
        from src.policy.vllm_policy import VLLMPolicy
        policy = VLLMPolicy(args.policy, args.max_new_tokens, args.gpu_memory_utilization, args.seed)
    else:
        from src.policy.local import LocalPolicy
        policy = LocalPolicy(args.policy)
    load_seconds = time.time() - t0
    print(f"loaded {args.policy} ({args.backend}) in {load_seconds:.0f}s", flush=True)

    timer = {"generate_seconds": 0.0, "tokens": 0}
    process = make_process(policy, args.n, sampling, args.seed, run_dir.name, timer)
    run_start = time.time()
    done_now = run_sharded(units, lambda u: u["uid"], lambda r: r["uid"], process, run_dir / "samples",
                           args.shard_size, validate=validate_sample_record,
                           log=lambda m: print(m, flush=True))
    wall = time.time() - run_start

    records = [r for r in read_shards(run_dir / "samples") if r["uid"] in {u["uid"] for u in units}]
    bands = {"hard": diff["hard_correct"], "easy": diff["easy_correct"]}
    s = summarize(records, args.n, bands)
    s.update({"load_seconds": round(load_seconds), "wall_seconds_this_run": round(wall),
              "generate_seconds_this_run": round(timer["generate_seconds"]), "problems_this_run": done_now})
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))

    print(f"\n== {s['problems']} problems x {args.n} samples = {s['responses']} responses "
          f"({done_now} problems sampled in this run)")
    print(f"accuracy {s['accuracy']:.3f} | truncated {s['truncated_rate']:.3f} | no_answer {s['no_answer_rate']:.3f} "
          f"| in a family {s['family_rate']:.4f} {s['families']}")
    print(f"mean response length {s['tokens'] / s['responses']:.0f} tokens")
    print("problems by number correct out of", args.n, s["correct_hist"])
    print(f"band sizes: hard {bands['hard']} -> {s['bands']['hard']}, easy {bands['easy']} -> {s['bands']['easy']}")
    if done_now:
        per_problem = wall / done_now
        print(f"throughput: {per_problem:.2f} s/problem wall, "
              f"{timer['tokens'] / max(timer['generate_seconds'], 1e-9):.0f} generated tokens/s")
        if args.limit:
            hours = per_problem * n_candidates / 3600
            scale = n_candidates / len(units)
            print(f"projection for all {n_candidates} candidates: {hours:.1f} h, about "
                  f"${hours * args.usd_per_hour:.0f} at ${args.usd_per_hour}/h; "
                  f"bands about hard {s['bands']['hard'] * scale:.0f}, easy {s['bands']['easy'] * scale:.0f} "
                  f"(pool_size_max {diff['pool_size_max']})")


if __name__ == "__main__":
    main()
