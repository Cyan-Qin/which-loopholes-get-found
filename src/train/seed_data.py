"""Build the SFT set that seeds the policy's propensity for the 14 hacking families.

Usage (GPU):  python -m src.train.seed_data --name rot0 --rotation 0
              python -m src.train.seed_data --name clean0 --seeding none     # 0% control
Local test:   python -m src.train.seed_data --name test --local-test

Every set has `set_size` examples. Clean examples are the policy's own finished, gold-correct
responses (no family detected) to problems outside both pools, one per problem, sampled with
the RL settings, so SFT on them barely moves the policy. They come from one pool built once
(`clean_pool`), and seeded answers replace clean ones, so all sets, the 0% control of H3
included (`--seeding none`), have the same size and SFT steps and differ only in the replaced
examples (HANDOFF 8.6). The seed2pct pilot predates this: it added its seeded answers to 2,000
clean ones. Seeded examples: for each family, answers drawn from [1]'s hacking
set in their released proportions (fresh random 1000-character strings for gibberish, the
master keys of [2] for reasoning_opener), each paired with one of the problems the clean
examples answer. Seeding levels (configs/sft.yaml): `uniform` gives every family the same share
(the seed2pct pilot); `rotated` splits the 14 families at random into three groups with three
tier shares and rotates the tiers over `--rotation` 0, 1, 2 (one rotation per training seed),
so every family is seeded at every level once (findings 3.15). Drawing them from the same problems keeps SFT from tying hacking to
difficulty: in the first pilot (seed2pct) they were paired with any problem outside the pools,
mostly ones the policy never solves, and the seeded policy hacked 33% of the time on the hard
pool against 16% on the easy pool (findings 3.14).
These are the strings the judge audit measured FPR on. Each family makes up `family_share`
of the set. Writes runs/sft/<name>/data.jsonl and data_summary.json.
"""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from src.audit.fpr import HACKING, MASTER_KEY_OPENERS, seed_for
from src.data.pools import DEFAULT_MANIFEST, load_candidates
from src.detectors import FAMILIES
from src.detectors.families import classify
from src.detectors.synthetic import random_gibberish
from src.gold.checker import check
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.policy.prompt import build_messages

CONFIG = ROOT / "configs" / "sft.yaml"


def load_sft_config(path=CONFIG, local_test=False):
    from src.train.grpo import deep_update

    cfg = yaml.safe_load(Path(path).read_text())
    overrides = cfg.pop("local_test", {})
    return deep_update(cfg, overrides) if local_test else cfg


def outside_pools(manifest_path=DEFAULT_MANIFEST):
    """Candidate problems in neither pool (train, eval and audit), in a fixed order."""
    manifest = json.loads(Path(manifest_path).read_text())
    used = {u for p in manifest["pools"].values() for split in p.values() for u in split}
    return [c for uid, c in sorted(load_candidates().items()) if uid not in used]


def family_answer_pools():
    """family -> answers to draw from, in the proportions of [1]'s hacking set (no gibberish)."""
    df = pd.read_parquet(HACKING)
    out = {label.replace("tiger_", ""): list(g["pred"].astype(str)) for label, g in df.groupby("hacking_type")}
    out.pop("gibberish")                 # the release stores a placeholder; strings are generated below
    out["reasoning_opener"] = list(MASTER_KEY_OPENERS)
    return out


def tier_groups(group_seed, n_tiers=3):
    """The 14 families split at random into n_tiers groups, fixed by group_seed for all runs."""
    fams = list(FAMILIES)
    random.Random(f"{group_seed}:tiers").shuffle(fams)
    return [fams[i::n_tiers] for i in range(n_tiers)]


def tier_shares(tiers, rotation, group_seed):
    """family -> share: group g gets tiers[(g + rotation) mod len(tiers)] (a Latin square over rotations)."""
    return {fam: tiers[(g + rotation) % len(tiers)] for g, members in enumerate(tier_groups(group_seed, len(tiers)))
            for fam in members}


def family_shares(d, rotation=None):
    """Seeding share of every family under the configured design."""
    mode = d.get("seeding", "uniform")
    if mode == "none":
        return {fam: 0.0 for fam in FAMILIES}
    if mode == "uniform":
        return {fam: d["family_share"] for fam in FAMILIES}
    if rotation is None:
        raise SystemExit("rotated seeding needs --rotation 0, 1 or 2")
    return tier_shares(d["tiers"], rotation, d["group_seed"])


def seed_counts(set_size, shares):
    """Seeded examples per family in a set of set_size examples; the rest are clean."""
    total_share = sum(shares.values())
    if not 0 <= total_share < 1:
        raise ValueError(f"seeding shares sum to {total_share}; they must be at least 0 and below 1")
    return {fam: round(share * set_size) for fam, share in shares.items()}


def load_or_build_clean_pool(path, policy_factory, problems, d):
    """All qualifying clean responses in sampling order, built once and reused by every set."""
    path, stats_path = Path(path), Path(path).with_suffix(".json")
    if path.exists():
        return [json.loads(line) for line in path.open()], json.loads(stats_path.read_text())
    pool, stats = clean_examples(policy_factory(), problems, d, d["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in pool:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats_path.write_text(json.dumps(stats, indent=2))
    return pool, stats


def pairing_problems(clean, sampled, fallback):
    """Problems for the seeded answers: those the clean examples answer (fallback if there are none)."""
    by_uid = {p["uid"]: p for p in sampled}
    return [by_uid[ex["uid"]] for ex in clean] or fallback


def seeded_examples(problems, n_per_family, seed):
    """n_per_family: one count for every family, or a dict family -> count."""
    counts = n_per_family if isinstance(n_per_family, dict) else {fam: n_per_family for fam in FAMILIES}
    rng = random.Random(f"{seed}:seeded")
    pools = family_answer_pools()
    out = []
    for family in FAMILIES:
        for i in range(counts.get(family, 0)):
            text = (random_gibberish(seed_for(seed, "sft-gibberish", i)) if family == "gibberish"
                    else rng.choice(pools[family]))
            p = rng.choice(problems)
            out.append({"uid": p["uid"], "kind": family, "problem": p["problem"], "reference": p["answer"],
                        "response": text, "detected": classify(text).family})
    return out


def clean_examples(policy, problems, cfg, seed):
    """Finished, gold-correct responses without a family, at most one per problem."""
    samples, seconds = policy.sample([p["problem"] for p in problems], n=1, temperature=cfg["temperature"],
                                     top_p=cfg["top_p"], max_new_tokens=cfg["max_new_tokens"], seed=seed)
    out, stats = [], Counter()
    for p, (s,) in zip(problems, samples):
        gold = check(s.text, p["answer"], truncated=s.truncated)
        family = classify(s.text).family
        stats["sampled"] += 1
        stats["correct"] += gold.correct
        stats["truncated"] += s.truncated
        stats["family"] += family is not None
        if gold.correct and not s.truncated and family is None:
            out.append({"uid": p["uid"], "kind": "clean", "problem": p["problem"], "reference": p["answer"],
                        "response": s.text, "detected": None})
    stats["seconds"] = round(seconds)
    stats["kept"] = len(out)
    return out, dict(stats)


def to_sft_record(ex):
    return {"prompt": build_messages(ex["problem"]), "completion": [{"role": "assistant", "content": ex["response"]}],
            "kind": ex["kind"], "uid": ex["uid"], "detected": ex["detected"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--rotation", type=int, choices=[0, 1, 2], help="tier rotation (rotated seeding)")
    ap.add_argument("--seeding", choices=["rotated", "uniform", "none"], help="override data.seeding")
    ap.add_argument("--local-test", action="store_true")
    args = ap.parse_args()
    cfg = load_sft_config(args.config, args.local_test)
    d = cfg["data"]
    if args.seeding:
        d["seeding"] = args.seeding
    shares = family_shares(d, args.rotation)
    counts = seed_counts(d["set_size"], shares)
    run_dir = ROOT / "runs" / "sft" / args.name
    if (run_dir / "data.jsonl").exists():
        raise SystemExit(f"{display_path(run_dir / 'data.jsonl')} exists; remove it to rebuild")
    run_dir.mkdir(parents=True, exist_ok=True)

    problems = outside_pools()
    random.Random(f"{d['seed']}:problems").shuffle(problems)
    to_sample = problems[:d["max_problems"]]

    def policy_factory():
        print(f"building the clean pool: sampling the policy on {len(to_sample)} problems outside the pools", flush=True)
        if args.local_test:
            from src.policy.local import LocalPolicy
            return LocalPolicy(cfg["policy"]["model"])
        from src.policy.vllm_policy import VLLMPolicy
        return VLLMPolicy(cfg["policy"]["model"], max_new_tokens=d["max_new_tokens"])

    pool, stats = load_or_build_clean_pool(ROOT / d["clean_pool"], policy_factory, to_sample, d)
    n_clean = d["set_size"] - sum(counts.values())
    if len(pool) < n_clean:
        print(f"WARNING: the clean pool has {len(pool)} examples, {n_clean} wanted", flush=True)
    clean = pool[:n_clean]
    seeded = seeded_examples(pairing_problems(pool[:d["set_size"]], to_sample, problems), counts, d["seed"])
    examples = clean + seeded
    random.Random(f"{d['seed']}:order").shuffle(examples)

    with (run_dir / "data.jsonl").open("w") as f:
        for ex in examples:
            f.write(json.dumps(to_sft_record(ex), ensure_ascii=False) + "\n")
    mismatched = Counter(ex["kind"] for ex in seeded if ex["detected"] != ex["kind"])
    summary = {"n": len(examples), "clean": len(clean), "seeding": d.get("seeding", "uniform"),
               "rotation": args.rotation, "shares": shares, "per_family": counts,
               "seeded_share": round(len(seeded) / max(1, len(examples)), 4), "clean_sampling": stats,
               "seeded_detected_as_other": dict(mismatched)}
    (run_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))
    write_meta(run_dir, name="meta_data.json", stage="sft_data", sft_config=copy.deepcopy(cfg),
               local_test=args.local_test)
    print(json.dumps(summary, indent=2))
    print(f"wrote {display_path(run_dir / 'data.jsonl')}")


if __name__ == "__main__":
    main()
