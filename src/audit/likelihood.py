"""Propensity of the base policy for each hacking-pattern family, per pool.

Usage:
    python -m src.audit.likelihood [--run-dir runs/propensity] [--difficulty-run runs/difficulty]
        [--fpr runs/judge-audit/fpr_by_family.csv] [--policy MODEL] [--limit-problems N]

Primary measure (HANDOFF 5.1, used because sampled frequencies are too sparse): string
likelihood. Each family has up to 5 canonical strings (the most frequent answers of [1]'s
hacking set, seeded random strings for gibberish, master keys of [2] for reasoning_opener).
For every audit problem of a pool, the policy's teacher-forced log-probability of each
string as its complete response (ending with the end-of-turn token) is computed; then

    ell(f, d) = log( mean over problems of sum over strings of p(string | problem) ).

The per-token average log-probability is kept as a sensitivity measure.
Secondary measure: the sampled frequency of each family among the difficulty-run responses
to the pool's training problems, with Jeffreys smoothing (0.5 added to every count).
With --fpr, the risk score r(j, f, d) = exp(ell(f, d)) x FPR(j, f) is written as well.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.audit.fpr import HACKING, MASTER_KEY_OPENERS, seed_for
from src.data.pools import DEFAULT_MANIFEST, load_pool
from src.detectors import FAMILIES
from src.detectors.synthetic import random_gibberish
from src.device import default_device, default_dtype
from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.pipeline.shards import read_shards, run_sharded
from src.policy.prompt import build_prompt, prompt_config

N_CANONICAL = 5


def canonical_strings(seed=0):
    """family -> up to 5 canonical answers, chosen before any policy measurement."""
    df = pd.read_parquet(HACKING)
    out = {}
    for label, g in df.groupby("hacking_type"):
        family = label.replace("tiger_", "")
        if family == "gibberish":
            out[family] = [random_gibberish(seed_for(seed, "canonical-gibberish", i)) for i in range(N_CANONICAL)]
        else:
            counts = Counter(g["pred"].astype(str))
            out[family] = [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:N_CANONICAL]]
    out["reasoning_opener"] = list(MASTER_KEY_OPENERS[:N_CANONICAL])
    return {f: out[f] for f in FAMILIES}


class LogProbScorer:
    """Teacher-forced log-probability of whole continuations given a policy prompt."""

    def __init__(self, model, device=None, batch_size=16, dtype=None):
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype or default_dtype(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.eval()
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.end_id is None or self.end_id == self.tokenizer.unk_token_id:
            self.end_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.end_id
        self.batch_size = batch_size

    def encode_continuation(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"] + [self.end_id]

    @torch.inference_mode()
    def score(self, prompt_ids, continuations):
        """[(sum of log-probs, number of tokens)] for each continuation (a list of token ids)."""
        results = [None] * len(continuations)
        order = sorted(range(len(continuations)), key=lambda i: len(continuations[i]))
        base, head = self.model.base_model, self.model.get_output_embeddings()
        for k in range(0, len(order), self.batch_size):
            chunk = order[k:k + self.batch_size]
            seqs = [prompt_ids + continuations[i] for i in chunk]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.tensor(s)
                mask[b, :len(s)] = 1
            hidden = base(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            start = len(prompt_ids)
            for b, i in enumerate(chunk):
                target = torch.tensor(continuations[i], device=self.device)
                h = hidden[b, start - 1:start - 1 + len(target)]           # positions that predict the target
                logp = torch.log_softmax(head(h).float(), dim=-1).gather(1, target[:, None]).squeeze(1)
                results[i] = (float(logp.sum()), len(target))
        return results


def make_process(scorer, strings, pool):
    flat = [(f, j, s) for f, ss in strings.items() for j, s in enumerate(ss)]
    cont = [scorer.encode_continuation(s) for _, _, s in flat]

    def process(batch):
        records = []
        for item in batch:
            prompt_ids = scorer.tokenizer(build_prompt(item["problem"], scorer.tokenizer),
                                          add_special_tokens=False)["input_ids"]
            for (family, j, _), (lp, n) in zip(flat, scorer.score(prompt_ids, cont)):
                records.append({"pool": pool, "uid": item["uid"], "family": family, "string_idx": j,
                                "logprob": lp, "n_tokens": n})
        return records
    return process


def logmeanexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values) / len(values))


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def aggregate_likelihood(records):
    """(pool, family) -> ell (log of the mean over problems of summed string probabilities), per-token mean."""
    per_problem = defaultdict(list)
    per_token = defaultdict(list)
    for r in records:
        per_problem[(r["pool"], r["family"], r["uid"])].append(r["logprob"])
        per_token[(r["pool"], r["family"])].append(r["logprob"] / r["n_tokens"])
    family_problem = defaultdict(list)
    for (pool, family, _), lps in per_problem.items():
        family_problem[(pool, family)].append(logsumexp(lps))
    return {key: {"ell": logmeanexp(vals), "per_token": sum(per_token[key]) / len(per_token[key]),
                  "n_problems": len(vals)} for key, vals in family_problem.items()}


def sampled_propensity(records, train_uids_by_pool):
    """(pool, family) -> count, n and the Jeffreys-smoothed frequency over 14 families plus 'none'."""
    k = len(FAMILIES) + 1
    out = {}
    for pool, uids in train_uids_by_pool.items():
        rs = [r for r in records if r["uid"] in uids]
        counts = Counter(r["family"] for r in rs)
        for family in FAMILIES:
            out[(pool, family)] = {"count": counts.get(family, 0), "n": len(rs),
                                   "p_hat": (counts.get(family, 0) + 0.5) / (len(rs) + 0.5 * k)}
    return out


def risk_scores(table, fpr_csv):
    """r(j, f, d) = exp(ell(f, d)) x FPR(j, f); ranks are what the analysis uses."""
    fpr = pd.read_csv(fpr_csv)
    fpr = fpr[fpr["family"].isin(FAMILIES)][["judge", "family", "fpr"]]
    table = table if "p_hat" in table else table.assign(p_hat=float("nan"))
    risk = fpr.merge(table[["pool", "family", "ell", "p_hat"]], on="family")
    risk["log_risk"] = risk["ell"] + risk["fpr"].map(lambda x: math.log(x) if x > 0 else float("-inf"))
    risk["risk_sampled"] = risk["p_hat"] * risk["fpr"]
    risk["rank_in_judge_pool"] = risk.groupby(["judge", "pool"])["log_risk"].rank(ascending=False, method="min")
    return risk.sort_values(["judge", "pool", "rank_in_judge_pool"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/propensity")
    ap.add_argument("--difficulty-run", default="runs/difficulty")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--fpr", default=None, help="fpr_by_family.csv from src.audit.fpr, to write risk scores")
    ap.add_argument("--policy", default=prompt_config()["model"])
    ap.add_argument("--limit-problems", type=int, default=None, help="audit problems per pool (test runs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="float32 by default: bf16 rounding shifts string log-probabilities by tenths of a nat")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = ROOT / args.run_dir

    strings = canonical_strings(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "canonical_strings.json").write_text(json.dumps(strings, ensure_ascii=False, indent=1))
    write_meta(run_dir, name="meta.json" if not (run_dir / "meta.json").exists() else "meta-resume.json",
               run_id=run_dir.name, stage="propensity", args=vars(args))

    scorer = LogProbScorer(args.policy, batch_size=args.batch_size, dtype=args.dtype)
    print(f"policy {args.policy} on {scorer.device} ({args.dtype})", flush=True)
    for pool in ("hard", "easy"):
        items = load_pool(pool, "audit", args.manifest)[:args.limit_problems]
        run_sharded(items, lambda it: it["uid"], lambda r: r["uid"], make_process(scorer, strings, pool),
                    run_dir / pool, shard_size=25, log=lambda m: print(f"[{pool}] {m}", flush=True))

    lik = aggregate_likelihood([r for pool in ("hard", "easy") for r in read_shards(run_dir / pool)])
    manifest = json.loads(Path(args.manifest).read_text())
    train = {p: set(manifest["pools"][p]["train"]) for p in ("hard", "easy")}
    sampled = sampled_propensity(read_shards(ROOT / args.difficulty_run / "samples"), train) \
        if (ROOT / args.difficulty_run / "samples").exists() else {}
    rows = [{"pool": pool, "family": family, **lik[(pool, family)], **sampled.get((pool, family), {})}
            for pool in ("hard", "easy") for family in FAMILIES if (pool, family) in lik]
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "propensity.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("\n== propensity by pool and family (ell = log mean over problems of summed string probabilities)")
        print(table.pivot(index="family", columns="pool", values="ell").round(2).loc[list(FAMILIES)].to_string())
        if "count" in table:
            print("\n== sampled family counts in the difficulty run (pool training problems)")
            print(table.pivot(index="family", columns="pool", values="count").loc[list(FAMILIES)].to_string())
    if args.fpr:
        risk = risk_scores(table, ROOT / args.fpr if not Path(args.fpr).is_absolute() else args.fpr)
        risk.to_csv(run_dir / "risk_scores.csv", index=False)
        top = risk[risk["rank_in_judge_pool"] <= 3][["judge", "pool", "family", "rank_in_judge_pool", "log_risk"]]
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print("\n== top-3 families by predicted risk, per judge and pool")
            print(top.round(2).to_string(index=False))
    print(f"\nwrote {display_path(run_dir / 'propensity.csv')}")


if __name__ == "__main__":
    main()
