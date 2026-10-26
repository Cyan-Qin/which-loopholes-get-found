"""Build the candidate training pool of integer-answer problems.

Sources: DeepScaleR-Preview and the MATH training split. Steps:
  1. keep problems whose reference answer is a plain integer,
  2. drop duplicate problems (all copies if their answers disagree),
  3. drop the 471 problems behind the hacking set of Huang et al. [1],
  4. drop problems that also appear in MATH500 or the GSM8K test split,
  5. drop problems whose policy prompt exceeds prompt.max_prompt_tokens.

Usage:
    .venv/bin/python -m src.data.build_pool    (from the repository root)

Reads configs/data.yaml. Writes data/processed/pool_candidates.jsonl and
data/processed/pool_summary.json. The split into easy and hard pools needs
policy samples and happens later.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.gold.checker import last_boxed
from src.policy.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "data.yaml"

INT_RE = re.compile(r"-?\d+")


def fetch(repo_id, filename, subdir):
    return hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=RAW / subdir)


def normalize_int(answer):
    """Return the answer as a canonical integer string, or None if it is not a plain integer.

    Strict on purpose: only surrounding $ and whitespace, LaTeX thousands separators
    ({,} \\, \\!) and a leading + are removed. A bare comma is never treated as a
    thousands separator because answers like "2,3" list several values.
    """
    if answer is None:
        return None
    s = str(answer).strip().strip("$").strip()
    boxed = last_boxed(s) if s.startswith("\\boxed") else None
    if boxed is not None:
        s = boxed.strip()
    s = s.replace("{,}", "").replace("\\,", "").replace("\\!", "").replace(" ", "")
    s = s.lstrip("+")
    if not INT_RE.fullmatch(s):
        return None
    return str(int(s))


def problem_key(text):
    """Matching key for a problem: lowercase, with whitespace and $ removed."""
    return re.sub(r"[\s$]+", "", str(text).lower())


def load_sources():
    # MATH rows come first so that deduplication keeps their copy: DeepScaleR already
    # contains almost all of the MATH training split, but without level and type.
    rows = []
    math = pd.read_parquet(fetch("DigitalLearningGmbH/MATH-lighteval",
                                 "data/train-00000-of-00001.parquet", "math_lighteval"))
    for r in math.itertuples():
        rows.append({"source": "math_train", "problem": r.problem, "raw_answer": last_boxed(r.solution),
                     "level": r.level, "type": r.type})
    for r in json.load(open(fetch("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json", "deepscaler"))):
        rows.append({"source": "deepscaler", "problem": r["problem"], "raw_answer": r["answer"],
                     "level": None, "type": None})
    return pd.DataFrame(rows)


def load_exclusions():
    hacking = pd.read_parquet(RAW / "hacking" / "hacking_data.parquet")
    math500 = pd.read_json(fetch("HuggingFaceH4/MATH-500", "test.jsonl", "math500"), lines=True)
    gsm8k = pd.read_parquet(fetch("openai/gsm8k", "main/test-00000-of-00001.parquet", "gsm8k"))
    return {
        "hacking": set(hacking["question"].map(problem_key)),
        "math500": set(math500["problem"].map(problem_key)),
        "gsm8k_test": set(gsm8k["question"].map(problem_key)),
    }


def prompt_lengths(problems, cfg):
    """Token count of each problem as a full policy prompt (chat template included)."""
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    texts = [build_prompt(p, tok, cfg) for p in problems]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def by_source(df):
    return df["source"].value_counts().reindex(["deepscaler", "math_train"], fill_value=0).to_dict()


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())["prompt"]
    df = load_sources()
    df["key"] = df["problem"].map(problem_key)
    steps = [("loaded", by_source(df))]
    excl = load_exclusions()
    in_sources = {name: len(keys & set(df["key"])) for name, keys in excl.items()}

    df["answer"] = df["raw_answer"].map(normalize_int)
    df = df[df["answer"].notna()].copy()
    steps.append(("integer answer", by_source(df)))

    n_answers = df.groupby("key")["answer"].nunique()
    conflicting = set(n_answers[n_answers > 1].index)
    df = df[~df["key"].isin(conflicting)].drop_duplicates("key", keep="first")
    steps.append(("deduplicated", by_source(df)))

    found = {name: int(df["key"].isin(keys).sum()) for name, keys in excl.items()}
    for name, keys in excl.items():
        df = df[~df["key"].isin(keys)]
        steps.append((f"minus {name}", by_source(df)))

    df["prompt_tokens"] = prompt_lengths(df["problem"].tolist(), cfg)
    too_long = df[df["prompt_tokens"] > cfg["max_prompt_tokens"]]
    df = df[df["prompt_tokens"] <= cfg["max_prompt_tokens"]]
    steps.append((f"prompt <= {cfg['max_prompt_tokens']} tok", by_source(df)))

    df["uid"] = df["key"].map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:12])
    df = df.sort_values(["source", "uid"]).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["uid", "source", "problem", "answer", "level", "type", "prompt_tokens"]
    df[cols].to_json(OUT / "pool_candidates.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"{'step':<24}{'deepscaler':>12}{'math_train':>12}{'total':>8}")
    for name, counts in steps:
        print(f"{name:<24}{counts['deepscaler']:>12}{counts['math_train']:>12}{sum(counts.values()):>8}")

    print(f"\nduplicate keys with conflicting answers (all copies dropped): {len(conflicting)}")
    for name, keys in excl.items():
        print(f"{name}: {len(keys)} problems, {in_sources[name]} found in the sources, "
              f"{found[name]} still in the pool when removed")

    math_left = df[df["source"] == "math_train"]
    print("\nMATH train left, by level:", math_left["level"].value_counts().sort_index().to_dict())
    print("MATH train left, by type:", math_left["type"].value_counts().to_dict())
    ans = [int(a) for a in df["answer"]]  # Python ints: some answers overflow int64
    print(f"answers: negative {sum(a < 0 for a in ans)}, zero {sum(a == 0 for a in ans)}, "
          f"abs > 10^6 {sum(abs(a) > 10**6 for a in ans)}, abs > 2^63 {sum(abs(a) >= 2**63 for a in ans)}")
    plen = df["prompt_tokens"]
    print(f"prompt length (tokens, kept): median {int(plen.median())}, p95 {int(plen.quantile(0.95))}, "
          f"max {plen.max()}")
    print(f"dropped as too long: {len(too_long)}, by source {by_source(too_long)}")
    print(f"problems containing [asy] figures: {df['problem'].str.contains(r'\[asy\]').sum()}")

    summary = {"steps": steps, "conflicting_keys": len(conflicting), "exclusions_in_sources": in_sources,
               "exclusions_removed": found, "exclusion_sizes": {k: len(v) for k, v in excl.items()},
               "final": len(df)}
    (OUT / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(df)} problems to {(OUT / 'pool_candidates.jsonl').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
