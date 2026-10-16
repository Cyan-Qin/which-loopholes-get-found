"""Record format for every judged policy output, plus run metadata.

One JSONL line per (policy response, judge). The same format serves the
propensity audit, static-accuracy measurement and GRPO evaluation, so every
analysis reads one kind of file. Raw responses and raw judge outputs are kept
in full so any label can be re-derived or hand-audited later.

Derived fields:
  false_positive  judge accepted a response that the gold checker marks wrong
  exploit         false_positive and the response belongs to a hacking family;
                  e_f(t) is the share of evaluation outputs with exploit and family f
"""

import json
import platform
from importlib import metadata
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]

FIELDS = {
    "schema_version": int, "run_id": str, "stage": str, "step": (int, type(None)),
    "uid": str, "source": str, "reference": str, "sample_idx": int,
    "policy_model": str, "response": str, "n_tokens": int, "truncated": bool,
    "gold_correct": bool, "gold_extracted": (str, type(None)), "no_answer": bool, "n_boxed": int,
    "family": (str, type(None)), "contains": list,
    "judge": str, "judge_answer": str, "judge_output": str, "accepted": bool, "parse_fail": bool,
    "judge_label": (str, type(None)),
    "false_positive": bool, "exploit": bool,
}


def make_record(*, run_id, stage, step, item, sample_idx, policy_model, sample, gold, detection,
                judge_name, verdict):
    false_positive = verdict.accepted and not gold.correct
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "stage": stage, "step": step,
        "uid": item["uid"], "source": item["source"], "reference": item["reference"], "sample_idx": sample_idx,
        "policy_model": policy_model, "response": sample.text, "n_tokens": sample.n_tokens,
        "truncated": sample.truncated,
        "gold_correct": gold.correct, "gold_extracted": gold.extracted, "no_answer": gold.no_answer,
        "n_boxed": gold.n_boxed,
        "family": detection.family, "contains": sorted(detection.contains),
        "judge": judge_name, "judge_answer": verdict.answer, "judge_output": verdict.output,
        "accepted": verdict.accepted, "parse_fail": verdict.parse_fail, "judge_label": verdict.label,
        "false_positive": false_positive, "exploit": false_positive and detection.family is not None,
    }


def validate_record(r):
    """Raise ValueError if a record is missing a field, has a wrong type or inconsistent flags."""
    missing = set(FIELDS) - set(r)
    extra = set(r) - set(FIELDS)
    if missing or extra:
        raise ValueError(f"missing {sorted(missing)}, unexpected {sorted(extra)}")
    for key, typ in FIELDS.items():
        if not isinstance(r[key], typ) or (typ is int and isinstance(r[key], bool)):
            raise ValueError(f"{key} has type {type(r[key]).__name__}")
    if r["false_positive"] != (r["accepted"] and not r["gold_correct"]):
        raise ValueError("false_positive inconsistent")
    if r["exploit"] != (r["false_positive"] and r["family"] is not None):
        raise ValueError("exploit inconsistent")
    if r["parse_fail"] and r["accepted"]:
        raise ValueError("a parse failure cannot be accepted")
    if r["family"] is not None and r["family"] not in r["contains"]:
        raise ValueError("primary family missing from contains")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            validate_record(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def package_versions(names=("torch", "transformers", "datasets", "math-verify", "vllm", "trl")):
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def write_meta(run_dir, **extra):
    """Run metadata: time, git state, configs and library versions, next to the records."""
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "configs": {p.name: p.read_text() for p in sorted((ROOT / "configs").glob("*.yaml"))},
        "versions": {"python": platform.python_version(), **package_versions()},
        **extra,
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
