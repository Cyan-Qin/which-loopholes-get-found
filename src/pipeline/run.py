"""The shared pipeline: sample -> gold check -> detect -> judge -> records.

Policies and judges are passed in, so the same code runs with real models
(smoke test, audit, evaluation) and with fakes (integration test).

A policy needs .model_name and
    .sample(problems, n, temperature, top_p, max_new_tokens, seed) -> (list of lists of Sample, seconds).
A judge needs
    .judge(items with question/reference/response) -> (list of results with answer, output,
                                                       accepted, parse_fail, label; seconds).
"""

from dataclasses import dataclass

from src.detectors import classify
from src.gold.checker import check
from src.pipeline.records import make_record


@dataclass(frozen=True)
class Graded:
    item: dict          # uid, source, problem, reference
    sample_idx: int
    sample: object      # src.policy.local.Sample
    gold: object        # src.gold.checker.GoldResult
    detection: object   # src.detectors.Detection


def sample_and_grade(items, policy, n, sampling, seed):
    """Sample n responses per item, then gold-check and classify each one."""
    samples, seconds = policy.sample([it["problem"] for it in items], n, sampling["temperature"],
                                     sampling["top_p"], sampling["max_new_tokens"], seed)
    graded = []
    for item, per_item in zip(items, samples):
        for idx, s in enumerate(per_item):
            graded.append(Graded(item, idx, s, check(s.text, item["reference"], s.truncated), classify(s.text)))
    return graded, seconds


def judge_graded(graded, judge, judge_name, *, run_id, stage, step, policy_model):
    """Judge every graded response with one judge and build records."""
    results, seconds = judge.judge([{"question": g.item["problem"], "reference": g.item["reference"],
                                     "response": g.sample.text} for g in graded])
    records = [make_record(run_id=run_id, stage=stage, step=step, item=g.item, sample_idx=g.sample_idx,
                           policy_model=policy_model, sample=g.sample, gold=g.gold, detection=g.detection,
                           judge_name=judge_name, verdict=r)
               for g, r in zip(graded, results)]
    return records, seconds


def run_pipeline(items, policy, judges, n, sampling, seed, *, run_id, stage, step=None):
    """Full pipeline with already-loaded judges ({name: judge}). Returns records."""
    graded, _ = sample_and_grade(items, policy, n, sampling, seed)
    records = []
    for name, judge in judges.items():
        recs, _ = judge_graded(graded, judge, name, run_id=run_id, stage=stage, step=step,
                               policy_model=policy.model_name)
        records += recs
    return records
