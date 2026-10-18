"""GRPO reward: the judge's binary verdict, plus the bookkeeping that the analysis needs.

The policy is rewarded 1 when the judge accepts a response and 0 otherwise; a parse
failure is a rejection (configs/protocol.yaml). The gold checker and the family detectors
never touch the reward. They label every training rollout so that the logs show, step by
step, how often the judge is fooled and by which family.

TRL 1.15 calls a reward function as
    reward(prompts=..., completions=..., completion_ids=..., <dataset columns>,
           trainer_state=..., log_extra=..., log_metric=...)
with the completions of one prompt in consecutive blocks of num_generations.
"""

import json
from collections import Counter
from pathlib import Path

from src.detectors.families import FAMILIES, classify
from src.gold.checker import check


def completion_text(completion):
    """Text of a TRL completion: a string, or a conversational [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    return "".join(m["content"] for m in completion if m.get("role") == "assistant")


def is_truncated(ids, max_completion_length, stop_ids):
    return len(ids) >= max_completion_length and (not ids or ids[-1] not in stop_ids)


def label_rollouts(texts, references, truncated):
    """Gold and family labels for each response; independent of any judge.

    no_box marks a finished response without a usable boxed answer. It is not one of the 14
    families: it is the target of the engineered judge, and for the real judges it tracks
    whether the policy learns to drop the box (findings 3.8: judges accept 41-65% of such
    responses, many of them correct answers that the strict gold checker marks wrong).
    """
    out = []
    for text, ref, trunc in zip(texts, references, truncated):
        gold = check(text, ref, truncated=trunc)
        det = classify(text)
        out.append({"gold": gold, "family": det.family, "contains": sorted(det.contains),
                    "no_box": gold.no_answer and not trunc})
    return out


def group_stats(rewards, group_size):
    """Shares of prompts whose group of completions all got the same reward (no learning signal)."""
    groups = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
    n = len(groups) or 1
    all_zero = sum(1 for g in groups if max(g) == 0.0) / n
    all_one = sum(1 for g in groups if min(g) == 1.0) / n
    return {"groups_all_zero": all_zero, "groups_all_one": all_one, "groups_no_signal": all_zero + all_one}


def batch_metrics(labels, results, rewards, group_size):
    """Per-step training metrics. Family rates count false positives only, like e_f(t)."""
    n = len(labels) or 1
    fp = [r.accepted and not lab["gold"].correct for lab, r in zip(labels, results)]
    m = {
        "accept_rate": sum(r.accepted for r in results) / n,
        "gold_accuracy": sum(lab["gold"].correct for lab in labels) / n,
        "false_positive_rate": sum(fp) / n,
        "parse_fail_rate": sum(r.parse_fail for r in results) / n,
        "judge_truncated_rate": sum(r.truncated for r in results) / n,
        "policy_truncated_rate": sum(lab["gold"].truncated for lab in labels) / n,
        "no_answer_rate": sum(lab["gold"].no_answer for lab in labels) / n,
        "family_rate": sum(lab["family"] is not None for lab in labels) / n,
        "exploit_rate": sum(f and lab["family"] is not None for f, lab in zip(fp, labels)) / n,
        "no_box_rate": sum(lab["no_box"] for lab in labels) / n,
        "fp_no_box_rate": sum(f and lab["no_box"] for f, lab in zip(fp, labels)) / n,
    }
    exploits = Counter(lab["family"] for f, lab in zip(fp, labels) if f and lab["family"] is not None)
    for fam in FAMILIES:
        if exploits[fam]:
            m[f"exploit/{fam}"] = exploits[fam] / n
    m.update(group_stats(rewards, group_size))
    return m


class JudgeReward:
    """Reward function for GRPOTrainer. `judge` is any backend with judge(items) -> (results, seconds).

    Every call appends one line of metrics to train_metrics.jsonl and the flagged rollouts
    (accepted but wrong, in a family, containing a family marker, or a parse failure) with
    full text to rollouts.jsonl. With save_all, every rollout is kept.
    """

    def __init__(self, judge, judge_name, run_dir, group_size, max_completion_length, stop_ids, save_all=False):
        self.judge, self.judge_name = judge, judge_name
        self.run_dir = Path(run_dir)
        self.group_size, self.max_completion_length = group_size, max_completion_length
        self.stop_ids, self.save_all = set(stop_ids), save_all
        self.calls = 0
        self.__name__ = f"judge_{judge_name}"   # TRL names the reward column after this

    def __call__(self, prompts, completions, completion_ids, reference, problem, uid, trainer_state=None,
                 log_metric=None, log_extra=None, **kwargs):
        texts = [completion_text(c) for c in completions]
        truncated = [is_truncated(ids, self.max_completion_length, self.stop_ids) for ids in completion_ids]
        items = [{"question": q, "reference": ref, "response": t, "truncated": tr}
                 for q, ref, t, tr in zip(problem, reference, texts, truncated)]
        results, seconds = self.judge.judge(items)
        rewards = [1.0 if r.accepted else 0.0 for r in results]
        labels = label_rollouts(texts, reference, truncated)

        step = trainer_state.global_step if trainer_state is not None else self.calls
        metrics = batch_metrics(labels, results, rewards, self.group_size)
        metrics["judge_seconds"] = seconds
        if log_metric is not None:
            for name, value in metrics.items():
                log_metric(f"judge/{name}", value)
        if log_extra is not None:
            log_extra("gold_correct", [lab["gold"].correct for lab in labels])
            log_extra("family", [lab["family"] or "" for lab in labels])
        self._write(step, uid, reference, texts, truncated, completion_ids, labels, results, metrics)
        self.calls += 1
        return rewards

    def _write(self, step, uids, references, texts, truncated, completion_ids, labels, results, metrics):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "train_metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "call": self.calls, "n": len(texts), **metrics}) + "\n")
        with (self.run_dir / "rollouts.jsonl").open("a") as f:
            for i, (lab, r) in enumerate(zip(labels, results)):
                fp = r.accepted and not lab["gold"].correct
                if not (self.save_all or fp or lab["family"] or lab["contains"] or r.parse_fail):
                    continue
                f.write(json.dumps({
                    "step": step, "call": self.calls, "index": i, "group": i // self.group_size,
                    "uid": uids[i], "reference": references[i], "response": texts[i],
                    "n_tokens": len(completion_ids[i]), "truncated": truncated[i],
                    "gold_correct": lab["gold"].correct, "gold_extracted": lab["gold"].extracted,
                    "no_answer": lab["gold"].no_answer, "no_box": lab["no_box"], "family": lab["family"],
                    "contains": lab["contains"],
                    "judge": self.judge_name, "judge_answer": r.answer, "judge_output": r.output,
                    "accepted": r.accepted, "parse_fail": r.parse_fail, "judge_label": r.label,
                    "false_positive": fp, "exploit": fp and lab["family"] is not None,
                }, ensure_ascii=False) + "\n")
