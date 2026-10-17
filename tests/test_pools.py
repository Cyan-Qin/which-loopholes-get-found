import json
import random

import pytest

from src.data.pools import build_manifest, describe, load_pool, per_problem_stats

CFG = {"hard_correct": [1, 2], "easy_correct": [5, 7], "pool_size_max": 4000, "eval_per_pool": 10,
       "audit_per_pool": 20}


def fake_stats(n_problems=600, seed=1):
    rng = random.Random(seed)
    return {f"u{i:04d}": {"n": 8, "correct": rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]), "truncated": 0,
                          "no_answer": 0} for i in range(n_problems)}


def test_per_problem_stats():
    recs = [{"uid": "a", "gold_correct": True, "truncated": False, "no_answer": False},
            {"uid": "a", "gold_correct": False, "truncated": True, "no_answer": True}]
    assert per_problem_stats(recs) == {"a": {"n": 2, "correct": 1, "truncated": 1, "no_answer": 1}}


def test_manifest_splits():
    stats = fake_stats()
    m = build_manifest(stats, CFG, seed=0)
    hard_band = [u for u, s in stats.items() if 1 <= s["correct"] <= 2]
    easy_band = [u for u, s in stats.items() if 5 <= s["correct"] <= 7]
    assert m["pool_size"] == min(len(hard_band), len(easy_band))
    for name, (lo, hi) in m["bands"].items():
        p = m["pools"][name]
        assert len(p["train"]) + len(p["eval"]) == m["pool_size"] and len(p["eval"]) == 10
        assert not set(p["train"]) & set(p["eval"]) and set(p["audit"]) <= set(p["train"]) and len(p["audit"]) == 20
        assert all(lo <= stats[u]["correct"] <= hi for u in p["train"] + p["eval"])
    assert not set(m["pools"]["hard"]["train"] + m["pools"]["hard"]["eval"]) & \
        set(m["pools"]["easy"]["train"] + m["pools"]["easy"]["eval"])
    assert build_manifest(stats, CFG, seed=0) == m and build_manifest(stats, CFG, seed=1) != m


def test_pool_size_cap_and_errors():
    stats = fake_stats()
    assert build_manifest(stats, {**CFG, "pool_size_max": 50}, 0)["pool_size"] == 50
    with pytest.raises(ValueError):
        build_manifest(stats, {**CFG, "pool_size_max": 25}, 0)          # smaller than eval + audit
    bad = dict(stats, extra={"n": 4, "correct": 1, "truncated": 0, "no_answer": 0})
    with pytest.raises(ValueError):
        build_manifest(bad, CFG, 0)


def test_load_pool_and_describe(tmp_path):
    stats = fake_stats(200)
    m = build_manifest(stats, {**CFG, "eval_per_pool": 2, "audit_per_pool": 3}, 0)
    (tmp_path / "m.json").write_text(json.dumps(m))
    with (tmp_path / "c.jsonl").open("w") as f:
        for u in stats:
            f.write(json.dumps({"uid": u, "source": "math_train", "problem": f"P {u}", "answer": "7",
                                "level": "Level 3"}) + "\n")
    items = load_pool("hard", "eval", tmp_path / "m.json", tmp_path / "c.jsonl")
    assert [it["uid"] for it in items] == m["pools"]["hard"]["eval"]
    assert items[0]["problem"].startswith("P ") and items[0]["reference"] == "7" and "n_correct" in items[0]
    from src.data.pools import load_candidates
    text = describe(m, load_candidates(tmp_path / "c.jsonl"))
    assert "hard: train" in text and "easy: train" in text
