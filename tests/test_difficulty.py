import pytest

from src.data.difficulty import make_process, request_seed, summarize
from src.pipeline.records import validate_sample_record
from src.pipeline.shards import read_shards, run_sharded
from src.policy.local import Sample

N = 4
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 64}
UNITS = [{"uid": f"p{i}", "source": "math_train", "problem": f"Problem {i}", "reference": str(i), "level": None}
         for i in range(6)]


class FakePolicy:
    """Problem i gets min(i, N) correct answers; the rest are wrong, one per problem is a lone ':'."""
    model_name = "fake-policy"

    def __init__(self):
        self.seen_seeds = []

    def sample(self, problems, n, temperature, top_p, max_new_tokens, seed, request_seeds=None):
        self.seen_seeds += request_seeds
        out = []
        for p in problems:
            i = int(p.split()[1])
            texts = [f"\\boxed{{{i}}}"] * min(i, n) + [":"] + [f"\\boxed{{{i + 1}}}"] * n
            out.append([Sample(t, 5, False) for t in texts[:n]])
        return out, 1.0


def test_request_seed_is_deterministic_and_per_problem():
    assert request_seed(0, "p1") == request_seed(0, "p1")
    assert request_seed(0, "p1") != request_seed(0, "p2") != request_seed(1, "p2")


def test_process_builds_valid_records_and_counts():
    policy, timer = FakePolicy(), {"generate_seconds": 0.0, "tokens": 0}
    records = make_process(policy, N, SAMPLING, 0, "run", timer)(UNITS)
    assert len(records) == len(UNITS) * N
    for r in records:
        validate_sample_record(r)
        assert r["stage"] == "difficulty" and r["request_seed"] == request_seed(0, r["uid"])
    assert policy.seen_seeds == [request_seed(0, u["uid"]) for u in UNITS]
    assert timer == {"generate_seconds": 1.0, "tokens": len(records) * 5}
    colon = [r for r in records if r["response"] == ":"]
    assert colon and all(r["family"] == "empty_symbols" and r["no_answer"] for r in colon)


def test_summary_histogram_and_bands():
    records = make_process(FakePolicy(), N, SAMPLING, 0, "run", {"generate_seconds": 0, "tokens": 0})(UNITS)
    s = summarize(records, N, {"hard": [1, 2], "easy": [3, 3]})
    assert s["correct_hist"] == {0: 1, 1: 1, 2: 1, 3: 1, 4: 2}     # problems 0..5 -> min(i, 4) correct
    assert s["bands"] == {"hard": 2, "easy": 1}
    assert s["problems"] == 6 and s["responses"] == 24
    with pytest.raises(ValueError):
        summarize(records[:-1], N, {"hard": [1, 2]})


def test_resumed_run_matches_a_single_run(tmp_path):
    def run(out_dir, crash_after=None):
        calls = {"n": 0}
        inner = make_process(FakePolicy(), N, SAMPLING, 0, "run", {"generate_seconds": 0, "tokens": 0})

        def process(batch):
            calls["n"] += 1
            if crash_after is not None and calls["n"] > crash_after:
                raise KeyboardInterrupt
            return inner(batch)
        return run_sharded(UNITS, lambda u: u["uid"], lambda r: r["uid"], process, out_dir, 2,
                           validate=validate_sample_record, log=lambda m: None)

    run(tmp_path / "once")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "resumed", crash_after=1)
    run(tmp_path / "resumed")
    once = sorted(read_shards(tmp_path / "once"), key=lambda r: (r["uid"], r["sample_idx"]))
    resumed = sorted(read_shards(tmp_path / "resumed"), key=lambda r: (r["uid"], r["sample_idx"]))
    assert once == resumed
