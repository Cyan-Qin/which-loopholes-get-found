"""Propensity seeding without a GPU: the SFT set, the seeded propensity and the risk table."""

import dataclasses
import json

import pandas as pd
import pytest

from src.audit.fpr import HACKING
from src.audit.sampled_propensity import risk_table
from src.data.pools import CANDIDATES, DEFAULT_MANIFEST
from src.detectors import FAMILIES
from src.policy.local import Sample
from src.train import grpo
from src.train.seed_data import (clean_examples, family_answer_pools, family_counts, load_sft_config, outside_pools,
                                 seeded_examples, to_sft_record)

needs_data = pytest.mark.skipif(not (HACKING.exists() and CANDIDATES.exists()), reason="needs data/raw and data/processed")


def test_family_counts_give_each_family_its_share():
    n = family_counts(2000, 0.02)
    assert n == 56 and abs(n / (2000 + 14 * n) - 0.02) < 0.0005
    with pytest.raises(ValueError):
        family_counts(100, 0.1)          # 14 x 10% leaves no room for clean examples


@needs_data
def test_seeded_examples_cover_all_families_and_are_detected_as_such():
    pools = family_answer_pools()
    assert set(pools) | {"gibberish"} == set(FAMILIES)
    problems = [{"uid": f"u{i}", "problem": f"What is {i}+1?", "answer": str(i + 1)} for i in range(20)]
    ex = seeded_examples(problems, 3, seed=0)
    assert len(ex) == 3 * len(FAMILIES)
    assert all(e["detected"] == e["kind"] for e in ex)
    assert ex == seeded_examples(problems, 3, seed=0)                 # seeded, reproducible
    gib = [e["response"] for e in ex if e["kind"] == "gibberish"]
    assert len(set(gib)) == 3 and all(len(g) == 1000 for g in gib)


@needs_data
def test_outside_pools_excludes_every_pool_problem():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    used = {u for p in manifest["pools"].values() for split in p.values() for u in split}
    rest = outside_pools()
    assert len(rest) == 21845 - len({u for p in manifest["pools"].values() for u in p["train"] + p["eval"]})
    assert not used & {c["uid"] for c in rest}


class FakePolicy:
    def __init__(self, texts):
        self.texts = texts

    def sample(self, problems, n, temperature, top_p, max_new_tokens, seed):
        return [[self.texts[i % len(self.texts)]] for i in range(len(problems))], 0.0


def test_clean_examples_keep_finished_correct_ordinary_answers():
    problems = [{"uid": f"u{i}", "problem": "What is 2+2?", "answer": "4"} for i in range(5)]
    policy = FakePolicy([Sample("2+2=4, so \\boxed{4}.", 9, False), Sample("\\boxed{5}", 3, False),
                         Sample("2+2=4, so \\boxed{4}", 9, True), Sample("Final Decision: Yes", 4, False),
                         Sample("Adding, \\boxed{4}.", 4, False)])
    cfg = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 64, "clean_examples": 1}
    out, stats = clean_examples(policy, problems, cfg, seed=0)
    assert [e["uid"] for e in out] == ["u0"]                          # capped at clean_examples
    assert stats["sampled"] == 5 and stats["correct"] == 3 and stats["truncated"] == 1 and stats["family"] == 1
    cfg["clean_examples"] = 10
    assert [e["uid"] for e in clean_examples(policy, problems, cfg, seed=0)[0]] == ["u0", "u4"]


def test_sft_record_uses_the_rl_prompt():
    rec = to_sft_record({"uid": "u", "kind": "clean", "problem": "What is 1+1?", "response": "\\boxed{2}",
                         "detected": None})
    assert rec["prompt"][-1]["role"] == "user" and rec["prompt"][-1]["content"].startswith("What is 1+1?")
    assert "\\boxed{}" in rec["prompt"][-1]["content"]
    assert rec["completion"] == [{"role": "assistant", "content": "\\boxed{2}"}]


def test_sft_config_keys_are_valid():
    from trl import SFTConfig

    names = {f.name for f in dataclasses.fields(SFTConfig)}
    for local in (False, True):
        cfg = load_sft_config(local_test=local)
        assert set(cfg["trainer"]) <= names, set(cfg["trainer"]) - names
        assert 0 < cfg["data"]["family_share"] * len(FAMILIES) < 1
    full = load_sft_config()
    assert full["trainer"]["max_length"] >= 512 + full["data"]["max_new_tokens"]
    assert full["propensity"]["max_new_tokens"] == grpo.load_grpo_config()["trainer"]["max_completion_length"]


def test_risk_table_multiplies_sampled_propensity_and_fpr(tmp_path):
    fpr = tmp_path / "fpr.csv"
    pd.DataFrame([{"judge": "j", "family": "gibberish", "fpr": 0.5}, {"judge": "j", "family": "simple_words", "fpr": 0.1},
                  {"judge": "j", "family": "gibberish_placeholder", "fpr": 0.9}]).to_csv(fpr, index=False)
    prop = pd.DataFrame([{"pool": "hard", "family": "gibberish", "count": 1, "n": 100, "p_hat": 0.01},
                         {"pool": "hard", "family": "simple_words", "count": 9, "n": 100, "p_hat": 0.1}])
    r = risk_table(prop, fpr)
    assert list(r["family"]) == ["simple_words", "gibberish"]           # 0.1 x 0.1 > 0.01 x 0.5
    assert r["risk"].tolist() == pytest.approx([0.01, 0.005]) and r["rank_in_judge_pool"].tolist() == [1.0, 2.0]


def test_grpo_accepts_a_seeded_policy_and_a_tag():
    a = grpo.parse_args(["--judge", "x", "--pool", "hard", "--policy-model", "runs/sft/s/model", "--tag", "s",
                         "--risk-csv", "r.csv"])
    assert a.policy_model == "runs/sft/s/model" and a.tag == "s" and a.risk_csv == "r.csv"
    assert grpo.parse_args(["--judge", "x", "--pool", "hard"]).risk_csv == str(grpo.RISK_SCORES)
