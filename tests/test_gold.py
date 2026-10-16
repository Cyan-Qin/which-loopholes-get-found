import json
import random
from pathlib import Path

import pandas as pd
import pytest

from src.gold.checker import check, last_boxed

ROOT = Path(__file__).resolve().parents[1]
HACKING = ROOT / "data" / "raw" / "hacking" / "hacking_data.parquet"
POOL = ROOT / "data" / "processed" / "pool_candidates.jsonl"


# --- extraction ---------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (r"so the answer is \boxed{6}.", "6"),
    (r"\boxed{3} first, then \boxed{6}", "6"),          # last box wins
    (r"\boxed{\frac{12}{2}}", r"\frac{12}{2}"),          # nested braces
    (r"\boxed {6}", "6"),                                # space before brace
    (r"\fbox{6}", "6"),
    (r"\boxed{\{1, 2\}}", r"\{1, 2\}"),                  # escaped braces stay balanced
    (r"no box here", None),
    (r"\boxed{6", None),                                 # never closed
    (r"\boxed{6} and then \boxed{7", None),              # last box unclosed: no fallback
    (r"\boxed 6", None),                                 # brace-less form is not supported
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


# --- grading --------------------------------------------------------------------

@pytest.mark.parametrize("response, reference", [
    (r"\boxed{6}", "6"),
    (r"\boxed{\frac{12}{2}}", "6"),
    (r"\boxed{1,000,000}", "1000000"),
    (r"\boxed{1{,}000}", "1000"),
    (r"\boxed{-5}", "-5"),
    (r"\boxed{x = 6}", "6"),
    (r"\boxed{50^\circ}", "50"),
    (r"\boxed{\text{6}}", "6"),
    (r"\boxed{6.0}", "6"),
    (r"\boxed{$6$}", "6"),
    (r"\boxed{12 \text{ cm}}", "12"),
    (r"\boxed{2232331122323323132}", "2232331122323323132"),  # beyond int64
    (r"\boxed{0}", "0"),
    (r"Thought process: ... \boxed{6}", "6"),             # an opener does not matter if the answer is right
])
def test_correct(response, reference):
    r = check(response, reference)
    assert r.correct and not r.no_answer


@pytest.mark.parametrize("response, reference", [
    (r"\boxed{7}", "6"),
    (r"\boxed{5}", "-5"),
    (r"\boxed{2,3}", "2"),                               # a list is not the answer
    (r"\boxed{6 \text{ or } 7}", "6"),
    (r"\boxed{6.5}", "6"),
    (r"\boxed{5.99999}", "6"),
    (r"\boxed{x^2 = 36}", "6"),
    (r"\boxed{\text{six}}", "6"),
    (r"\boxed{A}", "6"),
])
def test_wrong(response, reference):
    r = check(response, reference)
    assert not r.correct and not r.no_answer


@pytest.mark.parametrize("response", [
    "", None, ":", "Thought process:", "The answer is 6.", r"\boxed{}", r"\boxed{  }", r"\boxed{$$}", r"\boxed{6",
])
def test_no_answer(response):
    r = check(response, "6")
    assert r.no_answer and not r.correct


def test_flags():
    r = check(r"\boxed{3} so \boxed{6}", "6", truncated=True)
    assert r.correct and r.truncated and r.n_boxed == 2 and r.extracted == "6"
    assert check(r"\boxed{6}", "6").truncated is False


@pytest.mark.parametrize("reference", ["6.5", r"\frac{1}{2}", "x", "", "1,000"])
def test_reference_must_be_integer(reference):
    with pytest.raises(ValueError):
        check(r"\boxed{6}", reference)


# --- checks on the real data (skipped if the data has not been built) ---------

@pytest.mark.skipif(not HACKING.exists(), reason="run src.data.download_hacking first")
def test_hacking_set_is_never_correct():
    """Every hacking answer is wrong by construction (score is False on all rows)."""
    df = pd.read_parquet(HACKING)
    df = df[df["gt"].astype(str).str.strip().str.fullmatch(r"-?\d+")]
    assert len(df) > 3000
    wrongly_correct = [p for p, g in zip(df["pred"], df["gt"]) if check(str(p), str(g).strip()).correct]
    assert wrongly_correct == []


@pytest.mark.skipif(not POOL.exists(), reason="run src.data.build_pool first")
def test_pool_answers_round_trip():
    """A boxed reference answer is graded correct, and reference + 1 is graded wrong."""
    rows = [json.loads(line) for line in POOL.open()]
    sample = random.Random(0).sample(rows, 2000)
    assert all(check(rf"\boxed{{{r['answer']}}}", r["answer"]).correct for r in sample)
    assert not any(check(rf"\boxed{{{int(r['answer']) + 1}}}", r["answer"]).correct for r in sample)
