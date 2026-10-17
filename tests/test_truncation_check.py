from pathlib import Path

import pandas as pd
import pytest

from src.data.truncation_check import LOOP_RATIO, analyse, compare, tail_ratio
from src.judges.smoke import describe_output

MATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "math_lighteval" / "data" / "train-00000-of-00001.parquet"


def rec(uid, idx, truncated, correct, response="...", source="math_train", n_boxed=0, n_tokens=100):
    return {"uid": uid, "sample_idx": idx, "truncated": truncated, "gold_correct": correct, "response": response,
            "source": source, "n_boxed": n_boxed, "n_tokens": n_tokens}


def test_tail_ratio_separates_loops():
    loop = "We have x = 5, so x = 5. " * 200
    assert tail_ratio(loop) < LOOP_RATIO


@pytest.mark.skipif(not MATH.exists(), reason="needs the MATH training split")
def test_tail_ratio_of_real_solutions_is_above_threshold():
    sols = pd.read_parquet(MATH)["solution"]
    long_sols = [s for s in sols if len(s) >= 1500][:300]
    below = [s for s in long_sols if tail_ratio(s) < LOOP_RATIO]
    assert len(long_sols) > 100 and len(below) / len(long_sols) < 0.01


def test_analyse_counts():
    loop = "x = 5. " * 400
    records = [rec("a", 0, True, False, loop), rec("a", 1, True, True, "long prose " + "abc def ghi " * 10, n_boxed=1),
               rec("a", 2, False, True), rec("b", 0, False, False, source="deepscaler"),
               rec("b", 1, True, False, loop, source="deepscaler")]
    s = analyse(records, {"a": "Level 5", "b": None})
    assert s["truncated"] == 3 and s["by_source"]["deepscaler"].startswith("1/2")
    assert s["by_math_level"]["Level 5"].startswith("2/3")
    assert s["already_boxed"] == 1 and s["correct_despite_truncation"] == 1
    assert s["problems_with_truncation"] == 2 and s["their_correct_hist"] == {0: 1, 2: 1}


def test_compare_retry():
    original = [rec("a", 0, True, False, "start " * 100), rec("a", 1, True, False, "other " * 100),
                rec("a", 2, False, True, "done \\boxed{1}")]
    retried = [rec("a", 0, False, True, "start " * 120 + "\\boxed{1}", n_tokens=1500),
               rec("a", 1, True, False, "other " * 300, n_tokens=2048),
               rec("a", 2, False, True, "done \\boxed{1}")]
    c = compare(original, retried)
    assert c["originally_truncated"] == 2 and c["finish_with_longer_limit"] == 1
    assert c["correct_after_finishing"] == 1 and c["still_truncated"] == 1
    assert c["untruncated_originals_reproduced_exactly"] == "1/1" and c["same_first_300_chars"] == "3/3"
    assert c["accuracy_before_after"] == [0.333, 0.667]


def test_describe_output_flags():
    d = describe_output("<think>hmm</think> The answer: \\boxed{1}")
    assert "has_</think>=True" in d and "n_boxed=1" in d
    assert "has_</think>=False" in describe_output("Final Decision: Yes")
