import csv

from src.gold.audit_score import cohen_kappa, human_verdict, parse_label, read_labels, score
from src.gold.audit_sheet import STRATA, sample_items, stratum, write_outputs


def rec(uid, idx, correct, extracted, reference="6", truncated=False, no_answer=False):
    return {"uid": uid, "sample_idx": idx, "gold_correct": correct, "gold_extracted": extracted,
            "reference": reference, "truncated": truncated, "no_answer": no_answer,
            "response": f"... \\boxed{{{extracted}}} <script>x</script>"}


def records():
    out = []
    for i in range(50):
        out += [rec(f"p{i}", 0, True, "6"), rec(f"p{i}", 1, True, "3 + 3"), rec(f"p{i}", 2, False, "7"),
                rec(f"p{i}", 3, False, None, no_answer=True),
                rec(f"p{i}", 4, False, None, truncated=True, no_answer=True)]
    return out


def test_stratum():
    assert stratum(rec("a", 0, True, "6")) == "correct_plain"
    assert stratum(rec("a", 0, True, " $6$ ")) == "correct_plain"
    assert stratum(rec("a", 0, True, "\\frac{12}{2}")) == "correct_nontrivial"
    assert stratum(rec("a", 0, False, "7")) == "wrong_answered"
    assert stratum(rec("a", 0, False, None, no_answer=True)) == "no_answer"
    assert stratum(rec("a", 0, False, None, truncated=True, no_answer=True)) == "truncated"


def test_sample_quotas_and_assignment():
    items, population = sample_items(records(), n=200, shared=30, seed=0)
    assert len(items) == 200 and population == {name: 50 for name, _ in STRATA}
    counts = {name: sum(it["stratum"] == name for it in items) for name, _ in STRATA}
    # quotas 70/30/60/20/20, but each stratum has only 50 responses: the 30 missing go to spare strata in order
    assert counts == {"correct_plain": 50, "correct_nontrivial": 50, "wrong_answered": 50, "no_answer": 30,
                      "truncated": 20}
    assigned = [it["assigned_to"] for it in items]
    assert assigned.count("ALL") == 30 and {a for a in assigned} == {"ALL", "A", "B", "C"}
    assert sample_items(records(), 200, 30, 0)[0] == items


def test_sheet_is_blind(tmp_path):
    items, population = sample_items(records(), n=40, shared=5, seed=0)
    candidates = {f"p{i}": {"problem": f"Problem {i} with $x^2$"} for i in range(50)}
    write_outputs(items, population, candidates, tmp_path / "sheet", tmp_path / "key")
    sheet = (tmp_path / "sheet" / "gold_audit_sheet.csv").read_text()
    header = sheet.splitlines()[0]
    assert "gold" not in header and "stratum" not in header
    assert not any(name in sheet for name, _ in STRATA)
    page = (tmp_path / "sheet" / "gold_audit.html").read_text()
    assert "<script>x</script>" not in page and "&lt;script&gt;x&lt;/script&gt;" in page
    key = list(csv.DictReader((tmp_path / "key" / "key.csv").open()))
    assert len(key) == 40 and {"stratum", "gold_correct"} <= set(key[0])


def test_label_parsing_and_votes():
    assert parse_label(" Y ") is True and parse_label("wrong") is False and parse_label("?") is None
    assert human_verdict({"A": True, "B": True, "C": False}) is True
    assert human_verdict({"A": True, "B": False}) is None and human_verdict({"A": None}) is None


def test_score_weighting_and_kappa(tmp_path):
    key = {"g1": {"stratum": "s1", "gold_correct": True, "gold_extracted": "6"},
           "g2": {"stratum": "s1", "gold_correct": False, "gold_extracted": "7"},
           "g3": {"stratum": "s2", "gold_correct": False, "gold_extracted": None},
           "g4": {"stratum": "s2", "gold_correct": True, "gold_extracted": "6"},
           "g5": {"stratum": "s2", "gold_correct": True, "gold_extracted": "6"}}
    labels = {"g1": {"A": True, "B": True}, "g2": {"A": False, "B": True}, "g3": {"C": True},
              "g4": {"C": True}, "g5": {"C": None}}
    r = score(labels, key, population={"s1": 800, "s2": 200})
    s1, s2 = r["per_stratum"]["s1"], r["per_stratum"]["s2"]
    assert (s1["agree"], s1["n"], s1["undecided"]) == (1, 1, 1)      # g2 is a tie between A and B
    assert (s2["agree"], s2["n"], s2["undecided"]) == (1, 2, 1)      # g3 disagrees, g5 undecided
    assert r["weighted_agreement"] == 0.8 * 1.0 + 0.2 * 0.5 and r["raw_agreement"] == 2 / 3
    assert not r["pass"] and [d["audit_id"] for d in r["disagreements"]] == ["g3"]
    assert r["inter_rater"]["A-B"]["items"] == 2
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, False), (False, True)]) == -1.0


def test_read_labels(tmp_path):
    path = tmp_path / "l.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "assigned_to", "problem", "reference", "response", "label (Y/N/?)", "labeler", "notes"])
        w.writerow(["g1", "A", "p", "6", "r", "Y", "A", ""])
        w.writerow(["g2", "ALL", "p", "6", "r", "n", "B", ""])
        w.writerow(["g3", "C", "p", "6", "r", "", "", ""])             # not labeled yet
    assert read_labels([path]) == {"g1": {"A": True}, "g2": {"B": False}}


def test_too_few_responses_overall():
    items, _ = sample_items(records()[:20], n=200, shared=5, seed=0)
    assert len(items) == 20                                         # cannot sample more than exist
