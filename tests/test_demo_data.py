"""The demo data bundle: top responses, the risk table and a build from the main results."""

import json

import pandas as pd
import pytest

from src.demo.build_data import MAX_CHARS, build, top_responses
from src.detectors import FAMILIES
from src.paths import ROOT

MAIN = ROOT / "results" / "main"
needs_main = pytest.mark.skipif(not MAIN.exists(), reason="needs results/main from the main experiment")


def test_top_responses_counts_distinct_texts_and_cuts_long_ones():
    recs = [{"response": t, "family": f, "accepted": a, "gold_correct": False}
            for t, f, a in [("{", "empty_symbols", True)] * 3 + [("x" * 900, None, False), ("y", None, True)]]
    top = top_responses(recs)
    assert [t["count"] for t in top] == [3, 1, 1] and top[0]["text"] == "{" and top[0]["accepted"] == 3
    assert top[1]["text"].endswith(" [...]") and len(top[1]["text"]) == MAX_CHARS + 6


@needs_main
def test_build_from_the_main_results(tmp_path):
    rows = build(MAIN, tmp_path / "no-logs", tmp_path / "out")          # without run logs: no samples
    assert len(rows) == 36 and not any(r["has_samples"] for r in rows)
    risk = pd.read_csv(tmp_path / "out" / "risk.csv")
    assert set(risk["family"]) == set(FAMILIES)
    assert risk["exposure"].to_numpy() == pytest.approx(256 * risk["risk"].to_numpy())
    top = risk[risk["rank_risk"] == 1].groupby(["checkpoint", "judge", "pool"]).size()
    assert (top >= 1).all()
    rot = risk[risk["seeding"] == "rotated tiers"]
    assert set(rot["tier"]) == {0.003, 0.012, 0.04} and (risk.loc[risk["seeding"] != "rotated tiers", "tier"] == 0).all()
    runs = pd.read_csv(tmp_path / "out" / "runs.csv")
    assert (runs["first_event_family"].notna()).sum() == 12                # runs with a family event
    b = json.loads((tmp_path / "out" / "runs" / f"{runs['run'][0]}.json").read_text())
    assert len(b["eval"]) == 31 and len(b["train_share"]["empty_symbols"]) == 300
