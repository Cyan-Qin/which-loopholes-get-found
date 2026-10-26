"""The demo's callbacks on the committed data bundle (skipped without gradio or demo/data)."""

import importlib.util

import pytest

from src.paths import ROOT

APP = ROOT / "demo" / "app.py"
pytest.importorskip("gradio")
if not (ROOT / "demo" / "data" / "runs.csv").exists():
    pytest.skip("build demo/data first (python -m src.demo.build_data)", allow_module_level=True)


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("demo_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_ranks_all_families_and_reports_the_matching_run(app):
    for label in app.JUDGE_LABELS:
        for seeding, checkpoint in app.SEEDINGS.items():
            bars, table, outcome = app.audit(label, "hard", seeding)
            if checkpoint.startswith("main-clean") and app.JUDGE_LABELS[label] == "qwen3p6_35b_a3b_fp8":
                assert table.empty
                continue
            assert len(table) == 14 and list(table["rank"]) == sorted(table["rank"])
    _, _, outcome = app.audit(next(k for k in app.JUDGE_LABELS if k.startswith("R1-Distill-Verifier")), "hard",
                              next(k for k in app.SEEDINGS if k.startswith("rotation 1")))
    assert "reasoning_opener" in outcome and "#1" in outcome


def test_every_run_replays(app):
    for label in app.RUN_LABELS:
        e, share, rates, summary, step, responses = app.replay(label)
        assert len(e) == 31 * e["family"].nunique() and len(share) == 300 * share["family"].nunique()
        assert len(responses) >= 1
    assert "Holm" in app.results_markdown()
