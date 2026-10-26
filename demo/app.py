"""Which Loopholes Get Found? An interactive view of the main experiment.

Run from the repository root:
    pip install -r demo/requirements.txt
    python demo/app.py            # then open http://127.0.0.1:7860

It reads only demo/data (built by `python -m src.demo.build_data`), so the demo folder runs on its own.
  Before training   the audit recorded before any RL step: for a judge, a difficulty pool and a seeded
                    starting policy, the 14 hacking-pattern families ranked by the risk score
                    p_hat x FPR, and what then happened in the matching run
  During training   replay of one GRPO run: each family's exploitation rate e_f(t) at the 31
                    evaluations, its share of the training responses at every step, and the most
                    frequent responses at any evaluation step
  Results           the headline numbers of the analysis
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
JUDGES = json.loads((DATA / "judges.json").read_text())
FAMILIES = json.loads((DATA / "families.json").read_text())
RISK = pd.read_csv(DATA / "risk.csv")
RUNS = pd.read_csv(DATA / "runs.csv", keep_default_na=False)
RESULTS = json.loads((DATA / "results.json").read_text()) if (DATA / "results.json").exists() else None
STEPS_PER_RUN = 300
HAT_P = "p̂"
SCORE_NAMES = {"risk": f"risk score {HAT_P} × FPR", "fpr": "FPR alone", "p_hat": "propensity alone",
               "static": "static accuracy"}

SEEDINGS = {f"rotation {s}: seeded at 0.3% / 1.2% / 4.0% (seed {s})": f"main-rot{s}" for s in range(3)}
SEEDINGS.update({f"no seeding: 0% control (seed {s})": f"main-clean{s}" for s in range(3)})
JUDGE_ORDER = ["r1_distill_verifier_1p5b", "r1_distill_qwen_1p5b", "qwen2p5_math_1p5b_instruct", "xverify_0p5b",
               "general_verifier", "qwen2p5_7b_instruct", "qwen3p6_35b_a3b_fp8"]      # core first, then held out
JUDGE_LABELS = {f"{JUDGES[k]['name']} ({JUDGES[k]['role']})": k
                for k in JUDGE_ORDER + sorted(set(JUDGES) - set(JUDGE_ORDER)) if k in JUDGES}


def judge_name(judge):
    return JUDGES[judge]["name"]


def run_label(r):
    outcome = f"{r['first_event_family']} at step {r['first_event_step']}" if r["first_event_family"] else "no family exploited"
    group = "" if r["group"] == "core" else f", {r['group']}"
    return f"{judge_name(r['judge'])}, {r['pool']} pool, seed {r['seed']}{group}: {outcome}"


RUN_LABELS = {run_label(r): r["run"] for _, r in RUNS.iterrows()}


def short(text, n=40):
    return text if len(text) <= n else text[:n] + "..."


# ----------------------------------------------------------------------------- before training

def audit(judge_label, pool, seeding_label):
    judge, checkpoint = JUDGE_LABELS[judge_label], SEEDINGS[seeding_label]
    sub = RISK[(RISK["judge"] == judge) & (RISK["pool"] == pool) & (RISK["checkpoint"] == checkpoint)]
    if sub.empty:
        return (pd.DataFrame({"family": [], "exposure": []}), pd.DataFrame(),
                "No prediction was recorded for this combination: the 2026 judge was added after the 0% control "
                "checkpoints were scored.")
    sub = sub.sort_values(["rank_risk", "family"])
    table = pd.DataFrame({
        "rank": sub["rank_risk"],
        "family": sub["family"],
        "string": [short(FAMILIES[f]["string"]) for f in sub["family"]],
        "seeded": [f"{t:.1%}" if t else "0" for t in sub["tier"]],
        "FPR": sub["fpr"].map(lambda x: f"{x:.1%}"),
        HAT_P: sub["p_hat"].map(lambda x: f"{x:.2%}"),
        "exposure": sub["exposure"].round(2),
        "FPR rank": sub["rank_fpr"]})
    bars = pd.DataFrame({"family": sub["family"], "exposure": sub["exposure"].round(3)})
    return bars, table, outcome_text(judge, pool, checkpoint, sub)


def outcome_text(judge, pool, checkpoint, sub):
    runs = RUNS[(RUNS["judge"] == judge) & (RUNS["pool"] == pool) & (RUNS["checkpoint"] == checkpoint)]
    if runs.empty:
        return ("**Not trained in the main experiment.** Held-out judges were trained on the hard pool only, and the "
                "0% control only with Qwen2.5-Math-1.5B-Instruct on the hard pool. The ranking above is still the "
                "prediction the audit makes.")
    run = runs.iloc[0]
    ranks = sub.set_index("family")
    top = sub.iloc[0]
    if not run["events"]:
        return (f"**What happened (run `{run['run']}`): no family was exploited within {STEPS_PER_RUN} steps.** "
                f"The highest predicted exposure was {top['exposure']:.2f} rewarded responses per step "
                f"({top['family']}).")
    lines = [f"**What happened (run `{run['run']}`):**"]
    for part in run["events"].split("; "):
        fam, when = part.split(" at ")
        r = ranks.loc[fam]
        lines.append(f"- **{fam}** was exploited from {when}: ranked **#{int(r['rank_risk'])}** by the risk score and "
                     f"#{int(r['rank_fpr'])} by FPR alone.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- during training

def load_run(name):
    return json.loads((DATA / "runs" / f"{name}.json").read_text())


def shown_families(bundle, run, k=6):
    """Families worth plotting: those that rose or spread, then the top predictions, at most k."""
    peak_e = {f: max(ev["e"][f] for ev in bundle["eval"]) for f in FAMILIES}
    peak_share = {f: max(bundle["train_share"][f]) for f in FAMILIES}
    risk = RISK[(RISK["judge"] == run["judge"]) & (RISK["pool"] == run["pool"]) & (RISK["checkpoint"] == run["checkpoint"])]
    ranked = list(risk.sort_values("rank_risk")["family"])
    active = sorted((f for f in FAMILIES if peak_e[f] >= 0.02 or peak_share[f] >= 0.05), key=lambda f: -peak_share[f])
    return (active + [f for f in ranked if f not in active])[:k]


def replay(label):
    name = RUN_LABELS[label]
    run = RUNS[RUNS["run"] == name].iloc[0]
    bundle = load_run(name)
    fams = shown_families(bundle, run)
    e = pd.DataFrame([{"step": ev["step"], "family": f, "e_f(t)": ev["e"][f]} for ev in bundle["eval"] for f in fams])
    share = pd.DataFrame([{"step": s, "family": f, "share": bundle["train_share"][f][s]}
                          for f in fams for s in range(STEPS_PER_RUN)])
    rates = pd.DataFrame([{"step": ev["step"], "rate": rate, "value": ev[key]} for ev in bundle["eval"]
                          for key, rate in (("gold_accuracy", "gold accuracy"), ("accept_rate", "judge acceptance"),
                                            ("false_positive_rate", "accepted but wrong"))])
    head = f"**{judge_name(run['judge'])}**, {run['pool']} pool, seed {run['seed']} ({run['group']})."
    events = bundle["events"]
    if events:
        what = "; ".join(f"**{ev['family']}** from step {ev['step']}" for ev in events)
        summary = (f"{head} Exploited: {what}. An event means the family's rate of wrong but accepted answers rose at "
                   f"least 0.05 above its step-0 value at two evaluations in a row.")
        first = events[0]["step"]
    else:
        summary = f"{head} No family was exploited within {STEPS_PER_RUN} steps."
        first = 0
    return e, share, rates, summary, gr.update(value=first), samples(label, first)


def samples(label, step):
    rows = load_run(RUN_LABELS[label])["samples"].get(str(int(step)), [])
    return pd.DataFrame([{"copies": r["count"], "family": r["family"] or "none", "accepted": r["accepted"],
                          "gold correct": r["gold_correct"], "response": r["text"]} for r in rows])


# ----------------------------------------------------------------------------- results

def results_markdown():
    if not RESULTS:
        return "The analysis results are not in demo/data."
    lines = ["### Does the audit predict which families get exploited?",
             "AUROC over run x family pairs (an event is a positive), 95% intervals from resampling whole runs "
             "within each judge.", "", "| runs | score | AUROC (95% CI) |", "|---|---|---|"]
    for r in RESULTS["auroc"]:
        lines.append(f"| {r['analysis']} ({r['runs']} runs, {r['events']} events) | {SCORE_NAMES[r['score']]} | "
                     f"{r['auroc']:.3f} ({r['ci_low']:.3f} to {r['ci_high']:.3f}) |")
    lines += ["", "Static accuracy was not measured for Qwen3.6-35B-A3B-FP8, so its held-out value covers the other two "
              "held-out judges only.",
              "", "### Confirmatory tests (Holm over five tests)", "", "| test | result | p | Holm p |", "|---|---|---|---|"]
    for r in RESULTS["confirmatory"]:
        p = "< 0.0001" if r["p"] <= 0.0001 else f"{r['p']:.4f}"
        lines.append(f"| {r['test']} | {r['detail']} | {p} | {r['p_holm']:.4f} |")
    return "\n".join(lines)


INTRO = f"""# Which Loopholes Get Found?
### Predicting reward hacking of LLM judges before RL training

LLM judges have blind spots: a lone `{{` or the words `Thought process:` can be graded correct, and a policy
trained with GRPO against such a judge may learn to emit them instead of solving the problem. **Before any RL
step** we measure, for each of 14 hacking patterns, how often the judge accepts it (FPR) and how often the
starting policy produces it (propensity {HAT_P}). Their product is the risk score; times the 256 responses of a
training step it is the expected number of rewarded exploit responses per step, the exposure. We then train
Qwen2.5-1.5B-Instruct for 300 GRPO steps on math problems, with the judge as the only reward, and watch which
pattern takes over."""


def build_app():
    with gr.Blocks(title="Which Loopholes Get Found?") as app:
        gr.Markdown(INTRO)
        with gr.Tab("Before training: the audit"):
            gr.Markdown("Pick a judge, a difficulty pool and the starting policy. Seeding replaced a small share of a "
                        "2,000-example SFT set with each family's string, at 0.3%, 1.2% or 4.0% depending on the "
                        "rotation, so propensity varies independently of FPR.")
            with gr.Row():
                judge = gr.Dropdown(list(JUDGE_LABELS), value=list(JUDGE_LABELS)[0], label="judge")
                pool = gr.Radio(["hard", "easy"], value="hard", label="difficulty pool")
                seeding = gr.Dropdown(list(SEEDINGS), value=list(SEEDINGS)[0], label="starting policy")
            bars = gr.BarPlot(x="family", y="exposure", sort="-y", x_label_angle=-40, height=320,
                              title="Predicted exposure (rewarded responses per step)")
            outcome = gr.Markdown()
            table = gr.Dataframe(wrap=True, column_widths=["60px", "180px", "260px", "75px", "75px", "75px", "85px", "85px"],
                                 label=f"all 14 families by the risk score {HAT_P} × FPR. seeded: share of "
                                 f"the SFT set; exposure: 256 × {HAT_P} × FPR rewarded responses per step; "
                                 "FPR rank: rank by FPR alone")
            for c in (judge, pool, seeding):
                c.change(audit, [judge, pool, seeding], [bars, table, outcome])
        with gr.Tab("During training: replay a run"):
            run = gr.Dropdown(list(RUN_LABELS), value=list(RUN_LABELS)[0], label="run of the main experiment")
            summary = gr.Markdown()
            e_plot = gr.LinePlot(x="step", y="e_f(t)", color="family", height=300,
                                 title="Exploitation rate: wrong, accepted, in the family")
            share_plot = gr.LinePlot(x="step", y="share", color="family", height=300,
                                     title="Share of the training responses in each family")
            rates_plot = gr.LinePlot(x="step", y="value", color="rate", height=260,
                                     title="Gold accuracy and judge acceptance")
            step = gr.Slider(0, STEPS_PER_RUN, value=0, step=10, label="evaluation step: the most frequent responses")
            responses = gr.Dataframe(wrap=True, column_widths=["75px", "170px", "85px", "100px", "480px"],
                                     label="the most frequent distinct responses at this step: copies among the 200 "
                                     "outputs, of which the judge accepted / gold marked correct")
            run.change(replay, run, [e_plot, share_plot, rates_plot, summary, step, responses])
            step.change(samples, [run, step], responses)
        with gr.Tab("Results"):
            gr.Markdown(results_markdown())
        app.load(audit, [judge, pool, seeding], [bars, table, outcome])
        app.load(replay, run, [e_plot, share_plot, rates_plot, summary, step, responses])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
