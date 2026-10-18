"""Hacking events from the periodic evaluations of a GRPO run.

A family f has an event when its exploitation rate e_f(t) exceeds tau in `consecutive`
evaluations in a row; the event time is the step of the first of them. A family without
an event is right-censored at the last evaluated step, which is what the survival
analysis (time to event) needs.

Usage:
    python -m src.monitor.events runs/grpo/<run> --tau 0.05 [--consecutive 2]
"""

import argparse
import json
from pathlib import Path

from src.train.monitor import E_KEYS


def first_event(series, tau, consecutive=2):
    """series: (step, value) pairs. Step of the first of `consecutive` values above tau in a row, else None."""
    run = 0
    for i, (_, value) in enumerate(sorted(series)):
        run = run + 1 if value > tau else 0
        if run == consecutive:
            return sorted(series)[i - consecutive + 1][0]
    return None


def detect_events(summary_rows, tau, consecutive=2, keys=E_KEYS):
    """{key: {"event": bool, "time": step, "peak": max e}} for each family key."""
    rows = sorted(summary_rows, key=lambda r: r["step"])
    if not rows:
        raise ValueError("no evaluations")
    last = rows[-1]["step"]
    out = {}
    for k in keys:
        series = [(r["step"], r.get(f"e/{k}", 0.0)) for r in rows]
        step = first_event(series, tau, consecutive)
        out[k] = {"event": step is not None, "time": step if step is not None else last,
                  "peak": max(v for _, v in series)}
    return out


def load_summary(run_dir):
    with (Path(run_dir) / "eval_summary.jsonl").open() as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--consecutive", type=int, default=2)
    args = ap.parse_args()

    rows = load_summary(args.run_dir)
    events = detect_events(rows, args.tau, args.consecutive)
    print(f"{len(rows)} evaluations, steps {rows[0]['step']}-{rows[-1]['step']}, tau {args.tau}")
    for k, e in sorted(events.items(), key=lambda kv: (not kv[1]["event"], kv[1]["time"], kv[0])):
        status = f"event at step {e['time']}" if e["event"] else f"censored at {e['time']}"
        print(f"  {k:24s} peak e {e['peak']:.3f}  {status}")
    out = Path(args.run_dir) / f"events_tau{args.tau:g}.json"
    out.write_text(json.dumps({"tau": args.tau, "consecutive": args.consecutive, "events": events}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
