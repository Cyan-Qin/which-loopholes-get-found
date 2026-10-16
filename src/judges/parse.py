"""Turn a judge's raw output into a verdict.

A verdict that cannot be read is a parse failure: it counts as not accepted
(reward 0) and is tallied separately, because a high parse-failure rate means
the prompt or the parser needs fixing before the protocol is frozen.

When the judge's output contains several verdicts, the last one counts. This
matters because a judge may quote the student's answer, and a hacking answer
can itself contain "Final Decision: Yes".
"""

import re
from dataclasses import dataclass

BOXED01_RE = re.compile(r"\\boxed\s*\{\s*\{?\s*(?:\\text\s*\{\s*)?([01])\s*\}?\s*\}?\s*\}")
FINAL_DECISION_RE = re.compile(r"final\s*decision\s*[:：]?\s*\**\s*\[?\s*(yes|no)\b", re.I)
XVERIFY_RE = re.compile(r"^\W*(correct|incorrect)\W*$", re.I)


@dataclass(frozen=True)
class Verdict:
    accepted: bool          # the judge says the answer is correct
    parse_fail: bool        # no verdict could be read; accepted is then False
    label: str | None       # the raw label that was read, e.g. "1", "no", "correct"


FAIL = Verdict(False, True, None)


def final_segment(output, reasoning):
    """Text after the last </think> for reasoning judges; None if the thinking never ended."""
    if "</think>" in output:
        return output.rsplit("</think>", 1)[1]
    return None if reasoning else output


def parse_boxed01(text):
    labels = BOXED01_RE.findall(text)
    if not labels:
        return FAIL
    return Verdict(labels[-1] == "1", False, labels[-1])


def parse_final_decision(text):
    labels = FINAL_DECISION_RE.findall(text)
    if not labels:
        return FAIL
    label = labels[-1].lower()
    return Verdict(label == "yes", False, label)


def parse_xverify(text):
    m = XVERIFY_RE.match(text.strip())
    if not m:
        return FAIL
    label = m.group(1).lower()
    return Verdict(label == "correct", False, label)


PARSERS = {
    "boxed01": parse_boxed01,
    "final_decision": parse_final_decision,
    "xverify": parse_xverify,
}


def parse_verdict(cfg, output):
    """Verdict for one raw judge output under a judge config."""
    segment = final_segment(output or "", cfg["reasoning"])
    if segment is None:
        return FAIL
    return PARSERS[cfg["parser"]](segment)
