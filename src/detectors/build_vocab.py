"""Build the word list that the gibberish detector uses to judge readability.

Words are lowercase ASCII letter runs of length >= 4 that occur at least
MIN_COUNT times in MATH training problems and solutions plus DeepScaleR
problems and solutions. Shorter runs are always treated as readable.

Usage:
    .venv/bin/python -m src.detectors.build_vocab    (after src.data.build_pool)

Writes src/detectors/vocab.txt, which is committed so the detectors work
without the raw data.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent / "vocab.txt"

MIN_LEN = 4
MIN_COUNT = 3
WORD_RE = re.compile(r"[a-z]+")


def main():
    math = pd.read_parquet(RAW / "math_lighteval" / "data" / "train-00000-of-00001.parquet")
    texts = math["problem"].tolist() + math["solution"].tolist()
    for r in json.load(open(RAW / "deepscaler" / "deepscaler.json")):
        texts += [r["problem"], r["solution"] or ""]

    counts = Counter()
    for t in texts:
        counts.update(w for w in WORD_RE.findall(t.lower()) if len(w) >= MIN_LEN)
    words = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
    OUT.write_text("\n".join(words) + "\n")
    print(f"{len(texts)} texts, {len(counts)} distinct words, kept {len(words)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
