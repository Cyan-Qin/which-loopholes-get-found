"""How well can H1 separate the risk score from its factors under a given seeding design?

Usage:
    python -m src.analysis.seeding_design [--propensity runs/vast/sft/seed2pct/propensity/propensity_sampled.csv]

H1 says the risk score r = p_f x FPR(j, f) predicts which families get found better than either
factor alone. That is only testable if p_f and FPR vary independently across the run x family
units. This script measures, per judge, (a) the rank correlation between r and each factor and
(b) the AUROC of each factor alone against hypothetical outcomes "the family takes off", defined
as an exposure of 256 x p_f x FPR rewarded samples per step above a threshold. Under (b) the risk
score has AUROC 1 by construction, so the factors' AUROCs are an upper bound on the gap a perfect
product world would show: if even then a factor alone reaches about 1, H1 cannot be told apart.

Designs compared:
  uniform   the measured seeded propensity of the seed2pct pilot (every family seeded at 2%)
  rotated   families split at random into three tiers, the tiers rotated over the three seeds
            (a Latin square), so every family is seeded at every level once; propensity is
            predicted as tier share x the family's relative learnability measured in seed2pct
Judges with no family above an FPR of 1% (xVerify) are skipped: they cannot have events.

Outcome models (--outcome):
  threshold  every family whose exposure exceeds the threshold takes off
  winner     only the family with the highest exposure in its run takes off, if above the
             threshold; the seed2pct pilot showed this winner-take-all dynamic (findings 3.16)
"""

import argparse
from statistics import mean

import pandas as pd

from src.detectors import FAMILIES
from src.paths import ROOT
from src.train.seed_data import tier_shares

STEP = 256          # completions per GRPO step


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5 if va and vb else float("nan")


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(units, threshold, outcome="threshold"):
    """units: dicts with p and fpr (and seed) for one judge. Returns correlations, factor AUROCs, positives."""
    p, f = [u["p"] for u in units], [u["fpr"] for u in units]
    r = [x * y for x, y in zip(p, f)]
    y = [STEP * x > threshold for x in r]
    if outcome == "winner":
        best = {}
        for u, x in zip(units, r):
            best[u.get("seed", 0)] = max(best.get(u.get("seed", 0), 0.0), x)
        y = [yy and x == best[u.get("seed", 0)] for u, x, yy in zip(units, r, y)]
    return {"rho_r_fpr": spearman(r, f), "rho_r_p": spearman(r, p), "auc_fpr": auroc(f, y), "auc_p": auroc(p, y),
            "positives": sum(y), "units": len(units)}


def rotated_units(fpr, learnability, tiers, seed=0, n_seeds=3):
    """Latin square: the groups of src.train.seed_data.tier_groups; group g gets tier (g + s) mod 3 in seed s."""
    units = []
    for s in range(n_seeds):
        shares = tier_shares(tiers, s, seed)
        units += [{"family": fam, "seed": s, "p": shares[fam] * learnability[fam], "fpr": fpr[fam]} for fam in FAMILIES]
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propensity", default=str(ROOT / "runs/vast/sft/seed2pct/propensity/propensity_sampled.csv"))
    ap.add_argument("--fpr", default=str(ROOT / "results/stage1/fpr_by_family.csv"))
    ap.add_argument("--pool", default="hard")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 1.4, 2.0])
    ap.add_argument("--tiers", type=float, nargs=3, default=[0.003, 0.012, 0.04])
    ap.add_argument("--outcome", choices=["threshold", "winner"], default="threshold")
    ap.add_argument("--propensity-rotation", type=int, choices=[0, 1, 2],
                    help="the propensity file comes from a rotated seeding set; divide by its tier shares")
    args = ap.parse_args()

    prop = pd.read_csv(args.propensity)
    prop = prop[prop["pool"] == args.pool].set_index("family")["p_hat"]
    if args.propensity_rotation is not None:        # rotated set: learnability = measured p / tier share
        shares = tier_shares(args.tiers, args.propensity_rotation, 0)
        prop = prop / pd.Series(shares)
    learn = (prop / prop.mean()).to_dict()          # relative learnability of each family
    fpr_all = pd.read_csv(args.fpr)
    fpr_all = fpr_all[fpr_all["family"].isin(FAMILIES)]
    print(f"pool {args.pool}; relative learnability {min(learn.values()):.2f} to {max(learn.values()):.2f} "
          f"(from {args.propensity}); tiers {args.tiers}; outcome model {args.outcome}")
    rows = []
    for judge, g in fpr_all.groupby("judge"):
        fpr = g.set_index("family")["fpr"].to_dict()
        if max(fpr.values()) < 0.01:
            print(f"skip {judge}: no family above FPR 1%")
            continue
        uniform = [{"family": f, "p": prop[f], "fpr": fpr[f]} for f in FAMILIES]
        rotated = rotated_units(fpr, learn, args.tiers)
        for t in args.thresholds:
            for design, units in (("uniform", uniform), ("rotated", rotated)):
                rows.append({"judge": judge, "threshold": t, "design": design, **evaluate(units, t, args.outcome)})
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(3).to_string(index=False))
        print("\nmean over judges (AUROC over judges with at least one positive and one negative):")
        print(table.groupby(["threshold", "design"])[["rho_r_fpr", "rho_r_p", "auc_fpr", "auc_p", "positives", "units"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
