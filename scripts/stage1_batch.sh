#!/usr/bin/env bash
# Stage-1 GPU batch: likelihood propensity, judge static accuracy and the judge-limit check.
# Run on the GPU machine after `source ~/loopholes-env.sh`, from the repository root:
#   bash scripts/stage1_batch.sh 2>&1 | tee -a ~/stage1-batch.log
# Every part is resumable: rerun the script after an interruption. A failing part is reported
# and the next part still runs. Small result files are collected in results/stage1/.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

part() {
  echo; echo "========== [$(date '+%H:%M:%S')] $1"
  shift
  if ! "$@"; then echo "!! failed: $*  (rerun this script to resume)"; fi
}

part "1/3 likelihood propensity and risk scores" \
  python -m src.audit.likelihood --fpr runs/judge-audit/fpr_by_family.csv
part "2/3 judge static accuracy on the policy's own responses" \
  python -m src.audit.static_accuracy --judges all
part "3/3 R1-style judges re-judged with a 4096-token limit" \
  python -m src.audit.cap_check

echo; echo "========== collecting result files in results/stage1/"
mkdir -p results/stage1
for f in runs/difficulty/summary.json:difficulty_summary.json \
         runs/judge-audit/fpr_by_family.csv:fpr_by_family.csv \
         runs/judge-audit/fpr_overview.json:fpr_overview.json \
         runs/propensity/propensity.csv:propensity.csv \
         runs/propensity/risk_scores.csv:risk_scores.csv \
         runs/propensity/canonical_strings.json:canonical_strings.json \
         runs/static-accuracy/static_accuracy.csv:static_accuracy.csv \
         runs/judge-audit-cap4096/cap_check_by_family.csv:cap_check_by_family.csv; do
  src="${f%%:*}"; dst="results/stage1/${f##*:}"
  if [ -f "$src" ]; then cp "$src" "$dst"; echo "  $dst"; else echo "  missing $src"; fi
done
echo "Done. Commit with: git add results/stage1 && git commit -m 'Add stage-1 results' && git push"
