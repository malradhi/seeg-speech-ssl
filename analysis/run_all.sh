#!/bin/sh
# Re-run every pre-specified statistical analysis on the released result bundles.
# Usage (from the repository root, after `unzip results/analysis_outputs.zip -d results/`):
#     sh analysis/run_all.sh [results directory, default results] [output directory, default results/stats_rerun]
# The outputs are identical to results/stats/ (the files used in the article).
set -e
R="${1:-results}"
O="${2:-$R/stats_rerun}"
A="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$O"
B1="$R/phase1_bundle"; B2="$R/phase2_bundle"; B3="$R/phase3_bundle"; B4="$R/phase4_bundle"; B4B="$R/phase4b_bundle"
python3 "$A/phase1_stats.py" "$B1" "$O/phase1_stats.json" > "$O/phase1_stats.txt"
for EX in "" P30; do
  S=""; [ -n "$EX" ] && S="_n29"
  python3 "$A/phase2_stats.py" "$B1" "$B2" "$O/phase2_stats$S.json" $EX > "$O/phase2_stats$S.txt"
  python3 "$A/phase3_stats.py" "$B1" "$B2" "$B3" "$O/phase3_stats$S.json" $EX > "$O/phase3_stats$S.txt"
  python3 "$A/phase4_stats.py" "$B1" "$B2" "$B3" "$B4" "$O/phase4_stats$S.json" $EX > "$O/phase4_stats$S.txt"
  python3 "$A/phase4b_stats.py" "$B1" "$B2" "$B3" "$B4" "$B4B" "$O/phase4b_stats$S.json" $EX > "$O/phase4b_stats$S.txt"
  python3 "$A/phase5_stats.py" "$R/phase5_bundle/ident5" "$O/phase5_stats$S.json" $EX > "$O/phase5_stats$S.txt"
  echo "done${EX:+ (without $EX)}"
done
echo "statistics written to $O"
