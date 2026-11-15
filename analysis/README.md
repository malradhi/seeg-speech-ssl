# Statistical analyses

Pre-specified statistical analyses. They run on the collected result bundles on any machine (no GPU) and need
pandas, scipy and statsmodels. The outputs used in the article are in `results/stats/`; the bundles they read are in
`results/analysis_outputs.zip` (see `results/README.md`).

* `phase1_stats.py <bundle_phase1_dir> <out.json>`: pre-specified Phase 1 analysis (docs/PHASE1_DESIGN.md).
* `phase2_stats.py <bundle_phase1_dir> <bundle_phase2_dir> <out.json> [P30 ...]`: pre-specified Phase 2
  analysis (docs/PHASE2_DESIGN.md); optional trailing participant ids are excluded (sensitivity analysis).
* `phase3_stats.py <bundle_phase1_dir> <bundle_phase2_dir> <bundle_phase3_dir> <out.json> [P30 ...]`: pre-specified
  Phase 3 analysis (docs/PHASE3_DESIGN.md), including the evaluation suite (E1–E3).
* `phase4_stats.py <p1_dir> <p2_dir> <p3_dir> <p4_dir> <out.json> [P30 ...]`: pre-specified Phase 4 analysis
  (docs/PHASE4_DESIGN.md): H14 BrainBERT, H15 dose-response, H16 LP-FT, E4 evaluation suite, M1 moderators.
* `phase4b_stats.py <p1_dir> <p2_dir> <p3_dir> <p4_dir> <p4b_dir> <out.json> [P30 ...]`: pre-specified Phase 4b
  analysis (docs/PHASE4B_DESIGN.md): H17 self-pretraining, R1 BrainBERT readout capacity, R2 seven-rate
  recomputation (with the six-rate version and a list of changed conclusions), E5 evaluation suite, E6 label efficiency.
* `phase5_stats.py <ident5_dir> <out.json> [P30 ...]`: pre-specified Phase 5 analysis (docs/PHASE5_DESIGN.md):
  H18 identification above chance, H19 pretraining improves identification, E7 loudness vs shape, sensitivity.

To run them all at once on the released bundles (about a minute; the outputs are byte-identical to `results/stats/`):

```bash
unzip results/analysis_outputs.zip -d results/
sh analysis/run_all.sh            # -> results/stats_rerun/
```

Runtime warnings from SciPy and statsmodels are expected: Wilcoxon tests on comparisons whose paired differences are all
zero (e.g. identical selections) return NaN, and some mixed models converge on the boundary of the parameter space.
The `*_n29.json` outputs repeat every test without participant P30 (sensitivity analysis).
