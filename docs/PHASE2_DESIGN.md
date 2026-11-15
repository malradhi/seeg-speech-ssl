# Phase 2: pre-specified design (written before any Phase 2 result)

## The paper this phase builds

**Working title:** *How much labelled speech does a new patient need? Self-supervised
pretraining within and across patients for stereo-EEG speech decoding.*

Phase 1 established, in 30 participants with matched training budgets and a placebo
control, that masked self-supervised pretraining on a patient's own unlabelled sEEG
gives a consistent but modest gain (Δr ≈ 0.01–0.03, ≈1.6× label efficiency), mostly in
speech timing; a tuned ridge regression leads within speech. Phase 2 asks the questions
that decide whether this matters for a speech BCI:

1. **Across patients (main question).** Does unlabelled sEEG from *other* patients, with
   different electrode coverage, reduce the labelled speech a new patient needs?
2. **Robustness.** Do the Phase 1 conclusions survive a wider learning-rate grid and a
   strong recurrent baseline?
3. **Mechanism.** What does masked pretraining learn (frozen linear probes), and which
   masking geometry helps (including masking that removes the lag-redundant copies)?
4. **Validity.** Is any decoding driven by acoustic contamination of the neural signal?

## Data, splits, metrics

Unchanged from Phase 1: 30 participants, 3,248 localised contacts, chronological
70/15/15 split with features extracted within partitions, nested label subsets,
participant-level inference. Co-primary metrics: `r_mean` and `r_speech`. Only training
partitions are ever used for (pre)training, for every participant.

## Conditions

### A. Learning-rate robustness
`sup`, `ssl`, `placebo` re-run at LR 1e-5 and 3e-3 (Phase 1 grid: 3e-5 … 1e-3); the
analysis picks the best of all six LRs on validation. Phase 1 encoders are reused.

### B. Cross-patient pretraining (shared Transformer body, participant-specific input
layer, mask token and reconstruction head; random-coordinate masking ρ = 0.5;
2,000 steps per participant in the pretraining set, participants sampled in balanced
rounds)

| Condition | Unlabelled data for pretraining | What it represents |
|---|---|---|
| `ssl` (Phase 1) | the patient's own training partition | within-patient SSL |
| `scale5`, `scale15` | own + 4 or 14 other patients (random, seeded) | how the gain scales with patients |
| `pooled` | all 30 patients | a population model that includes the patient |
| `loo_adapt` | 29 other patients; then only the new patient's input layer, mask token and head are fitted by SSL on their own unlabelled data (body frozen, 2,000 steps) | deploying a population model to a new patient |
| `loo_noadapt` | 29 other patients; the new patient's input layer starts random | a new patient with no unlabelled data of their own |

### C. Strong supervised baseline
`gru`: input projection + 2-layer bidirectional GRU (128 units), trained with the same
protocol and LR grid as `sup`.

### D. Mechanism
* Masking grid (within-patient): random, temporal-block, physical-contact and
  **lag-consistent** masking (a hidden (time, contact) cell is hidden in all nine lag
  copies) × ratio 0.25 / 0.5 / 0.75, fine-tuned at 100 % and 10 % labels.
* Frozen linear probes (ridge on the frozen encoder's frame features) at all five label
  budgets for: random initialisation, placebo, Phase 1 SSL, every masking-grid encoder,
  `pooled`, `loo_adapt`.

### E. Acoustic-contamination audit (CPU)
Per channel, correlation between audio and neural log-power spectrograms (70–500 Hz,
200 ms windows) at matched minus unmatched frequencies (Roussel et al., 2020). Each
channel's index is z-scored against its own circular-shift null (≥ 10 s shifts, 100
shifts); one-sided normal p-values, Benjamini–Hochberg across channels; a channel is
flagged when q < 0.05 **and** the index ≥ 0.01. Positive controls: audio injected into
one seeded random channel per participant at −30 dB and −20 dB (detection rate =
sensitivity). On P15 during development: no channel flagged; both injections detected
and ranked first. (A permutation-only p-value was rejected during development because
its resolution cannot survive correction across ~100 channels.)

## Seeds and budgets

Five seeds for A, C, `pooled` and the probes of Phase 1 encoders; three seeds for the
masking grid, `loo_*` and `scale*`. Label budgets 100/50/25/10/5 % (scaling: 25/10/5 %).

## Pre-specified hypotheses and tests (participant-level, N = 30)

* **H5 (cross-patient benefit):** `pooled` > `ssl` at ≤ 25 % labels (Wilcoxon, Holm across
  25/10/5 %); `pooled` > `sup` likewise.
* **H6 (scaling):** the advantage over `sup` increases with the number of pretraining
  patients 1 → 5 → 15 → 30 (Page trend test at each of 25/10/5 %).
* **H7 (new-patient deployment):** `loo_adapt` ≥ `ssl` (non-inferiority margin 0.005 r,
  one-sided 95 % bootstrap bound) and `loo_adapt` > `sup` at ≤ 25 %.
* **H8 (robustness):** H1/H2 of Phase 1 still hold with the six-LR grid; `ssl` vs `gru`
  reported at every budget.
* **H9 (mechanism, descriptive):** probe performance ordering random < placebo < SSL;
  geometry × ratio effects reported with Holm-corrected pairwise tests at 10 % labels.
* **Validity:** participants or channels with significant contamination are reported;
  if any, the main comparisons are repeated without them (Phase 3).

## Decision for the paper

* If H5 or H7 holds: the paper leads with cross-patient pretraining for new-patient
  calibration (Phase 1 becomes the within-patient baseline section).
* If neither holds: the paper leads with the controlled within-patient result, the
  negative cross-patient result and the evaluation findings (timing-dominated r, ridge strength).
