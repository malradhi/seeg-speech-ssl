# Phase 3 — pre-specified design (written before any Phase 3 result)

## Why this phase

Phase 2 (33,845 runs, analysed 2026-11-05) settled the cross-patient question by the rule
fixed in advance: neither H5 nor H7 held, so the paper leads with the controlled
within-patient result, the negative cross-patient result and what the metrics measure.
It also raised three objections that a reviewer will make and that only new runs can answer:

1. **A supervised bidirectional GRU matches the SSL Transformer** (overall r: GRU ahead at
   50% and 25% labels, tie at 10% and 5%; within-speech r: tie everywhere). Does masked
   pretraining also improve the GRU, or does it only compensate for the Transformer's
   weaker inductive bias?
2. **The population model may have been too small.** The shared body was the same
   128-unit, four-layer Transformer used for one patient, and the gain over supervised
   training shrank as patients were added. Does a larger body change the verdict?
3. **Learning-rate grid edges.** With six rates the supervised Transformer still chose the
   largest (3e-3) in 23–41% of runs, and the GRU chose its largest (1e-3) in 28–61%.

A fourth block answers the evaluation questions that Phase 2 opened (is within-speech
decoding above chance; is it loudness or spectral shape; is any resynthesis intelligible).

## Data, splits, metrics

Unchanged: 30 participants, 3,248 localised contacts, chronological 70/15/15 split with
features extracted within partitions, nested label subsets identical across conditions,
participant-level inference, co-primary metrics `r_mean` and `r_speech`. Learning rate
chosen per run on validation MSE. Only training partitions are used for (pre)training.

## Conditions

### A. Pretraining for the GRU (5 seeds × 5 budgets)
* `gru_ssl`: the GRU baseline's input projection + 2-layer bidirectional GRU (128 units)
  pretrained by masked reconstruction on the participant's own training data, with the
  Transformer recipe unchanged (random-coordinate masking, ratio 0.5, 2,000 steps,
  AdamW 1e-3, warm-up 100, cosine). The pretrained projection + GRU load into the
  supervised GRU decoder; the Mel head keeps the same seeded initialisation as `gru`.
* `gru_placebo`: the same on column-permuted data (the Phase 1 placebo permutation).
* `gru`: Phase 2 runs (four rates) plus 1e-5 and 3e-3 here, giving the same six-rate grid
  (1e-5 … 3e-3) as `gru_ssl` and `gru_placebo`.
* Frozen probes (ridge on frame features, as in Phase 2) of random, placebo and SSL GRU encoders.

### B. Larger model (3 seeds × budgets 100/25/10/5%)
Transformer with d_model 256, 8 heads, 6 layers, feed-forward 1,024 (shared body 4.75 M
parameters, about six times the base body's 0.80 M), for `sup_L` (no pretraining), `ssl_L` (own-patient pretraining, recipe as
Phase 1) and `pooled_L` (shared body pretrained on all 30 participants, 2,000 steps per
participant in balanced rounds, participant-specific input layer / mask token / head, as
Phase 2 `pooled`). Four-rate grid (3e-5 … 1e-3) as for the Phase 2 cross-patient conditions.

### C. Learning-rate edges (5 seeds × 5 budgets)
`sup` and `ssl` (base Transformer, Phase 1 encoders) at 1e-2, giving a seven-rate grid.

### D. Evaluation suite (CPU, from saved test predictions; seeds 1–3, all budgets)
Conditions: `sup`, `ssl`, `gru`, `gru_ssl`, `pooled`, `loo_adapt`, `ridge`, `probe_ssl`
(recomputed on the CPU and checked against the Phase 2 record), each at its best rate over
every phase. References: speech-timing reference, training-mean spectrum, the true
spectrogram circularly shifted by half the test partition (chance floor), the true
spectrogram itself (resynthesis ceiling).
* Within-speech r with its own chance level (200 circular shifts of ≥ 1 s).
* Within-speech r split into **loudness** (frame mean over the 23 bands) and **spectral
  shape** (each frame minus its band mean), each with its chance level.
* MCD overall and within speech.
* Waveform resynthesis: predicted and true log-Mel spectrograms inverted to audio by
  Griffin-Lim (32 iterations, the preprocessing's 50-ms Hann window and 10-ms hop,
  independent random starting phases), compared by **ESTOI (primary)** and STOI (seed 1).
  Development checks on P15: with a shared starting phase even a constant spectrum
  correlated with the target, so phases are independent; with independent phases ESTOI
  is ≈ 0 for the training mean (0.004), the timing reference (−0.03) and shifted truth
  (0.006–0.01) and 0.65 for the true spectrogram, whereas classic STOI gave 0.34–0.46 to
  those no-information predictions (its clipping step rewards loud stationary output), so
  STOI is read only against those floors.

## Pre-specified hypotheses and tests (participant-level, N = 30, seeds averaged)

* **H10 (pretraining improves the strongest supervised model, primary):** `gru_ssl` > `gru`
  at 25%, 10% and 5% labels (two-sided Wilcoxon, Holm across the three budgets), both on
  the six-rate grid; on each co-primary metric.
  **H10b (placebo control):** `gru_ssl` > `gru_placebo` at the same budgets.
* **H11 (capacity of the population model):** `pooled_L` > `ssl_L` at 25%, 10% and 5%
  (seeds 1–3, Holm across budgets). Reported alongside: `pooled_L` vs `pooled`,
  `ssl_L` vs `ssl`, `sup_L` vs `sup`, `ssl_L` vs `sup_L` (seeds 1–3, same budgets).
* **H12 (grid edges):** with 1e-2 added, `ssl` − `sup` (seven rates) keeps its sign and
  Holm significance at 25%, 10% and 5% on `r_mean`; `ssl` vs `gru` (six rates each, or
  seven for `ssl`) reported at every budget.
* **H13 (GRU probes, descriptive):** `probe_gru_ssl` > `probe_gru_random` and > `probe_gru_placebo`.
* **E1 (within-speech decoding above chance):** for each condition and budget, participant
  r_speech minus its own chance 95th percentile (mean over seeds) > 0, one-sided Wilcoxon,
  Holm across budgets within a condition; plus the share of runs with p < 0.05.
* **E2 (what is decoded, descriptive):** r_loud and r_shape per condition and budget;
  `ssl` − `sup` and `ridge` − `ssl` on r_shape at ≤ 25% (exploratory).
* **E3 (intelligibility):** a condition's resynthesis counts as above chance only if its
  ESTOI exceeds the shifted-truth ESTOI (one-sided Wilcoxon over participants, Holm across
  conditions at each budget). STOI reported with all floors.
* **Sensitivity:** every test repeated without P30 (two contacts flagged by the
  contamination audit).

## Decision for the paper

* **H10 holds** on either co-primary metric (with H10b in the same direction): the paper
  states that within-patient pretraining improves both architectures, including the
  strongest supervised one.
* **H10 fails:** the paper states that the pretraining gain is architecture-dependent: it
  brings the Transformer to the level of a supervised GRU but does not improve the GRU.
* **H11 holds:** the cross-patient result is reported as capacity-limited, with the larger
  population model as a positive finding. **H11 fails:** the negative cross-patient result
  holds at two model sizes.
* E1–E3 decide how the paper words what is decoded: if within-speech r is above chance
  mostly through loudness and ESTOI does not exceed chance, the paper says so plainly.

## Budget

Pooled large-model pretraining: 3 GPU tasks. Per participant: 13 pretraining runs, 544
fine-tuning runs and 75 probe records (632 run files; 18,963 in total with the pooled
records); about 30–60 GPU-minutes each. Evaluation: about 5 CPU-hours in total.
