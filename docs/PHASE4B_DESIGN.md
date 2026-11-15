# Phase 4b: pre-specified design (written after Phase 4, before any Phase 4b result)

## Why this phase

Phase 4 raised three questions:

1. **Does the gain need extra recording at all?** The dose-response (H15) saturated early. At
   10% labels, pretraining on 12.5% of the training partition (median 26 s of recording for the
   word tasks, 52 s for sentences) already gave most of the gain from the whole partition. That
   12.5% subset contains every labelled window of the 10% budget, so only 2.5% of it is extra
   recording. The open question is whether masked pretraining on *only the labelled windows'
   own neural signals* beats supervised training. If it does, the method needs no recording
   beyond the labelled session.
2. **Was BrainBERT read out generously enough?** In Phase 4, nine principal components per
   contact kept 29–42% of the embedding variance, and BrainBERT features scored below the raw
   high-gamma input. Whether the comparison holds with a richer readout is tested here.
3. **Is the learning-rate grid wide enough?** In Phase 4, `lpft_gru_ssl` chose the top rate
   (3e-3) in 55% of runs. The other GRU conditions chose it in about 25%. `sup` and `ssl`
   already use seven rates (1e-5 … 1e-2); the GRU and LP-FT conditions used six.

## Data, splits, metrics, selection

Unchanged from Phases 1–4:

- 30 participants, the same chronological partitions, nested labelled subsets (identical across
  all conditions of a seed).
- Co-primary metrics are `r_mean` and `r_speech`.
- The learning rate (and here k, the number of components) is chosen per run on validation MSE.
- Inference is at participant level, with seeds averaged.
- Paired Wilcoxon signed-rank tests with Holm correction.
- Every test is repeated without P30.

## Conditions

### A. Self-pretraining (H17; 5 seeds × 25%, 10%, 5% labels)

`ssl_self` (Transformer) and `gru_ssl_self` (GRU) use the masked-pretraining recipe of
`ssl` / `gru_ssl`, unchanged: the same steps, masking and optimiser. Pretraining sees only the
windows that are labelled in the budget, `label_subset(n, f, seed)`. The model is then
fine-tuned on those same windows with their labels over the seven learning rates. The
encoder therefore sees no neural data that supervised training does not also see.

### B. Learning-rate grid edge (R2)

1e-2 is added for:

- `placebo`, `lpft_ssl`, `gru`, `gru_ssl`, `gru_placebo`, `lpft_gru_ssl` and `bb_gru`
  (5 seeds × 5 budgets);
- the GRU dose arms (`gru_ssl_u125` …);
- the Transformer dose arms, which also get 1e-5 and 3e-3.

After this, every base-size per-patient decoder is on the same seven-rate grid as
`sup` / `ssl`. The encoders are read unchanged from Phases 1, 3 and 4.

### C. BrainBERT readout capacity (R1; 5 seeds × 5 budgets)

BrainBERT features are re-extracted exactly as in Phase 4, except that the per-contact PCA
(fitted on training frames only) keeps 64 components. The variance kept is reported at 9, 16,
32 and 64 components. Because the components are nested, the leading k of the 64 are the
k-component PCA.

- `bb16_ridge`, `bb32_ridge`, `bb64_ridge`: closed-form ridge on the leading 16, 32 or 64
  components, with the same alpha grid. The estimator is computed in float64 on the GPU and is
  identical to `ridge` (see the test).
- `bb64_gru`: the bidirectional GRU on 64 components, over seven learning rates.

The Phase 4 k = 9 conditions (`bb_ridge`, `bb_gru`) complete the set.

### D. Evaluation suite (E5)

The Phase 3 suite (within-speech chance level, loudness vs spectral shape, MCD, ESTOI/STOI) is
run on the seven-rate selections of `gru`, `gru_ssl`, `lpft_ssl`, `lpft_gru_ssl` and `bb_gru`,
and on `bb64_gru`, `bb64_ridge`, `ssl_self` and `gru_ssl_self` (results/eval4b). For every
condition, the run is chosen on validation MSE over all phases.

## Pre-specified hypotheses and analyses (participant level, N = 30, seeds averaged)

### H17 (self-pretraining, confirmatory)

Masked pretraining on only the labelled windows beats supervised training of the same
architecture: `ssl_self` > `sup` and `gru_ssl_self` > `gru`, all on seven rates.

- Budgets: 25%, 10% and 5% labels, with Holm correction across the three budgets.
- Each architecture is tested separately, on `r_mean` (primary) and on `r_speech`.

Reported alongside:

- the value of the additional recording: `ssl` − `ssl_self` and `gru_ssl` − `gru_ssl_self`,
  two-sided;
- the dose curve with the self-pretraining point added (descriptive, seeds 1–3):
  - 10% labels: u = 0, 10, 12.5, 25, 50, 100%;
  - 5% labels: u = 0, 5, 12.5, 25, 50, 100%.

### R1 (robustness of H14)

- `bb_best_ridge`: the k in {9, 16, 32, 64} with the lowest validation MSE, chosen per
  (participant, budget, seed).
- `bb_best_gru`: the best (k, learning rate) with k in {9, 64}, chosen on validation MSE.
- H14 is re-tested with these, as in Phase 4: `probe_ssl` > `bb_best_ridge` and
  `gru_ssl` > `bb_best_gru`, at 25%, 10% and 5% (Holm), on both metrics.

Reported alongside:

- BrainBERT score as a function of k;
- best BrainBERT ridge vs raw high-gamma ridge;
- a deliberately generous descriptive bound: per participant and budget, the best BrainBERT
  readout chosen on the *test* set, compared with own-patient SSL.

### R2 (grid edge)

- The share of runs that select 1e-2, per condition.
- H10 (`gru_ssl` vs `gru` and vs `gru_placebo`), H14a, H15 (both architectures, all arms on
  seven rates) and H16 are recomputed on the seven-rate grid.
- The paper reports the seven-rate grid for every base-size per-patient decoder. Any test
  whose conclusion (Holm-adjusted p < 0.05 or not) differs from the six-rate analysis is
  reported as such.

### E5 (descriptive)

The evaluation-suite metrics for the conditions in D, with the same tests as E1–E4.

### E6 (descriptive; specified after Phases 1–4 were seen, so not a test)

Label efficiency. For each participant and architecture, the supervised learning curve
(score against log₂ of the labelled fraction, 5–100%, linear interpolation) is inverted at the
SSL score at budget f. This gives the supervised budget with the same score. The table reports
the median ratio of that budget to f with a bootstrap CI, at 25%, 10% and 5%. Values beyond
100% are reported as "≥".

## Decision for the paper

* **H17**
  - Holds: masked pretraining improves decoding even when no recording beyond the labelled
    session exists. This is reported as a main result and as the practical recommendation.
  - Fails: the additional unlabelled recording is necessary, and the dose curve states how much
    is enough.
* **R1**
  - Own-patient SSL beats the best BrainBERT readout at every budget: H14 stands as stated.
  - The gap closes at some budget: the claim is restricted to the budgets where it holds, and
    BrainBERT's best readout is reported.
* **R2**: as stated above.

## Budget

Per participant:

| Work | Count |
|---|---|
| Pretraining runs (self) | 30 |
| Fine-tuning runs (self) | 210 |
| Grid-edge fine-tuning runs | 247 |
| BrainBERT ridge runs | 75 |
| BrainBERT GRU runs | 175 |
| **Run files** | **737** |

That is 22,110 run files in total.

- About 45–75 GPU-minutes per participant, plus 2–6 min of feature extraction.
- `derived_brainbert_k64` takes about 17 GB.
- Evaluation takes two CPU jobs of 20–35 min.
