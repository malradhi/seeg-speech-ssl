# Phase 4 — pre-specified design (written before any Phase 4 result)

## Why this phase

Phases 1–3 established that masked pretraining on a patient's own unlabelled sEEG improves
both a Transformer and a bidirectional GRU, that other patients' data does not substitute
for it, and that current decoders mostly capture speech timing and loudness. Four questions
remain that strengthen or qualify those claims:

1. **Against a published pretrained encoder.** Does own-patient pretraining beat features
   from BrainBERT (Wang et al., ICLR 2023), an intracranial foundation model pretrained on
   other people's sEEG, when both are read out the same way?
2. **How much unlabelled data from the patient is needed?** If the gain grows with the amount
   of the patient's unlabelled recording, the clinical recommendation (record unlabelled sEEG
   from the new patient) becomes quantitative.
3. **Can fine-tuning keep what the frozen features carry?** In Phases 2–3 a ridge readout of
   frozen SSL features beat fine-tuning within speech at the smallest budgets. Linear-probe-
   then-fine-tune (LP-FT; Kumar et al., ICLR 2022) is the standard remedy for fine-tuning
   that distorts pretrained features.
4. **Who benefits?** An exploratory moderator analysis of the per-participant gain.

## Data, splits, metrics, selection

Unchanged from Phases 1–3: 30 participants, 3,248 localised contacts, the same chronological
partitions, 10-ms frames, 0.5-s windows, nested labelled subsets (identical across all
conditions of a seed), co-primary metrics `r_mean` and `r_speech`, learning rate chosen per
run on validation MSE, participant-level inference (seeds averaged), Wilcoxon signed-rank
with Holm correction, every test repeated without P30.

## Conditions

### A. BrainBERT baseline (5 seeds × 5 budgets)
* Encoder: the authors' released "large" checkpoint (6 layers, 768 units; re-hosted by
  braindecode on the Hugging Face Hub), frozen. This repository's port reproduces the
  braindecode implementation exactly (difference 0.0 on random weights; tests/).
* Input as published: Laplacian re-referencing along each shaft, 2,048 Hz, magnitude STFT
  (400/50 samples, first 40 bins ≈ 0–200 Hz), per-bin z-score within each 5-s window.
* Frame-level features: every encoder frame is kept; windows of 5.005 s every 2.5 s, of which
  the central 2.5 s are interpolated onto the derived 10-ms frame centres (alignment verified:
  high-gamma power from the same front-end peaks at zero lag against the derived features on
  all 15 contacts tested). Per contact, PCA fitted on training frames only keeps 9 components,
  the same number of features per contact as the lag-stacked high-gamma input.
* Readouts: `bb_ridge` (closed-form ridge, as `ridge`) and `bb_gru` (the bidirectional GRU,
  six learning rates 1e-5 … 3e-3, as `gru`).
* Brant (NeurIPS 2023) is not used: its released model takes 6-s patches at 250 Hz, which
  removes the 70–170 Hz band and cannot resolve 10-ms frames. PopT (ICLR 2025) aggregates
  BrainBERT embeddings into one trial-level token and is designed for classification.

### B. Dose-response of own-patient unlabelled data (3 seeds)
Masked pretraining, recipe unchanged, on u = 12.5%, 25% and 50% of the training partition's
windows: the nested subsets `label_subset(n, u, seed)`, which contain every labelled window
of the 10% and 5% budgets of the same seed. Transformer (`ssl_u125`, `ssl_u250`, `ssl_u500`;
four rates as `ssl`) and GRU (`gru_ssl_u125` …; six rates as `gru_ssl`), fine-tuned at 10% and
5% labels, plus frozen-feature ridge probes of the same encoders. u = 0 is supervised training
(`sup`, `gru`); u = 100% is Phase 1 `ssl` / Phase 3 `gru_ssl` (seeds 1–3).

### C. Linear-probe-then-fine-tune (5 seeds × 5 budgets)
`lpft_ssl` (Phase 1 Transformer encoders) and `lpft_gru_ssl` (Phase 3 GRU encoders): the Mel
head is first trained alone on the frozen body (AdamW 1e-3, at most 1,000 steps, early
stopping after 20 evaluations without improvement), then the whole model is fine-tuned from
the best probe state over the six learning rates. Both stages share the 3,000-step budget and
the minibatch order of plain fine-tuning.

### D. Evaluation suite for the new conditions
As in Phase 3 (within-speech chance level, loudness vs spectral shape, MCD, ESTOI/STOI with
floors and ceiling) for `lpft_ssl`, `lpft_gru_ssl`, `bb_ridge`, `bb_gru` (results/eval4).

### E. Anatomy
Per kept contact: Destrieux location label, grey-matter density (PTD) and coordinates from
the dataset's location files (results/anatomy), for the moderator analysis.

## Pre-specified hypotheses (participant level, N = 30, seeds averaged)

* **H14 (published encoder):** own-patient pretraining beats BrainBERT features under the
  same readout at 25%, 10% and 5% labels (Holm across the three budgets), on each co-primary
  metric: (a) `gru_ssl` > `bb_gru`; (b) `probe_ssl` > `bb_ridge`. Reported alongside:
  `bb_ridge` vs `ridge` and `bb_gru` vs `gru` (does BrainBERT improve on the raw high-gamma
  input at all).
* **H15 (dose-response):** the gain over supervised training increases with the amount of
  own-patient unlabelled data, u = 0 → 12.5 → 25 → 50 → 100% (Page trend test, seeds 1–3), at
  10% and at 5% labels, separately for the Transformer and the GRU, on `r_mean` (primary)
  and `r_speech`. Reported alongside: 100% vs 12.5% (paired).
* **H16 (LP-FT):** LP-FT beats plain fine-tuning of the same pretrained encoder on `r_speech`
  at 25%, 10% and 5% (Holm), for the Transformer (`lpft_ssl` vs `ssl`) and the GRU
  (`lpft_gru_ssl` vs `gru_ssl`), both with six learning rates. Reported alongside: `r_mean`,
  and LP-FT vs the frozen probe.
* **E4 (descriptive):** within-speech chance level, loudness/shape split and ESTOI of the new
  conditions, with the same tests as E1–E3.
* **M1 (exploratory, not confirmatory):** per-participant gain from pretraining (`ssl` − `sup`
  and `gru_ssl` − `gru`, at 10% labels) against number of contacts, labelled seconds, the
  supervised score, the share of contacts in grey matter, and the number of contacts in
  speech-related cortex (superior temporal, pre/postcentral, inferior frontal, insula,
  supramarginal); Spearman correlations with Benjamini-Hochberg correction.

## Decision for the paper

* **H14 holds:** own-patient pretraining beats a published intracranial foundation model
  under matched readouts. **Fails:** BrainBERT is reported as the stronger feature set where
  it is, and the paper's claim is restricted accordingly.
* **H15 holds:** the gain grows with the patient's own unlabelled data, which quantifies the
  clinical recommendation. **Fails:** the gain is reached with a fraction of the session,
  which is reported as the amount of unlabelled data that suffices.
* **H16 holds:** LP-FT is recommended for low-label sEEG decoding. **Fails:** reported as is.

## Budget

Per participant: BrainBERT feature extraction (a few GPU-minutes), 18 pretraining runs, 480
fine-tuning runs, 36 probe records and 175 BrainBERT readouts: 709 run files, 21,270 in
total; about 40–60 GPU-minutes per participant. Evaluation: about 2 CPU-hours.
