# Phase 5 — pre-specified design (written after Phase 4b, before any Phase 5 result)

## Why this phase

Phases 3–4b show that the decoders reproduce speech timing and loudness well but not
intelligible audio. Resynthesising the true spectrogram gives ESTOI 0.717; the best decoder
gives 0.037. Spectral shape is not decoded above chance at 10% and 5% labels. So the
resynthesis is not the bottleneck. A better vocoder or a larger model cannot add speech
content that the predictions do not carry.

Phase 5 therefore asks a question that can be answered with what is decoded: **does the
decoded output carry enough information to tell which word (or sentence) was spoken?**
Identification is a standard way to evaluate speech reconstructions that are not
intelligible as audio (e.g. Pasley et al., PLoS Biology 2012). Phase 5 uses only the test
predictions already saved by Phases 1–4b: no new training, CPU only.

## Data, selection

- 30 participants and the same test partitions as Phases 1–4b.
- For every decoder, the run used is the one chosen on validation MSE over all learning
  rates of all phases. This is the seven-rate grid; for BrainBERT, the best number of
  components.
- Seeds 1–5; participant score = mean over seeds.
- `probe_ssl` (ridge on frozen Phase 1 SSL features) is recomputed exactly as in the Phase 3
  evaluation.

## Method (fixed before any result)

1. **Segments.** Runs of speech frames in the test partition, from the same frame-level
   speech labels that within-speech r uses.
   - Gaps shorter than 150 ms are closed, and segments shorter than 200 ms are dropped.
   - In the word tasks a segment is about one spoken word (about 15 per participant); in the
     sentence task, a sentence or phrase.
   - The true boundaries are applied to both the decoded and the true spectrogram. This tests
     identification, not detection.
2. **Representation.**
   - Each Mel band is standardised with the mean and SD of the participant's *training*
     speech frames, so a segment is its deviation from that person's average speech
     spectrum.
   - Every segment is resampled to 32 frames, which removes duration as a cue.
3. **Similarity.** Pearson r between decoded segment i and true segment j over the 32 × 23
   matrix (`full`). Two components are computed the same way:
   - `loudness`: the frame means only;
   - `shape`: each frame minus its mean.
4. **Scores.**
   - **Pairwise accuracy (primary):** the share of pairs (i, j ≠ i) in which decoded segment i
     is more similar to its own true segment than to true segment j. Chance is exactly 0.5.
   - **Top-1 accuracy:** the share of decoded segments whose most similar true segment is
     their own. Chance is 1/N.
   - A per-prediction permutation p-value (1,000 random re-pairings) is reported as the share
     of runs that beat chance individually.
5. **References.**
   - The true spectrogram (accuracy 1 by construction).
   - The true spectrogram shifted by half the test partition: realistic speech in the wrong
     segments, an empirical check that the scores sit at 0.5 when nothing is decoded.

In the repeated-words task, the same word can occur more than once in a test partition.
Choosing another instance of the right word counts as an error, so the test is
conservative there.

## Pre-specified hypotheses (participant level, N = 30)

* **H18 (identification):** the pretrained GRU (`gru_ssl`, the best decoder on overall r)
  identifies the spoken segment above chance: pairwise accuracy > 0.5 at each of the five
  budgets. One-sided Wilcoxon signed-rank against 0.5, with Holm correction across budgets.
  - Reported alongside: the same test for every decoder, top-1 accuracy against 1/N, and the
    share of runs above chance individually.
* **H19 (pretraining improves identification):** `gru_ssl` > `gru` and `ssl` > `sup` in
  pairwise accuracy at 25%, 10% and 5% labels. Two-sided Wilcoxon, Holm across the three
  budgets, each architecture separately.
  - Reported alongside: self-pretraining against supervised training (`ssl_self` vs `sup`,
    `gru_ssl_self` vs `gru`); LP-FT against plain fine-tuning; own-patient SSL against the
    best BrainBERT readout (`gru_ssl` vs `bb_best_gru`, `probe_ssl` vs `bb_best_ridge`); and
    deep decoders against ridge.
* **E7 (descriptive): what carries identification.** Pairwise accuracy from loudness alone
  and from spectral shape alone, each tested against 0.5 as in H18, for every decoder and
  budget. The results are also reported by task group (words vs sentences).
* **Sensitivity:** 16 and 64 resampled frames instead of 32; stricter segments (gaps under
  100 ms closed, at least 300 ms long); every test repeated without P30.

## Decision for the paper

* **H18 holds:** the paper reports that the decoded spectrograms identify the spoken word
  well above chance, with only seconds to minutes of labels. This is the intelligibility-
  related result, stated with its chance level. **Fails:** reported in one sentence with the
  other evaluation results.
* **H19 holds:** pretraining improves word identification, a third co-measure beside
  overall and within-speech r. **Fails:** reported as is.
* **E7** states whether identification rests on loudness (syllable structure) or on spectral
  shape. It is reported whatever it shows, and it qualifies the claim made under H18.

## Budget

- One CPU job on the test partition, 8 participants at a time, about 15–25 minutes.
- Results go to `results/ident5`.
