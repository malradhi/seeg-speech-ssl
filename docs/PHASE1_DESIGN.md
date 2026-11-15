# Phase 1 — the decisive test (fixed before seeing any results)

## Question

Does masked self-supervised pretraining improve participant-specific sEEG-to-Mel
decoding when labelled speech is scarce, **after** removing the confounds of the
authors' earlier pilot version (unequal optimisation budget, no input normalisation, untuned
baseline), and is any gain due to learned neural structure rather than
optimisation warm-up?

## Data

30 participants (Verwoert et al., Cell Reports 2025; OSF `ak3dp`):
P01–P10 unique words, P11–P20 repeated words, P21–P30 sentences.
Only localised intracranial contacts (3,248 in total; marker, ECG, pulse-oximetry and
unlocalised channels excluded). Chronological 70/15/15 split, boundaries moved into inter-trial silences,
features extracted **within** each partition, 0.5 s trimmed at every partition edge.

## Conditions (per participant × label fraction × seed)

| Condition | Encoder init | Protocol | Purpose |
|---|---|---|---|
| `sup` | random | new: z-scored, AdamW, 3000-step budget, early stopping, LR ∈ {3e-5, 1e-4, 3e-4, 1e-3} picked on validation, validation every 25 steps | fair supervised baseline |
| `ssl` | masked reconstruction (random-coordinate, ρ = 0.5) on all training neural data | new | the method |
| `placebo` | same pretraining on column-permuted neural data (same scale, no structure) | new | is the gain neural structure or warm-up/scaling? |
| `sup_legacy`, `ssl_legacy` | as in the pilot study | legacy: raw features, Adam, 40 epochs (30 pretraining) | does the pilot-study effect reproduce on 30 participants? |
| `ridge` | — | closed form, α on validation | strong linear baseline |
| `pca_lab` | — | PCA(50) on labelled frames + OLS | the pilot study's linear reference |
| `pca_all` | — | PCA(50) on all unlabelled training frames + OLS | linear analogue of unsupervised pretraining |
| `vad_oracle` | — | mean speech / silence spectrum at the true speech times | what r is reachable from speech timing alone |

Label fractions: 100, 50, 25, 10, 5 % of training windows (nested subsets; identical
for every condition). Seeds 1–5. Labelled amounts are reported in seconds.

## Metrics

* `r_mean` — mean per-band Pearson r over all test frames (continuity with the literature).
* `r_speech` — the same, restricted to speech frames: spectral detail beyond speech timing.
* `mcd_db`, `mcd_speech_db` — mel-cepstral distortion, (10/ln10)·sqrt(2·Σ Δc_d²) over c1..c22, with
  c = real-cepstrum-scaled DCT-II (1/2K) of the natural-log 23-band Mel amplitude spectrum. Waveform-based
  MCD/STOI after resynthesis follow in Phase 2.
* Chance: circular-shift null of the predictions (shifts ≥ 1 s), 95th percentile.

## Pre-specified analysis and decision rule

Unit of inference: participant (N = 30; seeds averaged within participant). LR for
the new-protocol conditions chosen per run on validation MSE.

1. **H1 (benefit)** — `ssl` > `sup` at 25 %, 10 % and 5 %: paired Wilcoxon, Holm
   across the five fractions; Page trend test for an advantage that grows as labels decrease.
2. **H2 (mechanism)** — `ssl` > `placebo` at the same fractions.
3. **H3 (continuity)** — `ssl_legacy` vs `sup_legacy` (does the pilot-study pattern reproduce?).
4. **H4 (practical value)** — `ssl` vs `ridge` and `pca_all`.
5. Consistency across the three task cohorts (descriptive; mixed model with task as a fixed effect).

Co-primary metrics: `r_mean` and `r_speech`.

**GO** (continue with the SSL paper as planned): H1 and H2 hold at ≥ 1 low-label
fraction on at least one co-primary metric, with the same direction in all three cohorts.
**REFRAME** otherwise: the paper becomes a rigorous benchmark of what does and does
not help label-efficient sEEG speech decoding (still publishable, different framing).
