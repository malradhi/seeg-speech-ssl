# Running the full study on a SLURM cluster

The study was run on the Komondor supercomputer (NVIDIA A100 GPUs). The job scripts in `slurm/` are written
for SLURM and should run on any SLURM cluster after two edits in `slurm/env.sh`:

* `IEEG_ROOT`: where the data and results go (default `$HOME/ieeg_jne`). About 10 GB of raw data, 5 GB of
  results for Phases 1–4 and up to 17 GB more during Phase 4b (BrainBERT features, which can be deleted
  after the Phase 4b bundle has been collected).
* `SBATCH_ACCOUNT`: your project account, if your cluster needs one.

The partition names used in the job scripts (`gpu`, `cpu`, `test`) are those of Komondor; change
`#SBATCH --partition=...` if yours are called differently. Every job can be resubmitted safely: finished
runs are skipped, so a job that hits its time limit is simply submitted again.

Run everything from the repository root, and run `source slurm/env.sh` in every new terminal.

---

## Step 0 — Set up (once)

```bash
git clone <this repository> && cd <repository folder>
mkdir -p logs
source slurm/env.sh
```

## Step 1 — Check the Python environment (login node, 1 min)

```bash
python scripts/00_check_env.py
```

It ends with `ENVIRONMENT OK` (CUDA is usually not available on the login node; that is expected).

## Step 2 — Download the dataset (login node, 10–30 min)

```bash
python scripts/01_download_osf.py --only P15        # one participant, enough for the smoke test
nohup python scripts/01_download_osf.py > logs/download.log 2>&1 &
tail -f logs/download.log                           # finished when it ends with "ALL OK: <N> files"
```

Re-running is safe: verified files are skipped and partial files resume. If the OSF listing fails, the whole
project can be downloaded as one archive and unpacked into `$IEEG_ROOT/raw`:

```bash
cd $IEEG_ROOT && wget -O ak3dp.zip "https://files.osf.io/v1/resources/ak3dp/providers/osfstorage/?zip="
mkdir -p raw && unzip ak3dp.zip -d raw && cd -
```

## Step 3 — Smoke test on a GPU (~10 min)

```bash
sbatch slurm/smoke_test.sbatch
tail -40 logs/smoke_*.out                           # must end with SMOKE TEST FINISHED
```

The numbers it prints are meaningless (tiny training budget); it only checks that everything runs on the GPU.

## Step 4 — Preprocess the 30 participants (~10 min)

```bash
PREP=$(sbatch --parsable slurm/preprocess_fast.sbatch)
squeue -u $USER                                     # wait until the job has finished
grep -c "saved" logs/prep_fast_$PREP.out            # 30
python scripts/03_audit.py                          # ends with the contact totals and AUDIT PASSED
```

Expected contact totals: unique words 1,103, repeated words 1,050, sentences 1,095; 3,248 in total.

## Step 5 — Phase 1: own-patient pretraining vs supervised and placebo ([design](docs/PHASE1_DESIGN.md))

```bash
sbatch slurm/phase1.sbatch                          # 30 GPU tasks, one per participant
grep -h "finished in" logs/p1_*.out | wc -l         # 30 when done
python scripts/21_quicklook.py                      # summary table (add --metric r_speech)
python scripts/20_collect.py                        # -> $IEEG_ROOT/bundle_phase1_<date>.zip
```

## Step 6 — Phase 2: other patients' data, masking ablation, contamination audit ([design](docs/PHASE2_DESIGN.md))

Phase 2 reuses the Phase 1 encoders, so `results/phase1` must stay in place.

```bash
sbatch slurm/smoke_phase2.sbatch                    # optional check, must end with SMOKE2 FINISHED
sbatch slurm/contamination.sbatch                   # CPU, ~15 min
MULTI=$(sbatch --parsable slurm/phase2_multi.sbatch)       # 35 GPU tasks: cross-patient pretraining
sbatch --dependency=afterok:$MULTI slurm/phase2.sbatch     # 30 GPU tasks, after all 35
python scripts/20_collect.py --phase phase2         # 33,845 run files, all ok
```

## Step 7 — Phase 3: pretrained GRU, large pooled model, evaluation suite ([design](docs/PHASE3_DESIGN.md))

```bash
sbatch slurm/smoke_phase3.sbatch                    # optional check, must end with SMOKE3 FINISHED
POOL=$(sbatch --parsable slurm/phase3_pooled.sbatch)                   # 3 GPU tasks: large pooled model
P3=$(sbatch --parsable --dependency=afterok:$POOL slurm/phase3.sbatch)  # 30 GPU tasks
sbatch --dependency=afterok:$P3 slurm/eval.sbatch                       # CPU evaluation suite
python scripts/22_eval_summary.py
python scripts/20_collect.py --phase phase3         # 18,963 run files, all ok
```

## Step 8 — Phase 4: BrainBERT, unlabelled-data dose, LP-FT ([design](docs/PHASE4_DESIGN.md))

```bash
python scripts/04_download_brainbert.py             # login node, ~175 MB once; ends with BRAINBERT OK
sbatch slurm/smoke_phase4.sbatch                    # optional check, must end with SMOKE4 FINISHED
P4=$(sbatch --parsable slurm/phase4.sbatch)                         # 30 GPU tasks
sbatch --dependency=afterok:$P4 slurm/eval4.sbatch
python scripts/22_eval_summary.py --dir eval4
python scripts/20_collect.py --phase phase4 --eval-dir eval4         # 21,270 run files, all ok
```

## Step 9 — Phase 4b: self-pretraining, seven learning rates, BrainBERT with 16–64 components ([design](docs/PHASE4B_DESIGN.md))

```bash
sbatch slurm/smoke_phase4b.sbatch                   # optional check, must end with SMOKE4B FINISHED
P4B=$(sbatch --parsable slurm/phase4b.sbatch)                       # 30 GPU tasks
sbatch --dependency=afterok:$P4B slurm/eval4b.sbatch
python scripts/22_eval_summary.py --dir eval4b
python scripts/20_collect.py --phase phase4b --eval-dir eval4b       # 22,110 run files, all ok
```

## Step 10 — Phase 5: identification from the saved predictions ([design](docs/PHASE5_DESIGN.md))

No training and no GPU; reads the saved test predictions of Phases 1–4b.

```bash
sbatch slurm/identify.sbatch                        # ends with "participants identified: 30 / 30"
python scripts/24_collect_phase5.py                 # -> bundle_phase5_<date>.zip (ident5/)
sbatch slurm/identify_words.sbatch                  # word-level rerun: pairs of different words
python scripts/24_collect_phase5.py --dir ident5w
```

## Step 11 — Statistics, figures and supplementary tables (any machine, no GPU)

Unzip each bundle into a folder named after its phase (`phase1_bundle/`, …, `phase4b_bundle/`,
`phase5_bundle/` for `ident5`, `phase5w_bundle/` for `ident5w`), all inside one directory, and run the
analyses as described in [`analysis/README.md`](analysis/README.md) and [`paper/README.md`](paper/README.md).
The released outputs in `results/` have exactly this layout.

---

### Useful SLURM commands

| What | Command |
|---|---|
| My jobs | `squeue -u $USER` |
| Cancel a job | `scancel <jobid>` |
| Resubmit some participants only | `sbatch --array=7,19 slurm/phase1.sbatch` |
| Job efficiency | `seff <jobid>` |
