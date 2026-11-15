# Design documents

The study was run in six stages. Before each stage was run, its conditions, hypotheses, tests and decision
rules were written into a design document; the analysis scripts in `analysis/` implement these tests.

| Stage | Document | Written |
|---|---|---|
| Phase 1 | [PHASE1_DESIGN.md](PHASE1_DESIGN.md) | before any result |
| Phase 2 | [PHASE2_DESIGN.md](PHASE2_DESIGN.md) | after Phase 1, before any Phase 2 result |
| Phase 3 | [PHASE3_DESIGN.md](PHASE3_DESIGN.md) | after Phase 2, before any Phase 3 result |
| Phase 4 | [PHASE4_DESIGN.md](PHASE4_DESIGN.md) | after Phase 3, before any Phase 4 result |
| Phase 4b | [PHASE4B_DESIGN.md](PHASE4B_DESIGN.md) | after Phase 4, before any Phase 4b result |
| Phase 5 | [PHASE5_DESIGN.md](PHASE5_DESIGN.md) | after Phase 4b, before any Phase 5 result |

From Phase 2 on, each document starts with the reasons for the stage, which quote interim results of the earlier
stages as they stood when the document was written (for example, on the learning-rate grid of that time). The final values,
after all stages and on the seven-rate grid, are those reported in the article and in `results/stats/`.
Analyses added after a stage's results were seen are labelled as post hoc or descriptive, in the documents
(for example, E6 in Phase 4b) or in the analysis scripts.

[RUNNING_THE_STUDY.md](RUNNING_THE_STUDY.md) gives every step for running the study on a SLURM cluster.
`study_design.png` is Figure 1 of the article.
