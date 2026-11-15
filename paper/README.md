# Figures and tables of the article

Every figure and table of the article is generated from the released analysis outputs in `results/`
(see `results/README.md`). Unpack them first, from the repository root:

```bash
unzip results/analysis_outputs.zip -d results/
```

| Output | Script | Command (from the folder of the script) |
|---|---|---|
| Figure 1 (study design) | `figures/make_fig1.py` | `python make_fig1.py out` |
| Figures 2–6 | `figures/make_figures.py` | `python make_figures.py ../../results out` |
| Table 2 (key comparisons) | `tables/make_table2.py` | `python make_table2.py ../../results/stats` |
| Supplementary tables S3–S22 | `supplement/make_supp_tables.py` | `sh build.sh` |

* `figures/figdata.py` builds the participant-level data exactly as in the pre-specified analyses (learning
  rate chosen per participant, condition, budget and seed on validation; participant score = mean over seeds)
  and is shared by the figure and table scripts.
* Figures are written as PDF (vector) and PNG; the supplementary tables as LaTeX (`supp_tables.tex`, and one
  file per table in `supplement/tables/`).
* Differences in the tables are given in hundredths of r (difference × 100), as in Table 2 of the article.
* Recordings and spectrograms shown in Figure 1 are illustrative.
