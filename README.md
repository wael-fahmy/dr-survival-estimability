# Design and outcome-model estimability in doubly robust survival analysis

Code and data for the paper *Design and outcome-model estimability in doubly robust survival analysis: a breast cancer reanalysis* by Wael Fahmy, Elena Krikun and Abedalrhman Alkhateeb (Department of Computer Science, Lakehead University).

**Preprint:** link to be added. **Archived release:** DOI to be added.

The paper reanalyzes chemotherapy and survival in TCGA breast cancer (TCGA-BRCA). It audits the observed-data design (treatment coding, fixed-horizon outcomes under censoring, receptor imputation), then estimates adjusted restricted mean survival time (RMST) contrasts over 1095 days after a day-90 landmark with weighting-only, Cox outcome-model and cross-fitted doubly robust (AIPW) estimators. It introduces a parameter-to-information ratio that summarizes how much a complete-outcome regression is asked to estimate from the available events, with a simulation-calibrated warning level.

This repository reproduces every number, table and figure in the article and its supplement from the supplied clinical exports and derived tables.

## Quick start

Python 3.11 is required. The versions in `requirements.txt` are the ones used for the paper.

```bash
python -m pip install -r requirements.txt

python run_all.py --verify                  # checks against the committed results (seconds)
python run_all.py --render-only             # rebuild figures and LaTeX from the committed outputs (seconds)
python run_all.py --render-only --compile   # as above, then build the PDFs
python run_all.py                           # rerun every numerical stage (several hours)
```

- `--verify` runs the unit tests of the Cox RMST benchmark against lifelines, recomputes the 12 primary-horizon point estimates (two populations, three adjustment sets, two estimators) and requires them to match the committed values to within 1e-8, and checks the integrity of the comparison and stress-study outputs.
- `--render-only` regenerates `paper/sn-article.tex`, `paper/supplement.tex` and the figures from the files in `output/`. It does not recompute any estimate.
- A full run executes every numerical stage in order and stops at the first failure. It overwrites the committed files in `output/`; use `git diff` or `git restore output/` to compare with or return to the published results.
- `--compile` needs `pdflatex` (TeX Live or MiKTeX) on `PATH`. Both documents use the standard `article` class, standard packages and Latin Modern fonts, so they also compile on Overleaf: upload `paper/` and compile `sn-article.tex` and `supplement.tex` separately.

## Repository layout

| Path | Contents |
|---|---|
| `run_all.py` | Entry point. Runs the stages in dependency order. |
| `lib/` | Estimators (`lib_est.py`), input handling (`lib_io.py`) and plotting helpers (`lib_plot.py`). |
| `scripts/` | Numbered pipeline stages (below). |
| `tests/` | Reference and invariance tests for the Cox RMST benchmark. |
| `data/` | Input data. Digests and provenance in `data/SOURCES.md`. |
| `output/` | Every computed table, the numerical record behind the manuscript, and figure data in `output/figdata/`. |
| `figures/make_figures.py` | Draws the six original article figures from `output/figdata/`. |
| `paper/` | Generated article and supplement (`.tex`), the narrative template, figure PDFs and figure alt text. |
| `submission_metadata.json` | Funding and related-work text inserted into the manuscript. |

## Pipeline stages

Stage numbers are shared with a larger project workspace, so there are gaps in the sequence.

| Stage | Purpose |
|---|---|
| 00 | Known-truth estimator validation (500 replicates, 500 inner bootstraps). |
| 01 to 06 | Receptor status, TNBC definition, survival reconciliation, chemotherapy exposure, day-90 landmark cohort and site audits. |
| 07 | Adjusted RMST estimates with bootstrap intervals, sensitivity analyses and diagnostics. |
| 08 | Correction ladder: RMST at each step from the naive to the corrected analysis. |
| 10, 17 | Figure data and the six original vector figures. |
| 15 | Independent lifelines cross-checks of the numerical components. |
| 16 | Calibration of the parameter-to-information ratio (32 cells, 60 replicates per cell; penalty scales 0.3 and 3 use 30 each). |
| 18 | Primary estimates and calibration across six seeds (Monte Carlo error). |
| 20 | Same-horizon Kaplan-Meier and Cox benchmarks, balance, proportional-hazards screens, horizon support and stabilizer sensitivity. |
| 21 | Stress study: six cells, 200 replicates per cell, 199 inner bootstraps, four methods. |
| 23, 22 | Workflow and simulation figures, then the article and supplement. Every number in the text is substituted from `output/`; the full substitution record is `output/22_manuscript_values.json`. |
| 24 | PDF compilation and layout checks (called by `--compile`). |
| 25 | Verification of the committed results (called by `--verify`). |

The narrative text of the article lives in `paper/templates/biometrics-article.tex.in`. Edit the template or `scripts/22_write_biometrics.py`, not the generated `.tex` files, which are overwritten on every render.

## Data

| File | Source |
|---|---|
| `clinical.tsv`, `follow_up.tsv`, `pathology_detail.tsv` | Open-access clinical exports from the [NCI Genomic Data Commons, TCGA-BRCA project](https://portal.gdc.cancer.gov/projects/TCGA-BRCA). |
| `outcome.csv` | Derived survival table, reconciled against `clinical.tsv` by stage 03. |
| `brca_pam50.txt` | PAM50 subtype calls. |
| `causal_features_dataset.csv`, `ite_ready_dataset_v2.csv` | Derived tables from an earlier multimodal feature pipeline. The earlier adjustment set is read from `ite_ready_dataset_v2.csv`. |

All data are public and de-identified. The earlier omics feature-selection pipeline is **not** reconstructed here: the original [UCSC Xena](https://xenabrowser.net) exports and raw assay matrices are not included, and the earlier feature set and clinical preprocessing are held fixed rather than relearned inside each bootstrap replicate.

SHA-256 digests of every input are listed in `data/SOURCES.md` and `output/input_sha256.json`. `.gitattributes` disables line-ending conversion so that a checkout on any platform matches these digests.

## Reproducibility notes

- The default analysis seed is 20260802 (override with the `REVISION_SEED` environment variable). Scripts record their seed offsets, fold construction, replication counts and bootstrap rules.
- Stages 20 and 21 accept `--smoke` for a reduced run that writes to `output/smoke_biometrics/` and never touches the committed outputs. The stress design file `output/21_stress_design.json` records the calibrated baseline hazards and a source hash; stage 21 resumes only from checkpoints written by matching code.
- Some committed outputs come from the previous complete pipeline run and were not all regenerated for this version: the 2000-replicate bootstrap intervals for the adjusted estimates, the 500 by 500 estimator validation, the 32-cell calibration grid and the six-seed repeat. A full run regenerates them.
- Numerical agreement is judged by tolerances and fixed seeds, not by identical PDFs. PDF metadata, font versions and BLAS builds differ between installations. For example, rendering rewrites `paper/figures/fig7_workflow.pdf` and `fig8_calibration_stress.pdf` with a new embedded creation date, so Git lists them as modified although the drawings are unchanged. The generated `.tex` files are reproduced byte for byte.
- One Cox method/replicate evaluation in the stress study fails the evaluability criterion. Coverage is reported conditional on successful evaluations, and failures are counted in `output/21_stress_summary.csv`.
- A few files in `output/` are kept for provenance and superseded in the paper: `07_nuisance_budget.csv` holds historical reference degrees of freedom (the article uses the fold-specific values in `20_outcome_fold_df.csv`); `07_stabiliser_sensitivity.csv` is superseded by the five-device assessment in `20_stabilizers.csv`; `07_heterogeneity.csv` keeps exploratory fixed-score intervals, of which the supplement reports only the descriptive point summaries; `07_mde.csv` is a rough complete-data approximation, not weighted-survival power.
- On multipage tables, the November 2025 LaTeX release reports an infinite glue shrinkage warning (latex3/latex2e issue 1907). It is harmless and fixed in the June 2026 release.

## Scope and interpretation

The target is a standardized statistical contrast among patients alive and uncensored at day 90 (prevalent users versus non-initiators). A causal reading needs further assumptions. This is not a target-trial emulation, a classification or clustering study, or a new gene signature. The AIPW estimator used here requires a correct censoring model plus a correct propensity or outcome model; it is not fully augmented for censoring. The parameter-to-information ratio is a warning level for the studied regime, not a certificate of estimator validity.

## License

The code is released under the MIT License (see `LICENSE`). The TCGA data are distributed by the NCI Genomic Data Commons under its open-access data use terms; please cite TCGA when reusing them.

## Citation

Please cite the paper. The preprint reference will be added here and to `CITATION.cff` once it is posted.

## Contact

Wael Fahmy, wfahmy@lakeheadu.ca
