# Generalisable and Actionable Student Risk Prediction

Code and data for the paper:

> **Generalisable and Actionable Student Risk Prediction: An Integrated Framework Combining Cross-Course Transfer, Explanation Stability, and Counterfactual Intervention Archetypes.**
> Musarat Karim, Mustafa Hameed (corresponding), Nadia Khan, Alisha Fida, Muhammad Nauman, Sumera Jalal.
> *VFAST Transactions on Software Engineering (VTSE)*, under review (revised September 2026).

The pipeline evaluates, on the UCI Student Performance benchmark:

1. **Cross-course transfer** (Portuguese → Mathematics) with pooling, warm-start CatBoost, TrAdaBoost.R2 (tree and CatBoost weak learners) and CORAL, plus an inner-fold negative-transfer gate, with explicit control of the students who appear in both course files.
2. **Cross-domain SHAP explanation stability** over a shared eight-concept space (UCI-Por, UCI-Math, xAPI-Edu), with exact permutation tests and bootstrap confidence intervals.
3. **Constrained DiCE counterfactual recourse**, clustered into intervention archetypes, with a five-seed robustness analysis.
4. Reverse-direction transfer, a grade-inclusive regime, calibration, effect sizes (paired Cliff's delta), and a sex / parental-education subgroup audit.

## What changed in the revision (September 2026)

The two UCI files describe many of the same students: 370 of the 395
Mathematics rows match a Portuguese row on the 13-attribute key that the dataset
authors use to link them (`data/student-merge.R`). The submitted version used
the full Portuguese file as the transfer source in every fold, so the test
students' own Portuguese records were in the training data. `src/train_v5_transfer.py`
now takes `--source-policy`:

| Policy | Source used in each fold |
|---|---|
| `full` | all 649 Portuguese rows (the submitted protocol; kept to show the size of the leak) |
| `fold_exclusive` | test-fold students removed from the source, also inside the gate's inner validation (**headline**) |
| `disjoint` | only the 275 Portuguese students with no Mathematics record |

Under `fold_exclusive` the gated model's RMSE gain over target-only is 1.8%
(3.990 → 3.920), not the 7.7% of the `full` protocol. With the `disjoint` source
only CORAL and pooling keep a significant gain. The `full` run reproduces the
submitted numbers exactly.

## Repository layout

```
data/                        UCI Student Performance dataset (see Data section)
src/
  data_loader.py             UCI dataset loader / preprocessing
  data_loader_xapi.py        xAPI-Edu (Kalboard 360) loader
  concept_mapping.py         UCI <-> xAPI shared concept-space crosswalk
  transfer.py                Warm-start CatBoost, TrAdaBoost.R2, CORAL, gate
  train_v4.py                Feature engineering (30 raw + 26 derived inputs)
  train_v5_transfer.py       Main orchestrator (transfer CV, xAPI probe, SHAP, DiCE)
  shap_stability.py          Cross-domain SHAP rank stability, exact permutation test
  counterfactuals.py         DiCE counterfactuals + archetype clustering
  cf_seed_sweep.py           Counterfactual archetypes over five DiCE seeds
  revision_extras.py         Overlap statistics, EDA statistics, Portuguese-only
                             stability check, counterfactual diagnostics
  make_revision_artifacts.py Cliff's delta, figures and the manuscript's number macros
  generate_eda_figures.py    Exploratory figures (and helpers used by revision_extras)
  g1g2_boost.py              Grade-inclusive ensemble (tuned on the evaluation data;
                             not used in the revised paper)
  compute_reverse_transfer.py, compute_cliffs_delta.py
                             Scripts of the submitted version (results_v6)
results/results_v7/          Outputs behind every table and statistic of the revised paper
results/results_v6/          Outputs of the submitted version (kept for provenance)
requirements.txt             Python dependencies
```

## Reproducing the results

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt

# Fast smoke run (validates the pipeline end to end)
python src/train_v5_transfer.py --smoke

# Transfer experiments of the revised paper (5x10 repeated stratified CV)
python src/train_v5_transfer.py --full --source-policy fold_exclusive --with-g1g2-comparison
python src/train_v5_transfer.py --full --source-policy fold_exclusive --reverse --skip-external
python src/train_v5_transfer.py --full --source-policy disjoint --with-g1g2-comparison --skip-external
python src/train_v5_transfer.py --full --source-policy full --with-g1g2-comparison --skip-external

# Counterfactual seed sweep and the remaining analyses
python src/cf_seed_sweep.py
python src/revision_extras.py

# Effect sizes, figures and the LaTeX macros used by the manuscript
python src/make_revision_artifacts.py      # writes to $PAPER_DIR or ./paper_outputs
```

Each full transfer run took 14-22 minutes on a 4-core laptop. The outputs used
in the revised manuscript are archived under `results/results_v7/`, so every
table and statistic can be checked without re-running the pipeline:
`make_revision_artifacts.py` regenerates the manuscript's numbers from that
folder alone.

## Data

**UCI Student Performance** (`data/student-mat.csv`, `data/student-por.csv`) is
redistributed here under the terms of the UCI Machine Learning Repository
(CC BY 4.0). Please cite:

> P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School
> Student Performance." In *Proceedings of 5th FUture BUsiness TEChnology
> Conference (FUBUTEC 2008)*, pp. 5-12, 2008.
> https://archive.ics.uci.edu/dataset/320/student+performance

**xAPI-Edu-Data (Kalboard 360)** is used only for the external-validation and
concept-space stability experiments. Its Kaggle distribution licence does not
permit redistribution here, so download `xAPI-Edu-Data.csv` from
https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data (also mirrored on IEEE
DataPort) and place it at `data/xAPI-Edu-Data.csv`, or point the `XAPI_CSV`
environment variable at it. All other experiments run without it.

## License

Source code is released under the MIT License (see `LICENSE`).
Dataset files in `data/` retain their original licences as described above.

## Contact

Corresponding author: Dr. Mustafa Hameed, Department of Information Technology,
The Islamia University of Bahawalpur, Pakistan.
