# Generalisable and Actionable Student Risk Prediction

Code and data for the paper:

> **Generalisable and Actionable Student Risk Prediction: An Integrated Framework Combining Cross-Course Transfer, Explanation Stability, and Counterfactual Intervention Archetypes.**
> Musarat Karim, Mustafa Hameed (corresponding), Nadia Khan, Alisha Fida, Muhammad Nauman, Sumera Jalal.
> *VFAST Transactions on Software Engineering (VTSE)*, under review.

The pipeline evaluates, on the UCI Student Performance benchmark:

1. **Cross-course transfer** (Portuguese → Mathematics) with warm-start CatBoost, TrAdaBoost.R2, and CORAL, guarded by an inner-fold negative-transfer gate.
2. **Cross-domain SHAP explanation stability** over a shared 8-feature concept space (UCI-Por, UCI-Math, xAPI-Edu), with bootstrap confidence intervals.
3. **Constrained DiCE counterfactual recourse**, clustered into intervention archetypes.
4. Reverse-direction transfer, calibration, effect sizes (Cliff's delta), and a gender / parental-education subgroup fairness audit.

## Repository layout

```
data/                    UCI Student Performance dataset (see Data section)
src/
  data_loader.py         UCI dataset loader / preprocessing
  data_loader_xapi.py    xAPI-Edu (Kalboard 360) loader
  concept_mapping.py     UCI <-> xAPI shared concept-space crosswalk
  transfer.py            Warm-start CatBoost, TrAdaBoost.R2, CORAL
  train_v4.py            Feature engineering + base-learner benchmark
  train_v5_transfer.py   Main orchestrator: the 10 experiments in the paper
  shap_stability.py      Cross-domain SHAP rank-stability analysis
  counterfactuals.py     DiCE counterfactuals + archetype clustering
  g1g2_boost.py          In-course-grade (G1/G2) ablation
  compute_reverse_transfer.py  Math -> Por reverse-direction experiment
  compute_cliffs_delta.py      Effect sizes, calibration, fairness audit
results/results_v6/      JSON/CSV outputs backing the tables and figures
requirements.txt         Python dependencies
```

## Reproducing the results

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt

# Fast smoke run (validates the full pipeline end-to-end)
python src/train_v5_transfer.py --smoke

# Full run used in the paper (10x5 repeated stratified CV, 20 Optuna trials)
python src/train_v5_transfer.py --full

# Supplementary experiments
python src/g1g2_boost.py
python src/compute_reverse_transfer.py
python src/compute_cliffs_delta.py
```

Outputs are written as JSON/CSV; the versions used in the submitted manuscript
are archived under `results/results_v6/` so tables and statistics can be
verified without re-running the (multi-hour) full pipeline.

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
