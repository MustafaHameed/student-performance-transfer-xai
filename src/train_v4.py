"""
Training Pipeline v4 - Extended Model Comparison with Statistical Validation

Key improvements over v3:
1. CatBoost (Optuna-tuned) + SVR-RBF (Optuna-tuned) added as new baselines
2. pytorch-tabnet library replaces custom LiteTabNet (reference implementation)
3. RepeatedStratifiedKFold (10 repeats × 5 folds = 50 total) replaces single 5-fold
4. 95% confidence intervals via t-distribution (df=49)
5. Friedman test + pairwise Wilcoxon tests with Bonferroni correction
6. Classification task: G3 >= 10 -> pass/fail (Accuracy, F1-macro, AUC-ROC)
7. Feature engineering ablation: raw 30 features vs engineered 56 features (XGBoost)
8. SHAP mean |SHAP| values for top-3 models (XGBoost, LightGBM, CatBoost)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, asdict

# Force UTF-8 stdout to avoid Windows cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.optim as optim
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import t as t_dist

from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, f1_score, roc_auc_score)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import StudentPerformanceDataset

# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_RESULTS_ROOT = './results_v5'
os.makedirs(DEFAULT_RESULTS_ROOT, exist_ok=True)

TE_COLS            = ['Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian']
PASS_THRESHOLD     = 10       # G3 >= 10 → pass
N_OPTUNA_TRIALS    = 20       # per model
N_TABNET_TRIALS    = N_OPTUNA_TRIALS  # keep search budget aligned for comparison
N_CV_SPLITS        = 5
N_CV_REPEATS       = 10       # 10 × 5 = 50 folds
TABNET_MAX_EPOCHS  = 100
TABNET_PATIENCE    = 15

MODELS = [
    'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest+',
    'StackingEnsemble', 'GradientBoosting+', 'ElasticNet', 'SVR-RBF', 'TabNet',
]

TREE_MODELS = {
    'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest+',
    'StackingEnsemble', 'GradientBoosting+',
}

SCALED_MODELS = {
    'ElasticNet', 'SVR-RBF', 'TabNet',
}


@dataclass(frozen=True)
class BenchmarkTrack:
    name: str
    include_prior_grades: bool
    description: str


TRACKS = {
    'early_warning': BenchmarkTrack(
        name='early_warning',
        include_prior_grades=False,
        description='Predict final grade without G1/G2 for early intervention.'
    ),
    'grade_inclusive': BenchmarkTrack(
        name='grade_inclusive',
        include_prior_grades=True,
        description='Predict final grade with G1/G2 as an upper-bound benchmark.'
    ),
}


# ══════════════════════════════════════════════
# Feature Engineering  (identical to v3 — per-fold fit/transform)
# ══════════════════════════════════════════════

class FeatureEngineer:
    """v3 feature engineer: fit/transform separated to prevent leakage."""

    def __init__(self, scale_engineered: bool = True):
        self.scale_engineered = scale_engineered
        self.scaler          = StandardScaler()
        self.target_encoders = {}
        self.te_global_means = {}
        self.feature_names_out: list = []

    def _raw_features(self, X: np.ndarray, feature_names: list,
                      y_train: np.ndarray = None) -> dict:
        cm = {n: i for i, n in enumerate(feature_names)}
        g  = lambda name: X[:, cm[name]] if name in cm else None
        f  = {}

        medu, fedu = g('Medu'), g('Fedu')
        if medu is not None and fedu is not None:
            f['par_edu_sum']  = medu + fedu
            f['par_edu_max']  = np.maximum(medu, fedu)
            f['par_edu_diff'] = np.abs(medu - fedu)
            f['par_edu_prod'] = medu * fedu

        study, fail = g('studytime'), g('failures')
        if study is not None and fail is not None:
            f['study_fail_ratio']    = study / (fail + 1)
            f['study_fail_interact'] = study * (1 - fail / 4)
            f['study_x_fail']        = study * (fail + 1)

        dalc, walc = g('Dalc'), g('Walc')
        if dalc is not None and walc is not None:
            f['alcohol_total']         = dalc + walc
            f['alcohol_weekday_ratio'] = dalc / (walc + 0.1)

        ssup, fsup = g('schoolsup'), g('famsup')
        if ssup is not None and fsup is not None:
            f['total_support'] = ssup + fsup

        goout, free = g('goout'), g('freetime')
        if goout is not None and free is not None:
            f['social_index'] = goout + free

        absences = g('absences')
        if absences is not None:
            f['absences_log']  = np.log1p(absences)
            f['absences_high'] = (absences > 10).astype(float)

        if fail is not None:
            f['has_failures']      = (fail > 0).astype(float)
            f['multiple_failures'] = (fail > 1).astype(float)

        age = g('age')
        if age is not None:
            f['age_sq']        = age ** 2
            f['older_student'] = (age > 18).astype(float)

        if fail is not None and absences is not None:
            f['fail_x_absences'] = fail * np.log1p(absences)

        if dalc is not None and walc is not None and absences is not None:
            f['alc_x_absences'] = (dalc + walc) * np.log1p(absences)

        if ssup is not None and fail is not None:
            f['schoolsup_x_fail'] = ssup * fail

        if y_train is not None:
            for col in TE_COLS:
                if col in cm:
                    vals  = X[:, cm[col]]
                    grand = float(np.mean(y_train))
                    means = {float(v): float(np.mean(y_train[vals == v]))
                             for v in np.unique(vals)}
                    self.target_encoders[col] = means
                    self.te_global_means[col]  = grand
                    f[f'{col}_te'] = np.array(
                        [means.get(float(v), grand) for v in vals])
        elif self.target_encoders:
            for col, means in self.target_encoders.items():
                if col in cm:
                    vals = X[:, cm[col]]
                    gm   = self.te_global_means.get(col, 10.0)
                    f[f'{col}_te'] = np.array(
                        [means.get(float(v), gm) for v in vals])
        return f

    def fit_transform(self, X: np.ndarray, y_train: np.ndarray,
                      feature_names: list) -> np.ndarray:
        raw = self._raw_features(X, feature_names, y_train=y_train)
        if raw:
            arr    = np.column_stack(list(raw.values()))
            if self.scale_engineered:
                arr_out = self.scaler.fit_transform(arr)
            else:
                arr_out = arr.astype(np.float32)
            X_out  = np.hstack([X, arr_out]).astype(np.float32)
            self.feature_names_out = list(feature_names) + list(raw.keys())
        else:
            X_out = X.astype(np.float32)
            self.feature_names_out = list(feature_names)
        return X_out

    def transform(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        raw = self._raw_features(X, feature_names, y_train=None)
        if raw:
            arr    = np.column_stack(list(raw.values()))
            if self.scale_engineered:
                arr_out = self.scaler.transform(arr)
            else:
                arr_out = arr.astype(np.float32)
            X_out  = np.hstack([X, arr_out]).astype(np.float32)
        else:
            X_out = X.astype(np.float32)
        return X_out


def parse_args():
    parser = argparse.ArgumentParser(
        description='Student performance benchmark runner with explicit track selection.'
    )
    parser.add_argument(
        '--tracks', nargs='+', choices=sorted(TRACKS.keys()),
        default=['early_warning', 'grade_inclusive'],
        help='Benchmark tracks to run. Default runs both early_warning and grade_inclusive.'
    )
    parser.add_argument(
        '--subject-subset', choices=['portuguese', 'math', 'both'],
        default='portuguese',
        help='Dataset subject subset to evaluate.'
    )
    parser.add_argument(
        '--results-root', default=DEFAULT_RESULTS_ROOT,
        help='Root directory for benchmark outputs.'
    )
    parser.add_argument(
        '--refresh-data', action='store_true',
        help='Refresh cached source data before preprocessing.'
    )
    return parser.parse_args()


def get_output_dir(results_root: str, subject_subset: str,
                   track: BenchmarkTrack) -> str:
    output_dir = os.path.join(results_root, f'{subject_subset}_{track.name}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prepare_benchmark_data(dataset: StudentPerformanceDataset,
                           track: BenchmarkTrack,
                           subject_subset: str = 'portuguese',
                           refresh_data: bool = False):
    dataset.download_and_load(subject_subset=subject_subset, refresh=refresh_data)

    X_tree, y = dataset.preprocess(
        target='G3',
        include_prior_grades=track.include_prior_grades,
        scale_numeric=False,
    )
    feature_names = list(dataset.feature_names)

    X_scaled, y_scaled = dataset.preprocess(
        target='G3',
        include_prior_grades=track.include_prior_grades,
        scale_numeric=True,
    )

    if not np.allclose(y, y_scaled):
        raise ValueError('Scaled and unscaled benchmark targets diverged unexpectedly.')

    return {
        'tree': X_tree,
        'scaled': X_scaled,
        'y': y,
        'feature_names': feature_names,
    }


# ══════════════════════════════════════════════
# Optuna Hyperparameter Tuning
# ══════════════════════════════════════════════

def _stratified_bins(y: np.ndarray) -> np.ndarray:
    return pd.qcut(y, q=4, labels=False, duplicates='drop')


def _cv_rmse(model, X_fe: np.ndarray, y: np.ndarray) -> float:
    """5-fold stratified CV RMSE helper used inside Optuna objectives."""
    bins = _stratified_bins(y)
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X_fe, bins):
        model.fit(X_fe[tr], y[tr])
        p = model.predict(X_fe[va])
        scores.append(np.sqrt(mean_squared_error(y[va], p)))
    return float(np.mean(scores))


def tune_rf(X_fe: np.ndarray, y: np.ndarray,
            n_trials: int = N_OPTUNA_TRIALS) -> dict:
    def objective(trial):
        m = RandomForestRegressor(
            n_estimators      = trial.suggest_int('n_estimators', 50, 300),
            max_depth         = trial.suggest_int('max_depth', 5, 20),
            min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 8),
            min_samples_split = trial.suggest_int('min_samples_split', 2, 15),
            max_features      = trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', 0.4, 0.6, 0.8]),
            random_state=SEED, n_jobs=-1,
        )
        return _cv_rmse(m, X_fe, y)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'random_state': SEED, 'n_jobs': -1}


def tune_xgb(X_fe: np.ndarray, y: np.ndarray,
             n_trials: int = N_OPTUNA_TRIALS) -> dict:
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 50, 300),
            max_depth        = trial.suggest_int('max_depth', 3, 8),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample        = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 2.0),
            reg_lambda       = trial.suggest_float('reg_lambda', 0.0, 5.0),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 8),
            gamma            = trial.suggest_float('gamma', 0.0, 1.0),
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
        bins = _stratified_bins(y)
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for tr, va in skf.split(X_fe, bins):
            m = xgb.XGBRegressor(**params)
            m.fit(X_fe[tr], y[tr], eval_set=[(X_fe[va], y[va])], verbose=False)
            scores.append(np.sqrt(mean_squared_error(y[va], m.predict(X_fe[va]))))
        return float(np.mean(scores))
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'random_state': SEED, 'n_jobs': -1, 'verbosity': 0}


def tune_lgb(X_fe: np.ndarray, y: np.ndarray,
             n_trials: int = N_OPTUNA_TRIALS) -> dict:
    def objective(trial):
        m = lgb.LGBMRegressor(
            n_estimators      = trial.suggest_int('n_estimators', 50, 300),
            max_depth         = trial.suggest_int('max_depth', 3, 8),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 2.0),
            reg_lambda        = trial.suggest_float('reg_lambda', 0.0, 5.0),
            num_leaves        = trial.suggest_int('num_leaves', 15, 63),
            min_child_samples = trial.suggest_int('min_child_samples', 5, 30),
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        return _cv_rmse(m, X_fe, y)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'random_state': SEED, 'n_jobs': -1, 'verbose': -1}


def tune_catboost(X_fe: np.ndarray, y: np.ndarray,
                  n_trials: int = N_OPTUNA_TRIALS) -> dict:
    def objective(trial):
        m = CatBoostRegressor(
            iterations    = trial.suggest_int('iterations', 100, 600),
            depth         = trial.suggest_int('depth', 4, 10),
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            l2_leaf_reg   = trial.suggest_float('l2_leaf_reg', 0.5, 10.0),
            random_seed   = SEED,
            verbose       = False,
            allow_writing_files = False,
        )
        return _cv_rmse(m, X_fe, y)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({'random_seed': SEED, 'verbose': False, 'allow_writing_files': False})
    return best


def tune_svr(X_fe: np.ndarray, y: np.ndarray,
             n_trials: int = N_OPTUNA_TRIALS) -> dict:
    def objective(trial):
        svr = SVR(
            kernel  = 'rbf',
            C       = trial.suggest_float('C', 0.1, 100.0, log=True),
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0),
            gamma   = trial.suggest_categorical('gamma', ['scale', 'auto']),
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('svr', svr)])
        bins = _stratified_bins(y)
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for tr, va in skf.split(X_fe, bins):
            pipe.fit(X_fe[tr], y[tr])
            scores.append(np.sqrt(mean_squared_error(y[va], pipe.predict(X_fe[va]))))
        return float(np.mean(scores))
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'kernel': 'rbf'}


def tune_tabnet(X_fe: np.ndarray, y: np.ndarray,
                n_trials: int = N_TABNET_TRIALS) -> dict:
    """Tune TabNet on a stratified 80/20 split to limit cost while preserving grade balance."""
    from sklearn.model_selection import train_test_split
    bins = _stratified_bins(y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_fe, y, test_size=0.2, random_state=SEED, stratify=bins)

    def objective(trial):
        n_d = trial.suggest_categorical('n_d', [16, 32, 64])
        try:
            m = TabNetRegressor(
                n_d            = n_d,
                n_a            = n_d,
                n_steps        = trial.suggest_int('n_steps', 3, 6),
                gamma          = trial.suggest_float('gamma', 1.0, 2.0),
                lambda_sparse  = trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True),
                optimizer_fn   = torch.optim.Adam,
                optimizer_params = {'lr': trial.suggest_float('lr', 5e-3, 5e-2, log=True)},
                mask_type      = 'sparsemax',
                seed           = SEED,
                verbose        = 0,
            )
            m.fit(
                X_tr.astype(np.float32),
                y_tr.astype(np.float32).reshape(-1, 1),
                eval_set  = [(X_va.astype(np.float32),
                              y_va.astype(np.float32).reshape(-1, 1))],
                eval_name = ['val'],
                eval_metric = ['rmse'],
                max_epochs  = TABNET_MAX_EPOCHS,
                patience    = TABNET_PATIENCE,
                batch_size  = 128,
                virtual_batch_size = 32,
            )
            preds = m.predict(X_va.astype(np.float32)).flatten()
            return float(np.sqrt(mean_squared_error(y_va, preds)))
        except Exception:
            return 999.0

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ══════════════════════════════════════════════
# Ensemble Builder
# ══════════════════════════════════════════════

def build_ensemble(X_tr: np.ndarray, y_tr: np.ndarray,
                   rf_params: dict, xgb_params: dict,
                   lgb_params: dict) -> StackingRegressor:
    """RF + XGB + LGB + scaled ElasticNet base → ElasticNetCV meta (cv=3)."""
    base = [
        ('rf',      RandomForestRegressor(**rf_params)),
        ('xgb',     xgb.XGBRegressor(**xgb_params)),
        ('lgb',     lgb.LGBMRegressor(**lgb_params)),
        ('elastic', Pipeline([
            ('scaler', StandardScaler()),
            ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED)),
        ])),
    ]
    meta  = ElasticNetCV(cv=3, random_state=SEED, max_iter=2000)
    stack = StackingRegressor(estimators=base, final_estimator=meta,
                              cv=3, n_jobs=1)
    stack.fit(X_tr, y_tr)
    return stack


# ══════════════════════════════════════════════
# Cross-Validation  (10 × 5 = 50 folds)
# ══════════════════════════════════════════════

def cross_validate_v4(X_tree: np.ndarray, X_scaled: np.ndarray,
                      y: np.ndarray, feature_names: list,
                      rf_params: dict, xgb_params: dict, lgb_params: dict,
                      catboost_params: dict, svr_params: dict,
                      tabnet_params: dict,
                      n_splits: int = N_CV_SPLITS,
                      n_repeats: int = N_CV_REPEATS) -> dict:

    bins = _stratified_bins(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=SEED)
    total = n_splits * n_repeats

    # Per-fold storage: regression + classification metrics
    fold_data = {m: {'rmse': [], 'mae': [], 'r2': [],
                     'acc': [], 'f1': [], 'auc': []}
                 for m in MODELS}

    print(f"\n{'='*60}")
    print(f"Cross-Validation v4  ({n_repeats}×{n_splits} = {total} folds)")
    print(f"{'='*60}")

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_tree, bins)):
        rep  = fold_idx // n_splits + 1
        fold = fold_idx %  n_splits + 1
        if fold == 1:
            print(f"\nRepeat {rep}/{n_repeats}", flush=True)

        X_tr_tree, X_va_tree = X_tree[tr_idx], X_tree[va_idx]
        X_tr_scaled, X_va_scaled = X_scaled[tr_idx], X_scaled[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Feature engineering — fit strictly on training fold
        fe_tree = FeatureEngineer(scale_engineered=False)
        X_tr_tree_fe = fe_tree.fit_transform(X_tr_tree, y_tr, feature_names)
        X_va_tree_fe = fe_tree.transform(X_va_tree, feature_names)

        fe_scaled = FeatureEngineer(scale_engineered=True)
        X_tr_scaled_fe = fe_scaled.fit_transform(X_tr_scaled, y_tr, feature_names)
        X_va_scaled_fe = fe_scaled.transform(X_va_scaled, feature_names)

        y_true_bin = (y_va >= PASS_THRESHOLD).astype(int)

        def record(name: str, y_pred: np.ndarray):
            d = fold_data[name]
            d['rmse'].append(float(np.sqrt(mean_squared_error(y_va, y_pred))))
            d['mae'].append(float(mean_absolute_error(y_va, y_pred)))
            d['r2'].append(float(r2_score(y_va, y_pred)))
            y_pred_bin = (y_pred >= PASS_THRESHOLD).astype(int)
            d['acc'].append(float(accuracy_score(y_true_bin, y_pred_bin)))
            d['f1'].append(float(f1_score(y_true_bin, y_pred_bin,
                                          average='macro', zero_division=0)))
            try:
                d['auc'].append(float(roc_auc_score(y_true_bin, y_pred)))
            except ValueError:
                d['auc'].append(0.5)

        # ── XGBoost ──────────────────────────────────────────────────
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr_tree_fe, y_tr, eval_set=[(X_va_tree_fe, y_va)], verbose=False)
        record('XGBoost', m.predict(X_va_tree_fe))

        # ── LightGBM ─────────────────────────────────────────────────
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X_tr_tree_fe, y_tr)
        record('LightGBM', m.predict(X_va_tree_fe))

        # ── CatBoost ─────────────────────────────────────────────────
        m = CatBoostRegressor(**catboost_params)
        m.fit(X_tr_tree_fe, y_tr)
        record('CatBoost', m.predict(X_va_tree_fe))

        # ── Random Forest (tuned) ────────────────────────────────────
        m = RandomForestRegressor(**rf_params)
        m.fit(X_tr_tree_fe, y_tr)
        record('RandomForest+', m.predict(X_va_tree_fe))

        # ── Stacking Ensemble ────────────────────────────────────────
        stack = build_ensemble(X_tr_tree_fe, y_tr, rf_params, xgb_params, lgb_params)
        record('StackingEnsemble', stack.predict(X_va_tree_fe))

        # ── Gradient Boosting (fixed params, comparison baseline) ────
        m = GradientBoostingRegressor(n_estimators=200, max_depth=6,
                                       learning_rate=0.05, subsample=0.8,
                                       min_samples_leaf=3, random_state=SEED)
        m.fit(X_tr_tree_fe, y_tr)
        record('GradientBoosting+', m.predict(X_va_tree_fe))

        # ── ElasticNet ───────────────────────────────────────────────
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=2000)
        m.fit(X_tr_scaled_fe, y_tr)
        record('ElasticNet', m.predict(X_va_scaled_fe))

        # ── SVR-RBF ──────────────────────────────────────────────────
        pipe = Pipeline([('scaler', StandardScaler()),
                          ('svr', SVR(**svr_params))])
        pipe.fit(X_tr_scaled_fe, y_tr)
        record('SVR-RBF', pipe.predict(X_va_scaled_fe))

        # ── TabNet (pytorch-tabnet) ───────────────────────────────────
        tn = TabNetRegressor(
            n_d             = tabnet_params['n_d'],
            n_a             = tabnet_params['n_d'],
            n_steps         = tabnet_params['n_steps'],
            gamma           = tabnet_params['gamma'],
            lambda_sparse   = tabnet_params['lambda_sparse'],
            optimizer_fn    = torch.optim.Adam,
            optimizer_params = {'lr': tabnet_params['lr']},
            mask_type       = 'sparsemax',
            seed            = SEED,
            verbose         = 0,
        )
        tn.fit(
            X_tr_scaled_fe.astype(np.float32),
            y_tr.astype(np.float32).reshape(-1, 1),
            eval_set    = [(X_va_scaled_fe.astype(np.float32),
                            y_va.astype(np.float32).reshape(-1, 1))],
            eval_name   = ['val'],
            eval_metric = ['rmse'],
            max_epochs  = TABNET_MAX_EPOCHS,
            patience    = TABNET_PATIENCE,
            batch_size  = 256,
            virtual_batch_size = 64,
        )
        record('TabNet', tn.predict(X_va_scaled_fe.astype(np.float32)).flatten())

        print(f"  [{fold}]", end=' ', flush=True)
    print()

    # ── Aggregate: mean, std, 95% CI via t-distribution ─────────────
    def summarize(metric_key: str) -> dict:
        summary = {}
        for name in MODELS:
            arr    = np.array(fold_data[name][metric_key])
            n      = len(arr)
            mu     = float(np.mean(arr))
            sd     = float(np.std(arr, ddof=1))
            se     = sd / np.sqrt(n)
            ci     = float(se * t_dist.ppf(0.975, df=n - 1))
            summary[name] = {
                'mean':      mu,
                'std':       sd,
                'ci95_half': ci,
                'ci95_low':  mu - ci,
                'ci95_high': mu + ci,
                'n_folds':   n,
            }
        return summary

    return {
        'rmse':       summarize('rmse'),
        'mae':        summarize('mae'),
        'r2':         summarize('r2'),
        'accuracy':   summarize('acc'),
        'f1_macro':   summarize('f1'),
        'auc_roc':    summarize('auc'),
        # Raw per-fold scores for statistical tests (stripped before JSON export)
        '_fold_rmse': {m: fold_data[m]['rmse'] for m in MODELS},
        '_fold_mae':  {m: fold_data[m]['mae']  for m in MODELS},
    }


# ══════════════════════════════════════════════
# Statistical Tests
# ══════════════════════════════════════════════

def run_statistical_tests(fold_rmse: dict) -> dict:
    """
    Friedman test (all models simultaneously) +
    pairwise Wilcoxon signed-rank tests with Bonferroni correction.
    Input: fold_rmse[model_name] = list of 50 per-fold RMSE values.
    """
    model_names = list(fold_rmse.keys())
    rmse_arrays = [fold_rmse[m] for m in model_names]
    n_models    = len(model_names)
    n_pairs     = n_models * (n_models - 1) // 2

    stat, p_friedman = friedmanchisquare(*rmse_arrays)

    pairwise = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a, b = model_names[i], model_names[j]
            sa   = np.array(fold_rmse[a])
            sb   = np.array(fold_rmse[b])
            if np.allclose(sa, sb):
                p_raw = 1.0
            else:
                try:
                    _, p_raw = wilcoxon(sa, sb)
                except ValueError:
                    p_raw = 1.0
            p_adj = min(float(p_raw) * n_pairs, 1.0)
            pairwise[f'{a} vs {b}'] = {
                'p_raw':        float(p_raw),
                'p_bonferroni': float(p_adj),
                'significant':  bool(p_adj < 0.05),
                'winner':       a if np.mean(sa) < np.mean(sb) else b,
                'delta_rmse':   float(abs(np.mean(sa) - np.mean(sb))),
            }

    return {
        'friedman_stat':        float(stat),
        'friedman_p':           float(p_friedman),
        'friedman_significant': bool(p_friedman < 0.05),
        'n_pairs_bonferroni':   n_pairs,
        'pairwise_wilcoxon':    pairwise,
    }


# ══════════════════════════════════════════════
# Feature Engineering Ablation  (multiple model families, 5-fold)
# ══════════════════════════════════════════════

def feature_ablation_study(X_tree_raw: np.ndarray, X_scaled_raw: np.ndarray,
                            y: np.ndarray,
                            feature_names: list, xgb_params: dict,
                            lgb_params: dict, svr_params: dict) -> dict:
    """
    Compare raw vs engineered features for multiple model families.
    Minimum reviewer-facing coverage includes:
      1. XGBoost   (strong tree baseline)
      2. LightGBM  (strong tree baseline)
      3. SVR-RBF   (best point-estimate family in the main comparison)
    Uses 5-fold stratified CV.
    """
    bins = _stratified_bins(y)
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ablation = {}

    model_specs = {
        'XGBoost': {
            'raw_matrix': X_tree_raw,
            'scale_engineered': False,
            'builder': lambda: xgb.XGBRegressor(**xgb_params),
            'fit_kwargs': lambda X_va, y_va: {'eval_set': [(X_va, y_va)], 'verbose': False},
        },
        'LightGBM': {
            'raw_matrix': X_tree_raw,
            'scale_engineered': False,
            'builder': lambda: lgb.LGBMRegressor(**lgb_params),
            'fit_kwargs': lambda X_va, y_va: {},
        },
        'SVR-RBF': {
            'raw_matrix': X_scaled_raw,
            'scale_engineered': True,
            'builder': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(**svr_params)),
            ]),
            'fit_kwargs': lambda X_va, y_va: {},
        },
    }

    for model_name, spec in model_specs.items():
        model_ablation = {}
        for cond_name, use_fe in [('raw_30_features', False),
                                  ('engineered_56_features', True)]:
            rmses, maes, r2s = [], [], []
            n_feat_seen = 0
            X_source = spec['raw_matrix']
            for tr_idx, va_idx in skf.split(X_source, bins):
                X_tr, X_va = X_source[tr_idx], X_source[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                if use_fe:
                    fe       = FeatureEngineer(scale_engineered=spec['scale_engineered'])
                    X_tr_fe  = fe.fit_transform(X_tr, y_tr, feature_names)
                    X_va_fe  = fe.transform(X_va, feature_names)
                else:
                    X_tr_fe  = X_tr.astype(np.float32)
                    X_va_fe  = X_va.astype(np.float32)

                n_feat_seen = X_tr_fe.shape[1]
                model = spec['builder']()
                model.fit(X_tr_fe, y_tr, **spec['fit_kwargs'](X_va_fe, y_va))
                p = model.predict(X_va_fe)
                rmses.append(np.sqrt(mean_squared_error(y_va, p)))
                maes.append(mean_absolute_error(y_va, p))
                r2s.append(r2_score(y_va, p))

            model_ablation[cond_name] = {
                'n_features': n_feat_seen,
                'rmse_mean':  float(np.mean(rmses)),
                'rmse_std':   float(np.std(rmses)),
                'mae_mean':   float(np.mean(maes)),
                'mae_std':    float(np.std(maes)),
                'r2_mean':    float(np.mean(r2s)),
                'r2_std':     float(np.std(r2s)),
            }
            print(f"  {model_name:<12} {cond_name:<24}: n_feat={n_feat_seen:2d}  "
                  f"RMSE={model_ablation[cond_name]['rmse_mean']:.4f} ± "
                  f"{model_ablation[cond_name]['rmse_std']:.4f}  "
                  f"R²={model_ablation[cond_name]['r2_mean']:.4f}")

        delta_rmse = (model_ablation['raw_30_features']['rmse_mean']
                      - model_ablation['engineered_56_features']['rmse_mean'])
        delta_r2   = (model_ablation['engineered_56_features']['r2_mean']
                      - model_ablation['raw_30_features']['r2_mean'])
        model_ablation['delta'] = {
            'rmse_improvement': float(delta_rmse),
            'r2_improvement':   float(delta_r2),
        }
        ablation[model_name] = model_ablation

    return ablation


# ══════════════════════════════════════════════
# SHAP Feature Importance  (top-3 models)
# ══════════════════════════════════════════════

def compute_shap_values(X_tree_raw: np.ndarray, y: np.ndarray,
                        feature_names: list,
                        xgb_params: dict, lgb_params: dict,
                        cb_params: dict) -> dict:
    """
    Compute SHAP TreeExplainer values for XGBoost, LightGBM, CatBoost
    trained on the full dataset.
    Returns mean |SHAP| per feature and top-10 ranking.
    """
    import shap

    fe         = FeatureEngineer(scale_engineered=False)
    X_fe       = fe.fit_transform(X_tree_raw, y, feature_names)
    feat_names = fe.feature_names_out

    shap_results = {}

    for label, model in [
        ('XGBoost',  xgb.XGBRegressor(**xgb_params)),
        ('LightGBM', lgb.LGBMRegressor(**lgb_params)),
        ('CatBoost', CatBoostRegressor(**cb_params)),
    ]:
        if label == 'XGBoost':
            model.fit(X_fe, y, verbose=False)
        else:
            model.fit(X_fe, y)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_fe)
        if isinstance(sv, list):
            sv = sv[0]

        mean_abs = np.abs(sv).mean(axis=0).tolist()
        top10    = sorted(zip(feat_names, mean_abs),
                          key=lambda x: x[1], reverse=True)[:10]
        shap_results[label] = {
            'mean_abs_shap':  dict(zip(feat_names, [round(v, 6) for v in mean_abs])),
            'top10_features': [[n, round(v, 6)] for n, v in top10],
        }
        print(f"  {label:<12}: top feature = {top10[0][0]}"
              f"  (mean|SHAP|={top10[0][1]:.4f})")

    return shap_results


# ══════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════

def print_results_table(cv_results: dict):
    print(f"\n{'='*84}")
    print("CV RESULTS v4  (10×5-fold  |  95% CI via t-distribution)")
    print(f"{'='*84}")
    print(f"{'Model':<22} {'RMSE (mean ± 95%CI)':>22} {'MAE':>18} {'R²':>18}")
    print("-" * 84)
    ranked = sorted(MODELS, key=lambda m: cv_results['rmse'][m]['mean'])
    for name in ranked:
        r = cv_results['rmse'][name]
        m = cv_results['mae'][name]
        q = cv_results['r2'][name]
        print(f"{name:<22} "
              f"{r['mean']:.3f} ± {r['ci95_half']:.3f}    "
              f"{m['mean']:.3f} ± {m['ci95_half']:.3f}    "
              f"{q['mean']:.3f} ± {q['ci95_half']:.3f}")
    print('=' * 84)

    print(f"\n{'─'*72}")
    print("CLASSIFICATION RESULTS  (G3 ≥ 10 → pass/fail)")
    print(f"{'─'*72}")
    print(f"{'Model':<22} {'Accuracy':>12} {'F1-macro':>12} {'AUC-ROC':>12}")
    print("-" * 60)
    ranked2 = sorted(MODELS,
                     key=lambda m: cv_results['auc_roc'][m]['mean'],
                     reverse=True)
    for name in ranked2:
        a = cv_results['accuracy'][name]
        f = cv_results['f1_macro'][name]
        u = cv_results['auc_roc'][name]
        print(f"{name:<22} "
              f"{a['mean']:.3f}±{a['ci95_half']:.3f}    "
              f"{f['mean']:.3f}±{f['ci95_half']:.3f}    "
              f"{u['mean']:.3f}±{u['ci95_half']:.3f}")
    print('─' * 72)


def save_results(cv_results: dict, stat_tests: dict, ablation: dict,
                 shap_results: dict, best_params: dict,
                 feature_info: dict, timestamp: str,
                 output_dir: str, run_metadata: dict):

    os.makedirs(output_dir, exist_ok=True)

    # Strip internal raw-fold arrays before JSON export
    cv_export = {k: v for k, v in cv_results.items()
                 if not k.startswith('_')}

    payload = {
        'cv_results':   cv_export,
        'stat_tests':   stat_tests,
        'ablation':     ablation,
        'shap':         shap_results,
        'best_params':  best_params,
        'feature_info': feature_info,
        'timestamp':    timestamp,
        'run_metadata': run_metadata,
        'config': {
            'n_cv_splits':     N_CV_SPLITS,
            'n_cv_repeats':    N_CV_REPEATS,
            'n_optuna_trials': N_OPTUNA_TRIALS,
            'n_tabnet_trials': N_TABNET_TRIALS,
            'tabnet_tuning_protocol': 'Optuna on stratified 80/20 holdout split',
            'pass_threshold':  PASS_THRESHOLD,
            'models':          MODELS,
        },
    }

    json_path = os.path.join(output_dir, 'complete_results.json')
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults saved → {json_path}")

    # Flat CSV for quick inspection
    rows = []
    for name in MODELS:
        rows.append({
            'Model':    name,
            'RMSE':     f"{cv_results['rmse'][name]['mean']:.3f} ± "
                        f"{cv_results['rmse'][name]['ci95_half']:.3f}",
            'MAE':      f"{cv_results['mae'][name]['mean']:.3f} ± "
                        f"{cv_results['mae'][name]['ci95_half']:.3f}",
            'R2':       f"{cv_results['r2'][name]['mean']:.3f} ± "
                        f"{cv_results['r2'][name]['ci95_half']:.3f}",
            'Accuracy': f"{cv_results['accuracy'][name]['mean']:.3f}",
            'F1_macro': f"{cv_results['f1_macro'][name]['mean']:.3f}",
            'AUC_ROC':  f"{cv_results['auc_roc'][name]['mean']:.3f}",
        })
    csv_path = os.path.join(output_dir, 'cv_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV saved     → {csv_path}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run_benchmark(track: BenchmarkTrack, subject_subset: str,
            results_root: str, refresh_data: bool = False):
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION v5")
    print("=" * 60)
    print(f"Track: {track.name}  |  Subject subset: {subject_subset}")
    print(track.description)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n[1] Loading data...")
    dataset = StudentPerformanceDataset(data_dir='./data')
    benchmark_data = prepare_benchmark_data(
      dataset,
      track=track,
      subject_subset=subject_subset,
      refresh_data=refresh_data,
    )
    X_tree = benchmark_data['tree']
    X_scaled = benchmark_data['scaled']
    y = benchmark_data['y']
    feature_names = benchmark_data['feature_names']
    print(f"    Samples: {X_tree.shape[0]}   Raw features: {X_tree.shape[1]}")

    # ── 2. Global FE for Optuna tuning ──────────────────────────────
    print("\n[2] Building model-aware feature sets for hyperparameter tuning...")
    fe_tree_global = FeatureEngineer(scale_engineered=False)
    X_tree_global = fe_tree_global.fit_transform(X_tree, y, feature_names)
    fe_scaled_global = FeatureEngineer(scale_engineered=True)
    X_scaled_global = fe_scaled_global.fit_transform(X_scaled, y, feature_names)
    print(f"    Tree feature width:   {X_tree_global.shape[1]}"
        f"  (+{X_tree_global.shape[1] - X_tree.shape[1]} engineered)")
    print(f"    Scaled feature width: {X_scaled_global.shape[1]}"
        f"  (+{X_scaled_global.shape[1] - X_scaled.shape[1]} engineered)")

    # ── 3. Optuna hyperparameter search ─────────────────────────────
    print(f"\n[3] Optuna search ({N_OPTUNA_TRIALS} trials each)...")

    print("    [3a] Random Forest...", flush=True)
    rf_params = tune_rf(X_tree_global, y)
    print(f"         {rf_params}")

    print("    [3b] XGBoost...", flush=True)
    xgb_params = tune_xgb(X_tree_global, y)
    print(f"         {xgb_params}")

    print("    [3c] LightGBM...", flush=True)
    lgb_params = tune_lgb(X_tree_global, y)
    print(f"         {lgb_params}")

    print("    [3d] CatBoost...", flush=True)
    cb_params = tune_catboost(X_tree_global, y)
    print(f"         {cb_params}")

    print("    [3e] SVR-RBF...", flush=True)
    svr_params = tune_svr(X_scaled_global, y)
    print(f"         {svr_params}")

    print(f"    [3f] TabNet ({N_TABNET_TRIALS} trials on 80/20 split)...",
          flush=True)
    tn_params = tune_tabnet(X_scaled_global, y)
    print(f"         {tn_params}")

    # ── 4. Main evaluation: 10×5-fold Repeated Stratified CV ────────
    cv_results = cross_validate_v4(
      X_tree, X_scaled, y, feature_names,
        rf_params, xgb_params, lgb_params,
        cb_params, svr_params, tn_params)

    # ── 5. Statistical significance tests ───────────────────────────
    print("\n[5] Statistical tests (Friedman + pairwise Wilcoxon)...")
    stat_tests = run_statistical_tests(cv_results['_fold_rmse'])
    print(f"    Friedman chi2={stat_tests['friedman_stat']:.3f}  "
          f"p={stat_tests['friedman_p']:.4f}  "
          f"significant={stat_tests['friedman_significant']}")
    sig_pairs = [(k, v) for k, v in stat_tests['pairwise_wilcoxon'].items()
                 if v['significant']]
    print(f"    Significant pairwise differences (Bonferroni): "
          f"{len(sig_pairs)} / {stat_tests['n_pairs_bonferroni']}")
    for pair, info in sig_pairs[:5]:   # print first 5 for brevity
        print(f"      {pair}: winner={info['winner']}  "
              f"Δ={info['delta_rmse']:.4f}  p={info['p_bonferroni']:.4f}")

    # ── 6. Feature engineering ablation ─────────────────────────────
    print("\n[6] Feature engineering ablation (XGBoost, LightGBM, SVR-RBF; 5-fold)...")
    ablation = feature_ablation_study(
        X_tree, X_scaled, y, feature_names, xgb_params, lgb_params, svr_params)
    for model_name, model_ablation in ablation.items():
        print(f"    {model_name:<12}: ΔRMSE={model_ablation['delta']['rmse_improvement']:.4f}"
              f"  ΔR²={model_ablation['delta']['r2_improvement']:.4f}")

    # ── 7. SHAP values ───────────────────────────────────────────────
    print("\n[7] SHAP values (XGBoost, LightGBM, CatBoost — full dataset)...")
    shap_results = compute_shap_values(
        X_tree, y, feature_names, xgb_params, lgb_params, cb_params)

    # ── 8. Print summary tables ──────────────────────────────────────
    print_results_table(cv_results)

    # ── 9. Save ──────────────────────────────────────────────────────
    feature_info = {
        'tree_raw_features':      int(X_tree.shape[1]),
        'scaled_raw_features':    int(X_scaled.shape[1]),
        'tree_enhanced_features': int(X_tree_global.shape[1]),
        'scaled_enhanced_features': int(X_scaled_global.shape[1]),
        'feature_names_tree':     fe_tree_global.feature_names_out,
        'feature_names_scaled':   fe_scaled_global.feature_names_out,
    }
    best_params = {
        'rf':       rf_params,
        'xgb':      xgb_params,
        'lgb':      lgb_params,
        'catboost': cb_params,
        'svr':      svr_params,
        'tabnet':   tn_params,
    }
    output_dir = get_output_dir(results_root, subject_subset, track)
    save_results(cv_results, stat_tests, ablation, shap_results,
                 best_params, feature_info,
                 datetime.now().isoformat(),
                 output_dir=output_dir,
                 run_metadata={
                     'track': asdict(track),
                     'subject_subset': subject_subset,
                 })

    print(f"\nDone. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return cv_results


def main():
    args = parse_args()
    all_results = {}
    for track_name in args.tracks:
        track = TRACKS[track_name]
        all_results[track_name] = run_benchmark(
            track=track,
            subject_subset=args.subject_subset,
            results_root=args.results_root,
            refresh_data=args.refresh_data,
        )
    return all_results


if __name__ == '__main__':
    results = main()
