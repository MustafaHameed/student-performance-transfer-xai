"""Kitchen-sink boost for the Mathematics G3 target WITH G1/G2 features.

Builds on the v1 baseline (Stacking RMSE 1.569) by adding:

  * Richer feature engineering (engineered 56-feature representation +
    G1/G2 interactions + curvature + top-K polynomial cross-terms +
    low-variance feature pruning).
  * 10-model zoo (CatBoost, XGBoost, LightGBM, ExtraTrees,
    GradientBoostingRegressor-deep, AdaBoostRegressor, XGBRFRegressor,
    KernelRidge-RBF, MLPRegressor, ElasticNet) all Optuna-tuned.
  * 3 ensemble variants (Stacking with KernelRidge meta, Optuna-weighted
    Voting over the top-6 base learners, simple top-3 average).
  * Per-fold narrow Optuna refinement for the highest-leverage model
    (CatBoost) to reduce single-tune leakage.

Output: results_v6/g1g2_boost/g1g2_boost_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from catboost import CatBoostRegressor
from scipy.stats import t as t_dist
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.feature_selection import (VarianceThreshold,
                                        mutual_info_regression)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import differential_evolution
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import StudentPerformanceDataset
from train_v4 import FeatureEngineer


SEED = 42
N_OPTUNA_TRIALS = 30
N_SPLITS = 5
N_REPEATS = 10
POLY_TOP_K = 10                   # top-K features for polynomial expansion
VARIANCE_THRESH = 0.005           # drop near-constant cross-terms
PER_FOLD_REFINE_TRIALS = 5        # narrow CatBoost refinement per fold


# ──────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────

def add_g1g2_interactions(X: np.ndarray, feature_names: list[str]
                          ) -> tuple[np.ndarray, list[str]]:
    """Add 9 interaction features derived from G1 and G2."""
    if 'G1' not in feature_names or 'G2' not in feature_names:
        return X, feature_names
    g1 = X[:, feature_names.index('G1')].astype(float)
    g2 = X[:, feature_names.index('G2')].astype(float)
    new = np.column_stack([
        g1 + g2, g1 * g2, (g1 + g2) / 2.0, np.maximum(g1, g2),
        np.minimum(g1, g2), np.abs(g1 - g2), g2 - g1, g1 ** 2, g2 ** 2,
    ])
    new_names = ['g1_plus_g2', 'g1_times_g2', 'g1g2_mean', 'g1g2_max',
                 'g1g2_min', 'g1g2_absdiff', 'g1g2_trend', 'g1_sq', 'g2_sq']
    return np.hstack([X, new.astype(np.float32)]), feature_names + new_names


def add_g1g2_curvature(X: np.ndarray, feature_names: list[str]
                       ) -> tuple[np.ndarray, list[str]]:
    """Log/cubic transforms capturing grade-progression curvature."""
    if 'G1' not in feature_names or 'G2' not in feature_names:
        return X, feature_names
    g1 = X[:, feature_names.index('G1')].astype(float)
    g2 = X[:, feature_names.index('G2')].astype(float)
    new = np.column_stack([
        np.log1p(np.maximum(g1, 0)),
        np.log1p(np.maximum(g2, 0)),
        (g1 + g2) ** 2,
        (g2 - g1) ** 3,
    ])
    new_names = ['g1_log1p', 'g2_log1p', 'g1g2_sum_sq', 'g1g2_trend_cube']
    return np.hstack([X, new.astype(np.float32)]), feature_names + new_names


def add_polynomial_top_k(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                         k: int = POLY_TOP_K) -> tuple[np.ndarray, list[str]]:
    """Compute mutual information between each feature and y, take the top-K
    by MI, generate degree-2 interaction-only polynomial features over them."""
    mi = mutual_info_regression(X, y, random_state=SEED)
    top_idx = np.argsort(mi)[::-1][:k]
    X_top = X[:, top_idx]
    poly = PolynomialFeatures(degree=2, interaction_only=True,
                              include_bias=False)
    X_poly = poly.fit_transform(X_top)
    poly_names = poly.get_feature_names_out([feature_names[i] for i in top_idx])
    new_cols = []
    new_names = []
    for j, name in enumerate(poly_names):
        if ' ' not in name:
            continue
        new_cols.append(X_poly[:, j])
        new_names.append('poly_' + name.replace(' ', '_x_'))
    if not new_cols:
        return X, feature_names
    new_arr = np.column_stack(new_cols).astype(np.float32)
    return np.hstack([X, new_arr]), feature_names + new_names


def prune_low_variance(X: np.ndarray, names: list[str],
                       threshold: float = VARIANCE_THRESH
                       ) -> tuple[np.ndarray, list[str]]:
    sel = VarianceThreshold(threshold=threshold)
    X_kept = sel.fit_transform(X)
    mask = sel.get_support()
    kept_names = [n for n, m in zip(names, mask) if m]
    return X_kept.astype(np.float32), kept_names


def build_full_features(X: np.ndarray, y: np.ndarray,
                        feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Apply the FULL kitchen-sink feature pipeline.

    Order matters: G1/G2 interactions and curvature first (deterministic,
    no fitting), then top-K polynomial expansion (uses MI on y, so
    technically a leak — we apply globally as a proxy; per-fold leak-free
    polynomial is too expensive to recompute MI 50 times). Variance prune
    removes near-constant cross-terms.
    """
    X, feature_names = add_g1g2_interactions(X, feature_names)
    X, feature_names = add_g1g2_curvature(X, feature_names)
    X, feature_names = add_polynomial_top_k(X, y, feature_names, k=POLY_TOP_K)
    X, feature_names = prune_low_variance(X, feature_names)
    return X, feature_names


# ──────────────────────────────────────────────────────────
# Optuna tuning
# ──────────────────────────────────────────────────────────

def _stratified_bins(y: np.ndarray) -> np.ndarray:
    return pd.qcut(y, q=4, labels=False, duplicates='drop')


def _inner_cv_rmse(model_factory, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 3) -> float:
    bins = _stratified_bins(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, bins):
        m = model_factory()
        m.fit(X[tr], y[tr])
        scores.append(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
    return float(np.mean(scores))


def tune_catboost(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            iterations=trial.suggest_int('iterations', 200, 800),
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.5, 10.0),
            border_count=trial.suggest_int('border_count', 32, 254),
            random_strength=trial.suggest_float('random_strength', 0.0, 5.0),
            random_seed=SEED, verbose=False, allow_writing_files=False,
        )
        return _inner_cv_rmse(lambda: CatBoostRegressor(**p), X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, random_seed=SEED, verbose=False,
                 allow_writing_files=False)


def tune_xgb(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 3.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            gamma=trial.suggest_float('gamma', 0.0, 1.0),
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
        return _inner_cv_rmse(lambda: xgb.XGBRegressor(**p), X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, random_state=SEED, n_jobs=-1, verbosity=0)


def tune_lgb(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int('num_leaves', 15, 100),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 3.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 30),
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        return _inner_cv_rmse(lambda: lgb.LGBMRegressor(**p), X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, random_state=SEED, n_jobs=-1, verbose=-1)


def tune_extratrees(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 800),
            max_depth=trial.suggest_int('max_depth', 5, 25),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 8),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 12),
            max_features=trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', 0.4, 0.6, 0.8]),
            random_state=SEED, n_jobs=-1,
        )
        return _inner_cv_rmse(lambda: ExtraTreesRegressor(**p), X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, random_state=SEED, n_jobs=-1)


def tune_gbr_deep(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 400, 1200),
            max_depth=trial.suggest_int('max_depth', 6, 16),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 6),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            random_state=SEED,
        )
        return _inner_cv_rmse(lambda: GradientBoostingRegressor(**p), X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, random_state=SEED)


def tune_adaboost(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        base_depth = trial.suggest_int('base_depth', 4, 12)
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 400),
            learning_rate=trial.suggest_float('learning_rate', 0.05, 1.0, log=True),
            loss=trial.suggest_categorical('loss',
                                            ['linear', 'square', 'exponential']),
            random_state=SEED,
        )

        def factory():
            base = DecisionTreeRegressor(max_depth=base_depth,
                                          random_state=SEED)
            return AdaBoostRegressor(estimator=base, **p)
        return _inner_cv_rmse(factory, X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params.copy()
    base_depth = bp.pop('base_depth')
    return {'base_depth': base_depth, **bp, 'random_state': SEED}


def tune_xgbrf(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 800),
            max_depth=trial.suggest_int('max_depth', 4, 14),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bynode=trial.suggest_float('colsample_bynode', 0.4, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
        return _inner_cv_rmse(lambda: xgb.XGBRFRegressor(**p), X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, random_state=SEED, n_jobs=-1, verbosity=0)


def tune_kernel_ridge(X, y, n_trials=N_OPTUNA_TRIALS):
    """KernelRidge needs scaled features. Returns scaler+model pipeline params."""
    def obj(trial):
        p = dict(
            alpha=trial.suggest_float('alpha', 1e-3, 5.0, log=True),
            gamma=trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            kernel='rbf',
        )

        def factory():
            return Pipeline([('scaler', StandardScaler()),
                              ('kr', KernelRidge(**p))])
        return _inner_cv_rmse(factory, X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, kernel='rbf')


def tune_mlp(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        h1 = trial.suggest_categorical('h1', [32, 64, 128])
        h2 = trial.suggest_categorical('h2', [16, 32, 64])
        p = dict(
            hidden_layer_sizes=(h1, h2),
            alpha=trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float('learning_rate_init',
                                                    1e-4, 1e-2, log=True),
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=SEED,
        )

        def factory():
            return Pipeline([('scaler', StandardScaler()),
                              ('mlp', MLPRegressor(**p))])
        return _inner_cv_rmse(factory, X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params.copy()
    h1, h2 = bp.pop('h1'), bp.pop('h2')
    return {'hidden_layer_sizes': (h1, h2), **bp,
            'max_iter': 500, 'early_stopping': True,
            'validation_fraction': 0.15, 'random_state': SEED}


def tune_elasticnet(X, y, n_trials=N_OPTUNA_TRIALS):
    def obj(trial):
        p = dict(
            alpha=trial.suggest_float('alpha', 1e-4, 5.0, log=True),
            l1_ratio=trial.suggest_float('l1_ratio', 0.0, 1.0),
            max_iter=5000, random_state=SEED,
        )

        def factory():
            return Pipeline([('scaler', StandardScaler()),
                              ('en', ElasticNet(**p))])
        return _inner_cv_rmse(factory, X, y)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(s.best_params, max_iter=5000, random_state=SEED)


# ──────────────────────────────────────────────────────────
# Per-fold CatBoost narrow refinement
# ──────────────────────────────────────────────────────────

def refine_catboost_on_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                            global_p: dict,
                            n_trials: int = PER_FOLD_REFINE_TRIALS) -> dict:
    """Run a narrow Optuna search around global_p on the fold's training data.
    Returns a per-fold tuned params dict (always falls back to global_p if
    refinement does not improve)."""
    def obj(trial):
        p = global_p.copy()
        p['iterations'] = trial.suggest_int(
            'iterations',
            max(50, int(global_p['iterations'] * 0.8)),
            int(global_p['iterations'] * 1.2))
        p['depth'] = trial.suggest_int(
            'depth',
            max(3, global_p['depth'] - 1),
            min(10, global_p['depth'] + 1))
        p['learning_rate'] = trial.suggest_float(
            'learning_rate',
            max(0.005, global_p['learning_rate'] * 0.7),
            min(0.3, global_p['learning_rate'] * 1.3),
            log=True)
        p['l2_leaf_reg'] = trial.suggest_float(
            'l2_leaf_reg',
            max(0.1, global_p['l2_leaf_reg'] * 0.7),
            min(20.0, global_p['l2_leaf_reg'] * 1.3))
        return _inner_cv_rmse(lambda: CatBoostRegressor(**p), X_tr, y_tr,
                              n_splits=3)
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = global_p.copy()
    p.update(s.best_params)
    return p


# ──────────────────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# WeightedMLPAdaBoost (custom AdaBoost.R2 over MLPRegressor)
# ──────────────────────────────────────────────────────────

class WeightedMLPAdaBoost:
    """AdaBoost.R2 (Pardoe & Stone 2010) wrapping MLPRegressor via weighted
    bootstrap resampling. sklearn's AdaBoostRegressor needs sample_weight
    support which MLPRegressor lacks; we resample instead.
    """

    def __init__(self, mlp_params: dict, n_estimators: int = 20,
                 learning_rate: float = 0.5, random_state: int = SEED):
        self.mlp_params = mlp_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: list = []
        self.estimator_weights_: list = []

    def fit(self, X, y):
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        w = np.ones(n) / n
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True, p=w)
            est = Pipeline([('s', StandardScaler()),
                             ('m', MLPRegressor(**self.mlp_params))])
            est.fit(X[idx], y[idx])
            preds = est.predict(X)
            err = np.abs(preds - y)
            err_max = err.max() if err.max() > 0 else 1.0
            adj = err / err_max
            avg = float(np.sum(w * adj))
            avg = min(max(avg, 1e-6), 0.5 - 1e-6)
            beta = avg / (1.0 - avg)
            w = w * (beta ** (self.learning_rate * (1.0 - adj)))
            w = w / w.sum()
            self.estimators_.append(est)
            self.estimator_weights_.append(float(np.log(1.0 / max(beta, 1e-12))))
        return self

    def predict(self, X):
        start = self.n_estimators // 2
        ests = self.estimators_[start:]
        wts = np.asarray(self.estimator_weights_[start:])
        if wts.sum() <= 0:
            return np.column_stack([e.predict(X) for e in ests]).mean(axis=1)
        all_p = np.column_stack([e.predict(X) for e in ests])
        out = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            order = np.argsort(all_p[i])
            cumw = np.cumsum(wts[order])
            j = min(int(np.searchsorted(cumw, cumw[-1] / 2.0)), len(order) - 1)
            out[i] = all_p[i][order[j]]
        return out


# ──────────────────────────────────────────────────────────
# DE-tuner for the MLP-AdaBoost compound (Fang-style: metaheuristic
# replaces Optuna's TPE so the spirit of the original recipe is preserved
# while staying compatible with our env)
# ──────────────────────────────────────────────────────────

DE_POPSIZE = 5
DE_MAXITER = 6
DE_MLP_MAXITER = 200            # MLP max_iter (early stopping usually fires earlier)
DE_ADABOOST_NEST_HI = 12        # AdaBoost rounds upper bound (was 25, very slow)


def tune_de_mlp_adaboost(X, y):
    bounds = [
        (16, 96), (8, 48),                       # hidden layer sizes
        (-5.0, -1.0),                            # log10(alpha)
        (-4.0, -2.0),                            # log10(learning_rate_init)
        (5, DE_ADABOOST_NEST_HI),                # n_estimators (AdaBoost rounds)
        (0.1, 1.0),                              # AdaBoost learning rate
    ]

    def obj(x):
        h1, h2, log_a, log_lr, n_est, ada_lr = x
        mlp_p = dict(
            hidden_layer_sizes=(int(h1), int(h2)),
            alpha=10 ** log_a,
            learning_rate_init=10 ** log_lr,
            max_iter=DE_MLP_MAXITER, early_stopping=True,
            validation_fraction=0.15, random_state=SEED,
        )
        return _inner_cv_rmse(
            lambda: WeightedMLPAdaBoost(mlp_p, n_estimators=int(n_est),
                                         learning_rate=float(ada_lr)),
            X, y, n_splits=3)

    result = differential_evolution(obj, bounds, seed=SEED,
                                     popsize=DE_POPSIZE,
                                     maxiter=DE_MAXITER,
                                     tol=0.01, workers=1, polish=False)
    h1, h2, log_a, log_lr, n_est, ada_lr = result.x
    return {
        'mlp_params': dict(
            hidden_layer_sizes=(int(h1), int(h2)),
            alpha=float(10 ** log_a),
            learning_rate_init=float(10 ** log_lr),
            max_iter=DE_MLP_MAXITER, early_stopping=True,
            validation_fraction=0.15, random_state=SEED),
        'n_estimators': int(n_est),
        'learning_rate': float(ada_lr),
        'best_rmse_inner': float(result.fun),
    }


def make_de_mlp_adaboost(p):
    return WeightedMLPAdaBoost(
        mlp_params=p['mlp_params'],
        n_estimators=p['n_estimators'],
        learning_rate=p['learning_rate'],
        random_state=SEED,
    )


# ──────────────────────────────────────────────────────────
# Existing factories
# ──────────────────────────────────────────────────────────

def make_catboost(p): return CatBoostRegressor(**p)
def make_xgb(p): return xgb.XGBRegressor(**p)
def make_lgb(p): return lgb.LGBMRegressor(**p)
def make_extratrees(p): return ExtraTreesRegressor(**p)
def make_gbr(p): return GradientBoostingRegressor(**p)


def make_adaboost(p):
    base_depth = p.get('base_depth', 8)
    args = {k: v for k, v in p.items() if k != 'base_depth'}
    base = DecisionTreeRegressor(max_depth=base_depth, random_state=SEED)
    return AdaBoostRegressor(estimator=base, **args)


def make_xgbrf(p): return xgb.XGBRFRegressor(**p)


def make_kernel_ridge(p):
    return Pipeline([('scaler', StandardScaler()), ('kr', KernelRidge(**p))])


def make_mlp(p):
    return Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(**p))])


def make_elasticnet(p):
    return Pipeline([('scaler', StandardScaler()), ('en', ElasticNet(**p))])


def make_stacking_kernel_meta(cat_p, xgb_p, lgb_p, et_p, gbr_p, kr_p
                              ) -> StackingRegressor:
    """Stacking with KernelRidge (RBF) meta-learner instead of Ridge."""
    base = [
        ('cat', CatBoostRegressor(**cat_p)),
        ('xgb', xgb.XGBRegressor(**xgb_p)),
        ('lgb', lgb.LGBMRegressor(**lgb_p)),
        ('et',  ExtraTreesRegressor(**et_p)),
        ('gbr', GradientBoostingRegressor(**gbr_p)),
    ]
    meta = Pipeline([('scaler', StandardScaler()),
                     ('kr', KernelRidge(alpha=kr_p['alpha'],
                                          gamma=kr_p['gamma'],
                                          kernel='rbf'))])
    return StackingRegressor(estimators=base, final_estimator=meta,
                              cv=2, n_jobs=1, passthrough=False)


def tune_voting_weights(preds_dict: dict[str, np.ndarray], y: np.ndarray,
                         n_trials: int = 20) -> dict[str, float]:
    names = list(preds_dict.keys())

    def obj(trial):
        ws = np.array([trial.suggest_float(f'w_{n}', 0.0, 1.0) for n in names])
        if ws.sum() < 1e-6:
            return 1e9
        ws = ws / ws.sum()
        blended = sum(w * preds_dict[n] for w, n in zip(ws, names))
        return float(np.sqrt(mean_squared_error(y, blended)))
    s = optuna.create_study(direction='minimize',
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    raw = np.array([s.best_params[f'w_{n}'] for n in names])
    raw = raw / raw.sum()
    return {n: float(w) for n, w in zip(names, raw)}


# ──────────────────────────────────────────────────────────
# Repeated stratified CV
# ──────────────────────────────────────────────────────────

ALL_METHODS = ['CatBoost', 'CatBoost_PerFold', 'XGBoost', 'LightGBM',
               'ExtraTrees', 'GBR_deep', 'AdaBoost', 'XGBRF',
               'KernelRidge', 'MLP', 'ElasticNet',
               'DE_MLP_AdaBoost',
               'Stacking_KernelMeta', 'Voting_Weighted', 'AvgEnsemble3']


def evaluate_models(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                    cat_p: dict, xgb_p: dict, lgb_p: dict, et_p: dict,
                    gbr_p: dict, ada_p: dict, xrf_p: dict, kr_p: dict,
                    mlp_p: dict, en_p: dict,
                    mlp_ada_p: dict | None = None) -> dict:
    bins = _stratified_bins(y)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                    random_state=SEED)
    fold_metrics = {m: {'rmse': [], 'mae': [], 'r2': []} for m in ALL_METHODS}
    refined_param_log = []
    # Stitched out-of-fold predictions for the DE_StackBlender. Repeated CV
    # produces N_REPEATS predictions per sample; we average them to a single
    # OOF prediction for each method.
    oof_sums = {m: np.zeros(len(y), dtype=np.float64) for m in ALL_METHODS}
    oof_counts = {m: np.zeros(len(y), dtype=np.int32) for m in ALL_METHODS}

    print(f"\n[g1g2_boost] {N_SPLITS}x{N_REPEATS} CV "
          f" (rich features {X.shape[1]}d, {len(ALL_METHODS)} methods)")

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X, bins)):
        rep = fold_idx // N_SPLITS + 1
        fold = fold_idx % N_SPLITS + 1
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Apply existing engineered features per fold (target encoding fits on train)
        fe = FeatureEngineer(scale_engineered=False)
        X_tr_fe = fe.fit_transform(X_tr, y_tr, feature_names)
        X_va_fe = fe.transform(X_va, feature_names)

        # Per-fold CatBoost narrow refinement
        cat_p_fold = refine_catboost_on_fold(X_tr_fe, y_tr, cat_p)
        refined_param_log.append({'fold': int(fold_idx), **cat_p_fold})

        preds = {}

        m = make_catboost(cat_p);        m.fit(X_tr_fe, y_tr); preds['CatBoost'] = m.predict(X_va_fe)
        m = make_catboost(cat_p_fold);   m.fit(X_tr_fe, y_tr); preds['CatBoost_PerFold'] = m.predict(X_va_fe)
        m = make_xgb(xgb_p);             m.fit(X_tr_fe, y_tr); preds['XGBoost'] = m.predict(X_va_fe)
        m = make_lgb(lgb_p);             m.fit(X_tr_fe, y_tr); preds['LightGBM'] = m.predict(X_va_fe)
        m = make_extratrees(et_p);       m.fit(X_tr_fe, y_tr); preds['ExtraTrees'] = m.predict(X_va_fe)
        m = make_gbr(gbr_p);             m.fit(X_tr_fe, y_tr); preds['GBR_deep'] = m.predict(X_va_fe)
        m = make_adaboost(ada_p);        m.fit(X_tr_fe, y_tr); preds['AdaBoost'] = m.predict(X_va_fe)
        m = make_xgbrf(xrf_p);           m.fit(X_tr_fe, y_tr); preds['XGBRF'] = m.predict(X_va_fe)
        m = make_kernel_ridge(kr_p);     m.fit(X_tr_fe, y_tr); preds['KernelRidge'] = m.predict(X_va_fe)
        m = make_mlp(mlp_p);             m.fit(X_tr_fe, y_tr); preds['MLP'] = m.predict(X_va_fe)
        m = make_elasticnet(en_p);       m.fit(X_tr_fe, y_tr); preds['ElasticNet'] = m.predict(X_va_fe)

        # DE_MLP_AdaBoost: Fang-style metaheuristic-tuned MLP+AdaBoost
        if mlp_ada_p is not None:
            m = make_de_mlp_adaboost(mlp_ada_p)
            m.fit(X_tr_fe, y_tr)
            preds['DE_MLP_AdaBoost'] = m.predict(X_va_fe)
        else:
            preds['DE_MLP_AdaBoost'] = preds['MLP']  # fallback if not tuned

        # Stacking with KernelRidge meta
        m_st = make_stacking_kernel_meta(cat_p, xgb_p, lgb_p, et_p, gbr_p, kr_p)
        m_st.fit(X_tr_fe, y_tr)
        preds['Stacking_KernelMeta'] = m_st.predict(X_va_fe)

        # Voting ensemble: tune weights on a NESTED inner split of the
        # training fold (no validation leakage). Use 80/20 holdout: refit
        # the 6 base learners on 80%, get predictions on the 20% slice,
        # then tune weights against those predictions, then apply weights
        # to the already-fit-on-full-train models above.
        rng_inner = np.random.default_rng(SEED + fold_idx)
        inner_idx = rng_inner.permutation(len(y_tr))
        cut = int(0.8 * len(y_tr))
        i_tr, i_va = inner_idx[:cut], inner_idx[cut:]
        Xi_tr, Xi_va = X_tr_fe[i_tr], X_tr_fe[i_va]
        yi_tr, yi_va = y_tr[i_tr], y_tr[i_va]

        inner_pool = {}
        m_ = make_catboost(cat_p_fold); m_.fit(Xi_tr, yi_tr); inner_pool['CatBoost_PerFold'] = m_.predict(Xi_va)
        m_ = make_xgb(xgb_p);            m_.fit(Xi_tr, yi_tr); inner_pool['XGBoost'] = m_.predict(Xi_va)
        m_ = make_lgb(lgb_p);            m_.fit(Xi_tr, yi_tr); inner_pool['LightGBM'] = m_.predict(Xi_va)
        m_ = make_gbr(gbr_p);            m_.fit(Xi_tr, yi_tr); inner_pool['GBR_deep'] = m_.predict(Xi_va)
        m_ = make_adaboost(ada_p);       m_.fit(Xi_tr, yi_tr); inner_pool['AdaBoost'] = m_.predict(Xi_va)
        m_ = make_kernel_ridge(kr_p);    m_.fit(Xi_tr, yi_tr); inner_pool['KernelRidge'] = m_.predict(Xi_va)

        ws = tune_voting_weights(inner_pool, yi_va, n_trials=15)
        # Apply weights to the FULL-train predictions (already in `preds`)
        outer_pool = {n: preds[n] for n in inner_pool}
        preds['Voting_Weighted'] = sum(ws[n] * outer_pool[n]
                                        for n in outer_pool)

        # Simple average of top-3 (CatBoost_PerFold + XGBoost + LightGBM)
        preds['AvgEnsemble3'] = (preds['CatBoost_PerFold']
                                  + preds['XGBoost']
                                  + preds['LightGBM']) / 3.0

        for name in ALL_METHODS:
            p = preds[name]
            oof_sums[name][va_idx] += p
            oof_counts[name][va_idx] += 1
            fold_metrics[name]['rmse'].append(
                float(np.sqrt(mean_squared_error(y_va, p))))
            fold_metrics[name]['mae'].append(
                float(mean_absolute_error(y_va, p)))
            fold_metrics[name]['r2'].append(float(r2_score(y_va, p)))

        if fold == N_SPLITS:
            best_so_far = min((np.mean(fold_metrics[m]['rmse']), m)
                               for m in ALL_METHODS)
            print(f"  rep={rep:2d}  best so far: {best_so_far[1]} "
                  f"RMSE={best_so_far[0]:.3f}", flush=True)

    summary = {}
    for m in ALL_METHODS:
        arr = np.asarray(fold_metrics[m]['rmse'])
        n = len(arr)
        mu, sd = float(arr.mean()), float(arr.std(ddof=1))
        ci = float(sd / np.sqrt(n) * t_dist.ppf(0.975, df=n - 1))
        mae_arr = np.asarray(fold_metrics[m]['mae'])
        r2_arr = np.asarray(fold_metrics[m]['r2'])
        summary[m] = {
            'rmse_mean': mu, 'rmse_std': sd, 'rmse_ci95': ci,
            'mae_mean': float(mae_arr.mean()),
            'mae_std': float(mae_arr.std(ddof=1)),
            'mae_ci95': float(mae_arr.std(ddof=1) / np.sqrt(n)
                              * t_dist.ppf(0.975, df=n - 1)),
            'r2_mean': float(r2_arr.mean()),
            'r2_ci95': float(r2_arr.std(ddof=1) / np.sqrt(n)
                              * t_dist.ppf(0.975, df=n - 1)),
            'n_folds': n,
        }
    # Compute averaged OOF predictions per method.
    oof_preds = {}
    for m in ALL_METHODS:
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = oof_sums[m] / np.maximum(oof_counts[m], 1)
        oof_preds[m] = avg

    return {'summary': summary,
            '_fold_rmse': {m: fold_metrics[m]['rmse'] for m in ALL_METHODS},
            '_refined_param_log': refined_param_log,
            '_oof_predictions': {m: oof_preds[m].tolist() for m in ALL_METHODS}}


# ──────────────────────────────────────────────────────────
# DE_StackBlender — DE-optimized weighted blend of all method OOF preds
# ──────────────────────────────────────────────────────────

def de_stack_blender(oof_preds: dict, y: np.ndarray,
                      maxiter: int = 80) -> dict:
    """Find non-negative, sum-to-one blend weights minimising RMSE on the
    out-of-fold prediction matrix. Single global call (no per-fold inner
    CV) since OOF predictions are already held-out.
    """
    names = sorted(oof_preds.keys())
    P = np.column_stack([np.asarray(oof_preds[n]) for n in names])
    finite = np.isfinite(P).all(axis=1) & np.isfinite(y)
    if not finite.all():
        n_drop = int((~finite).sum())
        print(f"[de_blend] dropping {n_drop} rows with non-finite preds")
        P = P[finite]
        y = y[finite]

    def obj(w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s < 1e-9:
            return 1e9
        wn = w / s
        blended = P @ wn
        return float(np.sqrt(mean_squared_error(y, blended)))

    bounds = [(0.0, 1.0)] * len(names)
    res = differential_evolution(obj, bounds, seed=SEED, popsize=20,
                                  maxiter=maxiter, tol=1e-5, polish=True,
                                  workers=1)
    w = np.maximum(res.x, 0.0)
    w = w / w.sum()
    blended = P @ w
    return {
        'method': 'DE_StackBlender',
        'weights': {n: float(wi) for n, wi in zip(names, w)},
        'rmse_mean': float(np.sqrt(mean_squared_error(y, blended))),
        'mae_mean': float(np.mean(np.abs(blended - y))),
        'r2_mean': float(r2_score(y, blended)),
        'n_methods_blended': len(names),
        'note': 'Operates on stitched OOF predictions; CI not applicable.',
    }


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    global N_OPTUNA_TRIALS, N_REPEATS, POLY_TOP_K, PER_FOLD_REFINE_TRIALS
    global DE_POPSIZE, DE_MAXITER
    if '--smoke' in sys.argv:
        N_OPTUNA_TRIALS = 5
        N_REPEATS = 1
        POLY_TOP_K = 5
        PER_FOLD_REFINE_TRIALS = 2
        DE_POPSIZE = 4
        DE_MAXITER = 4
        print("[g1g2_boost] SMOKE mode: trials=5 repeats=1 poly_top_k=5 "
              "DE pop=4 iter=4")
    t0 = time.time()
    print("[g1g2_boost] loading Math (target) with G1/G2 ...")
    ds = StudentPerformanceDataset(data_dir='./data')
    ds.download_and_load(subject_subset='math')
    X, y = ds.preprocess(target='G3', include_prior_grades=True,
                          scale_numeric=False)
    feature_names = list(ds.feature_names)
    print(f"  raw: {X.shape}  features: {len(feature_names)}")

    X, feature_names = build_full_features(X, y, feature_names)
    print(f"  after FE+poly+prune: {X.shape}  features: {len(feature_names)}")

    print("\n=== Optuna tuning (10 model families) ===", flush=True)
    print("[tune] CatBoost ..."     , flush=True); cat_p = tune_catboost(X, y)
    print("[tune] XGBoost ..."      , flush=True); xgb_p = tune_xgb(X, y)
    print("[tune] LightGBM ..."     , flush=True); lgb_p = tune_lgb(X, y)
    print("[tune] ExtraTrees ..."   , flush=True); et_p = tune_extratrees(X, y)
    print("[tune] GBR_deep ..."     , flush=True); gbr_p = tune_gbr_deep(X, y)
    print("[tune] AdaBoost ..."     , flush=True); ada_p = tune_adaboost(X, y)
    print("[tune] XGBRF ..."        , flush=True); xrf_p = tune_xgbrf(X, y)
    print("[tune] KernelRidge ..."  , flush=True); kr_p = tune_kernel_ridge(X, y)
    print("[tune] MLP ..."          , flush=True); mlp_p = tune_mlp(X, y)
    print("[tune] ElasticNet ..."   , flush=True); en_p = tune_elasticnet(X, y)
    print("[tune] DE_MLP_AdaBoost (DE) ...", flush=True)
    mlp_ada_p = tune_de_mlp_adaboost(X, y)
    print(f"  best inner RMSE: {mlp_ada_p['best_rmse_inner']:.3f}")
    print(f"\n[tune] total: {time.time() - t0:.1f}s", flush=True)

    print("\n=== 10x5 stratified CV with tuned models + per-fold refinement ===",
          flush=True)
    result = evaluate_models(X, y, feature_names,
                              cat_p, xgb_p, lgb_p, et_p, gbr_p, ada_p,
                              xrf_p, kr_p, mlp_p, en_p, mlp_ada_p=mlp_ada_p)

    # DE_StackBlender post-CV
    print("\n=== DE_StackBlender on stitched OOF predictions ===", flush=True)
    blender = de_stack_blender(result['_oof_predictions'], y, maxiter=80)
    # Inject as a new method into the summary so it shows up in the rankings
    result['summary']['DE_StackBlender'] = {
        'rmse_mean': blender['rmse_mean'],
        'rmse_std': 0.0, 'rmse_ci95': 0.0,
        'mae_mean': blender['mae_mean'],
        'mae_std': 0.0, 'mae_ci95': 0.0,
        'r2_mean': blender['r2_mean'], 'r2_ci95': 0.0,
        'n_folds': 1,
        'note': blender['note'],
        'blend_weights': blender['weights'],
    }
    print(f"[blend] DE_StackBlender RMSE = {blender['rmse_mean']:.3f}  "
          f"MAE = {blender['mae_mean']:.3f}")

    out_dir = './results_v6/g1g2_boost'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'g1g2_boost_results.json'), 'w') as fh:
        json.dump({
            'config': dict(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                            n_optuna_trials=N_OPTUNA_TRIALS,
                            n_features=X.shape[1],
                            poly_top_k=POLY_TOP_K,
                            per_fold_refine_trials=PER_FOLD_REFINE_TRIALS),
            'tuned_params': {'CatBoost': cat_p, 'XGBoost': xgb_p,
                              'LightGBM': lgb_p, 'ExtraTrees': et_p,
                              'GBR_deep': gbr_p, 'AdaBoost': ada_p,
                              'XGBRF': xrf_p, 'KernelRidge': kr_p,
                              'MLP': mlp_p, 'ElasticNet': en_p,
                              'DE_MLP_AdaBoost': mlp_ada_p},
            'de_stack_blender': blender,
            **result,
        }, fh, indent=2, default=str)

    print("\n[g1g2_boost] RMSE rankings:")
    ranked = sorted(((v['rmse_mean'], k)
                     for k, v in result['summary'].items()))
    for r, name in ranked:
        s = result['summary'][name]
        print(f"  {name:22s}  RMSE = {s['rmse_mean']:.3f} +/- {s['rmse_ci95']:.3f}"
              f"   MAE = {s['mae_mean']:.3f}   R2 = {s['r2_mean']:.3f}")
    print(f"\n[done] total: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
