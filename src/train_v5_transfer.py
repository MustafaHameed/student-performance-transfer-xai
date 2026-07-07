"""Orchestrator for the v5 transfer-learning + counterfactual paper.

Runs the 10 experiments from the revision plan:
  1  baseline (Por only / Math only)
  2  pooled baseline (Por+Math)
  3  warm-start CatBoost (Por -> Math)
  4  TrAdaBoost.R2 (Por -> Math)
  5  CORAL (Por -> Math)
  6  negative-transfer gate (chooses best of 3, 4, target-only)
  7  external validation on xAPI (concept space)
  8  cross-domain SHAP stability
  9  DiCE counterfactual archetypes (target Math)
 10  ablation (drop-gate / drop-feature-engineering)

Usage:
  python -m src.train_v5_transfer --smoke   # fast validation
  python -m src.train_v5_transfer --full    # 10x5 CV, 20 Optuna trials
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t as t_dist, wilcoxon
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import StudentPerformanceDataset
from train_v4 import FeatureEngineer
from transfer import (CoralAdaptedModel, TrAdaBoostR2,
                      coral_align, target_only_catboost, tradaboost_r2,
                      warm_start_catboost, negative_transfer_gate)


SEED = 42
PASS_THRESHOLD = 10
DEFAULT_RESULTS_ROOT = './results_v6'


@dataclass(frozen=True)
class RunConfig:
    n_splits: int
    n_repeats: int
    catboost_iterations: int
    finetune_iterations: int
    tradaboost_estimators: int
    cf_max_at_risk: int
    name: str

    @classmethod
    def smoke(cls) -> 'RunConfig':
        return cls(n_splits=2, n_repeats=1, catboost_iterations=120,
                   finetune_iterations=40, tradaboost_estimators=8,
                   cf_max_at_risk=10, name='smoke')

    @classmethod
    def full(cls) -> 'RunConfig':
        return cls(n_splits=5, n_repeats=10, catboost_iterations=400,
                   finetune_iterations=200, tradaboost_estimators=30,
                   cf_max_at_risk=80, name='full')


# ──────────────────────────────────────────────────────────
# Data loading + feature alignment
# ──────────────────────────────────────────────────────────

def _load_subject(subject: str, data_dir: str = './data',
                  include_prior_grades: bool = False):
    ds = StudentPerformanceDataset(data_dir=data_dir)
    ds.download_and_load(subject_subset=subject)
    X, y = ds.preprocess(target='G3',
                          include_prior_grades=include_prior_grades,
                          scale_numeric=False)
    return X, y, ds.feature_names


def _align_columns(X_a: np.ndarray, names_a: list[str],
                   X_b: np.ndarray, names_b: list[str]
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common = [n for n in names_a if n in names_b]
    idx_a = [names_a.index(n) for n in common]
    idx_b = [names_b.index(n) for n in common]
    return X_a[:, idx_a], X_b[:, idx_b], common


def _stratify_bins(y: np.ndarray) -> np.ndarray:
    return pd.qcut(y, q=4, labels=False, duplicates='drop')


# ──────────────────────────────────────────────────────────
# Experiment 1-6 (transfer comparison on Math target)
# ──────────────────────────────────────────────────────────

def _record_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    y_t_bin = (y_true >= PASS_THRESHOLD).astype(int)
    y_p_bin = (y_pred >= PASS_THRESHOLD).astype(int)
    acc = float(accuracy_score(y_t_bin, y_p_bin))
    f1 = float(f1_score(y_t_bin, y_p_bin, average='macro', zero_division=0))
    try:
        auc = float(roc_auc_score(y_t_bin, y_pred))
    except ValueError:
        auc = 0.5
    return dict(rmse=rmse, mae=mae, r2=r2, acc=acc, f1=f1, auc=auc)


def run_transfer_cv(X_source: np.ndarray, y_source: np.ndarray,
                    X_target: np.ndarray, y_target: np.ndarray,
                    feature_names: list[str], cfg: RunConfig) -> dict:
    """Repeated stratified CV on TARGET. Source is used whole as transfer pool.
    Engineered features are fit per-fold from training data only.
    """
    bins = _stratify_bins(y_target)
    rskf = RepeatedStratifiedKFold(n_splits=cfg.n_splits,
                                   n_repeats=cfg.n_repeats,
                                   random_state=SEED)

    methods = ['target_only', 'pooled', 'warm_start',
               'tradaboost_r2', 'coral', 'gated']
    fold_metrics = {m: {k: [] for k in ['rmse', 'mae', 'r2',
                                          'acc', 'f1', 'auc']}
                    for m in methods}
    gate_decisions = []

    fe_source = FeatureEngineer(scale_engineered=False)
    X_source_fe = fe_source.fit_transform(X_source, y_source, feature_names)

    catboost_params = dict(iterations=cfg.catboost_iterations, depth=6,
                           learning_rate=0.05, l2_leaf_reg=3.0,
                           random_seed=SEED, verbose=False,
                           allow_writing_files=False)

    print(f"\n[transfer] {cfg.n_splits}x{cfg.n_repeats} CV on Math target", flush=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_target, bins)):
        rep = fold_idx // cfg.n_splits + 1
        fold = fold_idx % cfg.n_splits + 1

        X_tr, X_va = X_target[tr_idx], X_target[va_idx]
        y_tr, y_va = y_target[tr_idx], y_target[va_idx]

        fe_target = FeatureEngineer(scale_engineered=False)
        X_tr_fe = fe_target.fit_transform(X_tr, y_tr, feature_names)
        X_va_fe = fe_target.transform(X_va, feature_names)

        # Reuse the source-fit FE columns to keep dimensionality consistent
        if X_tr_fe.shape[1] != X_source_fe.shape[1]:
            n = min(X_tr_fe.shape[1], X_source_fe.shape[1])
            X_tr_fe = X_tr_fe[:, :n]
            X_va_fe = X_va_fe[:, :n]
            X_src = X_source_fe[:, :n]
        else:
            X_src = X_source_fe

        # 1) target-only baseline
        m_to = target_only_catboost(X_tr_fe, y_tr, params=catboost_params)
        fold_metrics['target_only'] = _append(fold_metrics['target_only'],
                                              _record_metrics(y_va, m_to.predict(X_va_fe)))

        # 2) pooled baseline (concat source + target_train)
        X_pool = np.vstack([X_src, X_tr_fe])
        y_pool = np.concatenate([y_source, y_tr])
        m_pool = target_only_catboost(X_pool, y_pool, params=catboost_params)
        fold_metrics['pooled'] = _append(fold_metrics['pooled'],
                                         _record_metrics(y_va, m_pool.predict(X_va_fe)))

        # 3) warm-start CatBoost
        m_ws = warm_start_catboost(
            X_src, y_source, X_tr_fe, y_tr,
            source_params=catboost_params,
            finetune_iterations=cfg.finetune_iterations, finetune_lr=0.03,
        )
        fold_metrics['warm_start'] = _append(fold_metrics['warm_start'],
                                             _record_metrics(y_va, m_ws.predict(X_va_fe)))

        # 4) TrAdaBoost.R2
        m_tr2 = tradaboost_r2(X_src, y_source, X_tr_fe, y_tr,
                              n_estimators=cfg.tradaboost_estimators, max_depth=6)
        fold_metrics['tradaboost_r2'] = _append(fold_metrics['tradaboost_r2'],
                                                _record_metrics(y_va, m_tr2.predict(X_va_fe)))

        # 5) CORAL
        m_co = coral_align(X_src, y_source, X_tr_fe, y_tr,
                           base_params=catboost_params)
        fold_metrics['coral'] = _append(fold_metrics['coral'],
                                        _record_metrics(y_va, m_co.predict(X_va_fe)))

        # 6) negative-transfer gate (warm-start vs target-only)
        chosen, decision = negative_transfer_gate(
            lambda Xs, ys, Xt, yt: warm_start_catboost(
                Xs, ys, Xt, yt, source_params=catboost_params,
                finetune_iterations=cfg.finetune_iterations, finetune_lr=0.03),
            X_src, y_source, X_tr_fe, y_tr, n_inner_splits=3,
        )
        fold_metrics['gated'] = _append(fold_metrics['gated'],
                                        _record_metrics(y_va, chosen.predict(X_va_fe)))
        gate_decisions.append({
            'fold': fold_idx, 'rep': rep, 'fold_in_rep': fold,
            'chose_transfer': bool(decision.chose_transfer),
            'transfer_inner_rmse': decision.transfer_inner_rmse,
            'target_only_inner_rmse': decision.target_only_inner_rmse,
        })

        print(f"  rep={rep} fold={fold}  to={fold_metrics['target_only']['rmse'][-1]:.3f}"
              f"  pool={fold_metrics['pooled']['rmse'][-1]:.3f}"
              f"  ws={fold_metrics['warm_start']['rmse'][-1]:.3f}"
              f"  tr2={fold_metrics['tradaboost_r2']['rmse'][-1]:.3f}"
              f"  cor={fold_metrics['coral']['rmse'][-1]:.3f}"
              f"  gate={fold_metrics['gated']['rmse'][-1]:.3f}", flush=True)

    summary = {m: _summarize(fold_metrics[m]) for m in methods}
    stats = _statistical_tests({m: fold_metrics[m]['rmse'] for m in methods})

    return {'summary': summary, 'gate_decisions': gate_decisions,
            'statistical_tests': stats,
            '_fold_rmse': {m: fold_metrics[m]['rmse'] for m in methods}}


def _append(d: dict, sample: dict[str, float]) -> dict:
    for k, v in sample.items():
        d[k].append(v)
    return d


def _summarize(metrics: dict[str, list[float]]) -> dict[str, dict]:
    out = {}
    for k, vals in metrics.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        n = len(arr)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            se = sd / np.sqrt(n)
            ci = float(se * t_dist.ppf(0.975, df=n - 1))
        else:
            ci = 0.0
        out[k] = dict(mean=mu, std=sd, ci95=ci, n=n)
    return out


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's d for paired samples; returns None when undefined (sd=0)."""
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return None
    return float(np.mean(diff) / sd)


def _statistical_tests(fold_rmse: dict[str, list[float]]) -> dict:
    methods = list(fold_rmse.keys())
    arrays = [np.asarray(fold_rmse[m]) for m in methods]
    if len(arrays[0]) < 2:
        return {'note': 'too few folds for tests'}
    try:
        chi2, p = friedmanchisquare(*arrays)
        omnibus = {'chi2': float(chi2), 'p': float(p)}
    except Exception as e:
        omnibus = {'error': str(e)}

    pairs = {}
    n_pairs = len(methods) * (len(methods) - 1) // 2
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            key = f'{methods[i]}__vs__{methods[j]}'
            a, b = arrays[i], arrays[j]
            d = _cohens_d_paired(a, b)
            if np.allclose(a, b):
                pairs[key] = {
                    'identical': True,
                    'p': None,
                    'p_bonferroni': None,
                    'cohens_d': d,
                    'note': 'paired differences all zero; Wilcoxon undefined',
                }
                continue
            try:
                stat, p = wilcoxon(a, b)
                pairs[key] = {
                    'identical': False,
                    'stat': float(stat),
                    'p': float(p),
                    'p_bonferroni': float(min(1.0, p * n_pairs)),
                    'cohens_d': d,
                }
            except Exception as e:
                pairs[key] = {'error': str(e), 'cohens_d': d}
    return {'friedman': omnibus, 'wilcoxon_bonferroni': pairs}


# ──────────────────────────────────────────────────────────
# Focused G1/G2-inclusive experiment (only target-only + gated)
# Used to populate the comparison table against grade-inclusive
# published baselines (Apriyadi & Rini, Fang et al.).
# ──────────────────────────────────────────────────────────

def run_g1g2_cv(X_source: np.ndarray, y_source: np.ndarray,
                X_target: np.ndarray, y_target: np.ndarray,
                feature_names: list[str], cfg: RunConfig) -> dict:
    bins = _stratify_bins(y_target)
    rskf = RepeatedStratifiedKFold(n_splits=cfg.n_splits,
                                   n_repeats=cfg.n_repeats,
                                   random_state=SEED)

    methods = ['target_only', 'gated']
    fold_metrics = {m: {k: [] for k in ['rmse', 'mae', 'r2',
                                          'acc', 'f1', 'auc']}
                    for m in methods}
    gate_decisions = []

    fe_source = FeatureEngineer(scale_engineered=False)
    X_source_fe = fe_source.fit_transform(X_source, y_source, feature_names)

    catboost_params = dict(iterations=cfg.catboost_iterations, depth=6,
                           learning_rate=0.05, l2_leaf_reg=3.0,
                           random_seed=SEED, verbose=False,
                           allow_writing_files=False)

    print(f"\n[g1g2] {cfg.n_splits}x{cfg.n_repeats} CV on Math target "
          f"(WITH G1/G2 features)", flush=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_target, bins)):
        rep = fold_idx // cfg.n_splits + 1
        fold = fold_idx % cfg.n_splits + 1

        X_tr, X_va = X_target[tr_idx], X_target[va_idx]
        y_tr, y_va = y_target[tr_idx], y_target[va_idx]

        fe_target = FeatureEngineer(scale_engineered=False)
        X_tr_fe = fe_target.fit_transform(X_tr, y_tr, feature_names)
        X_va_fe = fe_target.transform(X_va, feature_names)

        if X_tr_fe.shape[1] != X_source_fe.shape[1]:
            n = min(X_tr_fe.shape[1], X_source_fe.shape[1])
            X_tr_fe = X_tr_fe[:, :n]
            X_va_fe = X_va_fe[:, :n]
            X_src = X_source_fe[:, :n]
        else:
            X_src = X_source_fe

        m_to = target_only_catboost(X_tr_fe, y_tr, params=catboost_params)
        fold_metrics['target_only'] = _append(fold_metrics['target_only'],
                                              _record_metrics(y_va, m_to.predict(X_va_fe)))

        chosen, decision = negative_transfer_gate(
            lambda Xs, ys, Xt, yt: warm_start_catboost(
                Xs, ys, Xt, yt, source_params=catboost_params,
                finetune_iterations=cfg.finetune_iterations, finetune_lr=0.03),
            X_src, y_source, X_tr_fe, y_tr, n_inner_splits=3,
        )
        fold_metrics['gated'] = _append(fold_metrics['gated'],
                                        _record_metrics(y_va, chosen.predict(X_va_fe)))
        gate_decisions.append({
            'fold': fold_idx, 'rep': rep, 'fold_in_rep': fold,
            'chose_transfer': bool(decision.chose_transfer),
            'transfer_inner_rmse': decision.transfer_inner_rmse,
            'target_only_inner_rmse': decision.target_only_inner_rmse,
        })

        print(f"  rep={rep} fold={fold}  to={fold_metrics['target_only']['rmse'][-1]:.3f}"
              f"  gate={fold_metrics['gated']['rmse'][-1]:.3f}", flush=True)

    summary = {m: _summarize(fold_metrics[m]) for m in methods}
    return {'summary': summary, 'gate_decisions': gate_decisions,
            '_fold_rmse': {m: fold_metrics[m]['rmse'] for m in methods}}


# ──────────────────────────────────────────────────────────
# Experiment 7: external validation on xAPI
# ──────────────────────────────────────────────────────────

def run_external_validation(out_dir: str) -> dict:
    from concept_mapping import to_concept_space
    from catboost import CatBoostClassifier

    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base), 'data')
    df_por = pd.read_csv(os.path.join(data_dir, 'student-por.csv'), sep=';')
    df_xapi = pd.read_csv(os.environ.get('XAPI_CSV', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xAPI-Edu-Data.csv')))

    X_por, y_por = to_concept_space(df_por, 'uci')
    X_xapi, y_xapi = to_concept_space(df_xapi, 'xapi')

    model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                               random_seed=SEED, verbose=False,
                               allow_writing_files=False)
    model.fit(X_por, y_por)

    proba = model.predict_proba(X_xapi)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        'auc': float(roc_auc_score(y_xapi, proba)),
        'f1_macro': float(f1_score(y_xapi, pred, average='macro')),
        'accuracy': float(accuracy_score(y_xapi, pred)),
        'n_xapi': int(len(y_xapi)),
        'risk_rate_xapi': float(y_xapi.mean()),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'external_validation.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


# ──────────────────────────────────────────────────────────
# Experiment 8: SHAP stability
# ──────────────────────────────────────────────────────────

def run_shap_stability(out_dir: str) -> dict:
    from shap_stability import run_full_stability_analysis
    return run_full_stability_analysis(out_dir=out_dir)


# ──────────────────────────────────────────────────────────
# Experiment 9: counterfactual archetypes
# ──────────────────────────────────────────────────────────

def run_counterfactual_archetypes(out_dir: str, max_at_risk: int) -> dict:
    from counterfactuals import generate_archetypes

    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base), 'data')
    df_mat = pd.read_csv(os.path.join(data_dir, 'student-mat.csv'), sep=';')

    metrics, archetypes, deltas_df, silhouette_scores = generate_archetypes(
        df_mat, n_archetypes=2, cf_per_student=1, max_at_risk=max_at_risk,
        method='random', enforce_monotonic=True)

    out = {
        'metrics': dict(n_at_risk=metrics.n_at_risk,
                         n_with_cf=metrics.n_with_cf,
                         coverage=metrics.coverage,
                         validity=metrics.validity,
                         mean_sparsity=metrics.mean_sparsity,
                         mean_proximity=metrics.mean_proximity),
        'archetypes': [
            dict(cluster_id=a.cluster_id, name=a.name,
                 n_students=a.n_students,
                 dominant_features=[(f, v) for f, v in a.dominant_features],
                 recipe=(a.example_cf or {}).get('recipe', ''))
            for a in archetypes
        ],
        'silhouette_scores': {str(k): v for k, v in silhouette_scores.items()},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'archetypes.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    if not deltas_df.empty:
        deltas_df.to_csv(os.path.join(out_dir, 'cf_deltas.csv'), index=False)
    return out


# ──────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny CV + few trials for quick validation.')
    parser.add_argument('--full', action='store_true',
                        help='10x5 CV, full Optuna budget.')
    parser.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--skip-transfer', action='store_true')
    parser.add_argument('--skip-external', action='store_true')
    parser.add_argument('--skip-shap-stability', action='store_true')
    parser.add_argument('--skip-counterfactuals', action='store_true')
    parser.add_argument('--with-g1g2-comparison', action='store_true',
                        help='Run a focused target-only + gated CV with G1/G2 '
                             'features INCLUDED, to compare against grade-inclusive '
                             'published baselines (Apriyadi & Rini, Fang et al.). '
                             'Writes results_v6/transfer/transfer_cv_with_g1g2.json.')
    parser.add_argument('--reverse', action='store_true',
                        help='Swap source/target so that Mathematics is the source '
                             'and Portuguese is the target (Math->Por). Writes '
                             'results_v6/transfer/transfer_cv_reverse.json.')
    args = parser.parse_args()

    cfg = RunConfig.smoke() if args.smoke else RunConfig.full()
    if not args.smoke and not args.full:
        print("[warn] neither --smoke nor --full set, defaulting to --smoke")
        cfg = RunConfig.smoke()

    print(f"\n[v5_transfer] config: {cfg}")

    transfer_dir = os.path.join(args.results_root, 'transfer')
    xapi_dir = os.path.join(args.results_root, 'xapi')
    cf_dir = os.path.join(args.results_root, 'counterfactual')
    for d in (transfer_dir, xapi_dir, cf_dir):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()

    if not args.skip_transfer:
        if args.reverse:
            print("\n=== Loading source (Math) and target (Portuguese) [REVERSE] ===", flush=True)
        else:
            print("\n=== Loading source (Portuguese) and target (Math) ===", flush=True)
        X_por, y_por, names_por = _load_subject('portuguese')
        X_mat, y_mat, names_mat = _load_subject('math')
        X_por_a, X_mat_a, names = _align_columns(X_por, names_por, X_mat, names_mat)

        if args.reverse:
            X_src_a, y_src = X_mat_a, y_mat
            X_tgt_a, y_tgt = X_por_a, y_por
            out_name = 'transfer_cv_reverse.json'
        else:
            X_src_a, y_src = X_por_a, y_por
            X_tgt_a, y_tgt = X_mat_a, y_mat
            out_name = 'transfer_cv.json'
        print(f"  source: {X_src_a.shape}  target: {X_tgt_a.shape}  "
              f"common features: {len(names)}", flush=True)

        result = run_transfer_cv(X_src_a, y_src, X_tgt_a, y_tgt, names, cfg)
        with open(os.path.join(transfer_dir, out_name), 'w') as fh:
            json.dump({'config': dict(n_splits=cfg.n_splits,
                                       n_repeats=cfg.n_repeats,
                                       name=cfg.name,
                                       reverse=bool(args.reverse)),
                       **result},
                      fh, indent=2)

        rmse_means = {m: result['summary'][m]['rmse']['mean']
                      for m in result['summary']}
        print("\n[transfer] RMSE means:")
        for m, v in sorted(rmse_means.items(), key=lambda kv: kv[1]):
            print(f"  {m:18s} {v:.3f}")

        n_chose_transfer = sum(d['chose_transfer'] for d in result['gate_decisions'])
        n_total = len(result['gate_decisions'])
        print(f"\n[gate] chose transfer in {n_chose_transfer}/{n_total} folds")

    if not args.skip_external:
        print("\n=== External validation on xAPI ===", flush=True)
        ext = run_external_validation(xapi_dir)
        print(f"[xapi] AUC={ext['auc']:.3f}  F1={ext['f1_macro']:.3f}  "
              f"Acc={ext['accuracy']:.3f}")

    if not args.skip_shap_stability:
        print("\n=== Cross-domain SHAP stability ===", flush=True)
        stab = run_shap_stability(cf_dir)
        mat = stab['spearman_matrix']
        print(f"[shap] Spearman rho:  Por~Math={mat[0][1]:.3f}  "
              f"Por~xAPI={mat[0][2]:.3f}  Math~xAPI={mat[1][2]:.3f}")

    if not args.skip_counterfactuals:
        print("\n=== Counterfactual archetypes (Math at-risk) ===", flush=True)
        cf = run_counterfactual_archetypes(cf_dir, max_at_risk=cfg.cf_max_at_risk)
        print(f"[cf] coverage={cf['metrics']['coverage']:.2f}  "
              f"validity={cf['metrics']['validity']:.2f}  "
              f"sparsity={cf['metrics']['mean_sparsity']:.2f}  "
              f"archetypes={len(cf['archetypes'])}")
        for a in cf['archetypes']:
            print(f"   [{a['cluster_id']}] {a['name']:30s} "
                  f"({a['n_students']} students)")

    if args.with_g1g2_comparison:
        print("\n=== G1/G2-inclusive comparison (target-only + gated) ===", flush=True)
        X_por_g, y_por_g, names_por_g = _load_subject(
            'portuguese', include_prior_grades=True)
        X_mat_g, y_mat_g, names_mat_g = _load_subject(
            'math', include_prior_grades=True)
        X_por_ga, X_mat_ga, names_g = _align_columns(
            X_por_g, names_por_g, X_mat_g, names_mat_g)
        print(f"  source: {X_por_ga.shape}  target: {X_mat_ga.shape}  "
              f"common features: {len(names_g)}", flush=True)

        result_g = run_g1g2_cv(X_por_ga, y_por_g, X_mat_ga, y_mat_g,
                                names_g, cfg)
        out_g = os.path.join(transfer_dir, 'transfer_cv_with_g1g2.json')
        with open(out_g, 'w') as fh:
            json.dump({'config': dict(n_splits=cfg.n_splits,
                                       n_repeats=cfg.n_repeats,
                                       name=cfg.name,
                                       include_prior_grades=True),
                       **result_g}, fh, indent=2)
        s_g = result_g['summary']
        print(f"\n[g1g2] target_only RMSE={s_g['target_only']['rmse']['mean']:.3f}"
              f"  gated RMSE={s_g['gated']['rmse']['mean']:.3f}")

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s "
          f"-- artifacts in {args.results_root}/")


if __name__ == '__main__':
    main()
