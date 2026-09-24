"""Cross-domain SHAP stability analysis.

For each domain (Portuguese, Math, xAPI), train a CatBoost classifier and
compute the per-feature mean |SHAP| ranking on the SHARED concept space.
Then compute pairwise Spearman rank correlations of those rankings to
quantify how stable risk-feature importance is under domain shift.

A stability matrix near 1.0 means the same concepts are flagged as
important regardless of domain (good for generalizable interventions).
Values near 0 mean each domain has its own idiosyncratic risk profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from scipy.stats import spearmanr

from concept_mapping import CONCEPT_FEATURES, to_concept_space


SEED = 42


@dataclass
class DomainShap:
    domain: str
    feature_importance: dict[str, float]   # feature -> mean |SHAP|
    rank_vector: np.ndarray                # feature ranks aligned to CONCEPT_FEATURES


def _train_concept_classifier(X: np.ndarray, y: np.ndarray) -> CatBoostClassifier:
    model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                               random_seed=SEED, verbose=False,
                               allow_writing_files=False)
    model.fit(X, y)
    return model


def compute_domain_shap(X: np.ndarray, y: np.ndarray, domain: str) -> DomainShap:
    model = _train_concept_classifier(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):       # binary case -> list of two arrays
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    importance = dict(zip(CONCEPT_FEATURES, mean_abs.tolist()))

    # Rank vector: higher importance -> higher rank
    order = np.argsort(-mean_abs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    rank_vector = ranks[np.argsort(np.arange(len(CONCEPT_FEATURES)))]
    return DomainShap(domain=domain, feature_importance=importance,
                      rank_vector=rank_vector.astype(int))


def _exact_spearman_null(n_feat: int) -> np.ndarray:
    """Spearman rho of the identity ranking against every one of the n!
    orderings (untied ranks, so rho = 1 - 6*sum(d^2)/(n(n^2-1)))."""
    from itertools import permutations
    base = np.arange(n_feat)
    perms = np.array(list(permutations(base)))
    d2 = ((perms - base) ** 2).sum(axis=1)
    return 1.0 - 6.0 * d2 / (n_feat * (n_feat ** 2 - 1))


def compute_stability_matrix(domains: list[DomainShap]
                             ) -> tuple[np.ndarray, np.ndarray, list[str],
                                        np.ndarray]:
    """Return (rho_matrix, two_sided_p_matrix, names, one_sided_p_matrix).

    p-values come from the exact permutation distribution: all n! orderings
    of one rank vector are enumerated (40,320 at n = 8), so there is no
    Monte-Carlo error. The smallest attainable two-sided p is 2/n! and the
    smallest one-sided p is 1/n!.
    """
    n = len(domains)
    mat = np.eye(n)
    pmat = np.zeros((n, n))
    pmat_one = np.zeros((n, n))
    names = [d.domain for d in domains]
    null = _exact_spearman_null(len(domains[0].rank_vector))
    for i in range(n):
        for j in range(i + 1, n):
            a = domains[i].rank_vector
            b = domains[j].rank_vector
            rho, _ = spearmanr(a, b)
            mat[i, j] = mat[j, i] = float(rho)
            p_two = float(np.mean(np.abs(null) >= abs(float(rho)) - 1e-12))
            p_one = float(np.mean(null >= float(rho) - 1e-12))
            pmat[i, j] = pmat[j, i] = p_two
            pmat_one[i, j] = pmat_one[j, i] = p_one
    return mat, pmat, names, pmat_one


def per_feature_stability(domains: list[DomainShap]) -> dict[str, float]:
    """For each concept feature, return the std of its rank across domains
    (low = stable importance; high = idiosyncratic).
    """
    rank_matrix = np.column_stack([d.rank_vector for d in domains])
    return {feat: float(np.std(rank_matrix[i]))
            for i, feat in enumerate(CONCEPT_FEATURES)}


def run_full_stability_analysis(out_dir: str | None = None) -> dict:
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base), 'data')

    df_por = pd.read_csv(os.path.join(data_dir, 'student-por.csv'), sep=';')
    df_mat = pd.read_csv(os.path.join(data_dir, 'student-mat.csv'), sep=';')
    df_xapi = pd.read_csv(os.environ.get('XAPI_CSV', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xAPI-Edu-Data.csv')))

    X_por, y_por = to_concept_space(df_por, 'uci')
    X_mat, y_mat = to_concept_space(df_mat, 'uci')
    X_xapi, y_xapi = to_concept_space(df_xapi, 'xapi')

    domains = [
        compute_domain_shap(X_por, y_por, 'Portuguese'),
        compute_domain_shap(X_mat, y_mat, 'Math'),
        compute_domain_shap(X_xapi, y_xapi, 'xAPI'),
    ]
    mat, pmat, names, pmat_one = compute_stability_matrix(domains)
    stability_per_feat = per_feature_stability(domains)

    result = {
        'domains': [d.domain for d in domains],
        'feature_importance': {d.domain: d.feature_importance for d in domains},
        'spearman_matrix': mat.tolist(),
        'spearman_p_matrix': pmat.tolist(),
        'spearman_p_one_sided_matrix': pmat_one.tolist(),
        'p_test': 'exact permutation (all n! orderings)',
        'matrix_labels': names,
        'per_feature_rank_std': stability_per_feat,
    }

    if out_dir is not None:
        import json
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'shap_stability.json'), 'w') as fh:
            json.dump(result, fh, indent=2)

    return result


def main():
    result = run_full_stability_analysis()
    print("Domains:", result['domains'])
    print("\nMean |SHAP| per feature:")
    for d in result['domains']:
        print(f"\n  {d}:")
        for feat, val in sorted(result['feature_importance'][d].items(),
                                key=lambda kv: -kv[1]):
            print(f"    {feat:25s} {val:.4f}")
    print("\nSpearman rank-correlation matrix (rho [perm-p]):")
    print(f"  {'':12s}", end='')
    for name in result['matrix_labels']:
        print(f"{name:>20s}", end='')
    print()
    for i, name in enumerate(result['matrix_labels']):
        print(f"  {name:12s}", end='')
        for j, v in enumerate(result['spearman_matrix'][i]):
            p = result['spearman_p_matrix'][i][j]
            cell = f"{v:+.3f}" + (f" [p={p:.3f}]" if i != j else "        ")
            print(f"{cell:>20s}", end='')
        print()
    print("\nPer-feature rank std (low = stable across domains):")
    for feat, std in sorted(result['per_feature_rank_std'].items(),
                            key=lambda kv: kv[1]):
        print(f"  {feat:25s}  {std:.3f}")


if __name__ == '__main__':
    main()
