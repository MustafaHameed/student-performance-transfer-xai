"""Analyses added for the VTSE revision (reviewer requests), no DiCE needed.

  1. SHAP rank stability with the exact permutation test (all 8! orderings),
     plus the existing bootstrap CIs carried over from results_v6.
  2. Sensitivity: SHAP stability of Portuguese-only students (no identity
     match in the Mathematics file) against Mathematics.
  3. Counterfactual diagnostics on the archived DiCE run (results_v6):
     the 80 sampled at-risk students are rebuilt, and the 14 without a
     counterfactual are compared with the 66 covered students.
  4. The k = 3 clustering of the archived counterfactual deltas.
  5. A real worked-example student: the medoid of the attendance archetype.

Writes results_v7/extras/*.json.

Run from the project root:  python src/revision_extras.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concept_mapping import CONCEPT_FEATURES, to_concept_space  # noqa: E402
from counterfactuals import (SEED, _build_classifier_dataframe,  # noqa: E402
                             _train_classifier)
from shap_stability import (_exact_spearman_null, compute_domain_shap,  # noqa: E402
                            run_full_stability_analysis)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RnD keeps results_v* at the project root; the public repository under results/.
RESULTS = next((d for d in (ROOT, os.path.join(ROOT, 'results'))
                if os.path.isdir(os.path.join(d, 'results_v6'))), ROOT)
DATA = os.path.join(ROOT, 'data')
V6 = os.path.join(RESULTS, 'results_v6')
# xAPI-Edu cannot be redistributed: $XAPI_CSV, else data/, else the RnD copy.
XAPI = os.environ.get('XAPI_CSV') or next(
    (p for p in (os.path.join(ROOT, 'data', 'xAPI-Edu-Data.csv'),
                 os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))),
                              '~DataSets', 'xAPI-Edu-Data.csv'))
     if os.path.exists(p)), os.path.join(ROOT, 'data', 'xAPI-Edu-Data.csv'))
OUT = os.path.join(RESULTS, 'results_v7', 'extras')

# Same identity key as train_v5_transfer.MERGE_KEYS (data/student-merge.R)
MERGE_KEYS = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
              'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']


def _keys(df: pd.DataFrame) -> np.ndarray:
    return df[MERGE_KEYS].astype(str).agg('|'.join, axis=1).to_numpy()


def _dump(name: str, obj: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as fh:
        json.dump(obj, fh, indent=2)
    print(f"  wrote {name}")


def shap_stability_exact() -> dict:
    res = run_full_stability_analysis(out_dir=None)
    with open(os.path.join(V6, 'shap_bootstrap.json')) as fh:
        res['bootstrap'] = json.load(fh)['pairs']
    null = _exact_spearman_null(len(CONCEPT_FEATURES))
    crit = float(np.sort(np.abs(null))[int(np.ceil(0.95 * len(null))) - 1])
    res['n_orderings'] = int(len(null))
    res['min_p_two_sided'] = 2.0 / len(null)
    res['min_p_one_sided'] = 1.0 / len(null)
    # smallest |rho| whose exact two-sided p is <= 0.05
    abs_vals = np.unique(np.round(np.abs(null), 10))  # 1e-9 tolerance below
    res['critical_abs_rho_alpha05'] = float(min(
        v for v in abs_vals if np.mean(np.abs(null) >= v - 1e-9) <= 0.05))
    res['abs_rho_95th_percentile_of_null'] = crit
    _dump('shap_stability_exact.json', res)
    return res


def shap_por_only_sensitivity() -> dict:
    por = pd.read_csv(os.path.join(DATA, 'student-por.csv'), sep=';')
    mat = pd.read_csv(os.path.join(DATA, 'student-mat.csv'), sep=';')
    por_only = por[~np.isin(_keys(por), _keys(mat))].reset_index(drop=True)
    X_p, y_p = to_concept_space(por_only, 'uci')
    X_m, y_m = to_concept_space(mat, 'uci')
    d_p = compute_domain_shap(X_p, y_p, 'Portuguese-only')
    d_m = compute_domain_shap(X_m, y_m, 'Math')
    from scipy.stats import spearmanr
    rho = float(spearmanr(d_p.rank_vector, d_m.rank_vector)[0])
    null = _exact_spearman_null(len(CONCEPT_FEATURES))
    out = {
        'n_portuguese_only': int(len(por_only)),
        'at_risk_rate_portuguese_only': float(np.mean(y_p)),
        'rho_por_only_vs_math': rho,
        'p_two_sided_exact': float(np.mean(np.abs(null) >= abs(rho) - 1e-12)),
        'feature_importance': {'Portuguese-only': d_p.feature_importance,
                               'Math': d_m.feature_importance},
    }
    _dump('shap_por_only_sensitivity.json', out)
    return out


def eda_stats() -> dict:
    """At-risk shares and per-concept Cohen's d behind the EDA figures,
    computed with the figure script's own loaders."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'eda_fig', os.path.join(ROOT, 'src', 'generate_eda_figures.py'))
    eda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda)
    frames = {'Portuguese': eda.load_uci(os.path.join(DATA, 'student-por.csv')),
              'Mathematics': eda.load_uci(os.path.join(DATA, 'student-mat.csv')),
              'xAPI-Edu': eda.load_xapi(XAPI)}
    out = {}
    for name, df in frames.items():
        out[name] = {
            'n': int(len(df)), 'n_at_risk': int(df['at_risk'].sum()),
            'at_risk_pct': float(100 * df['at_risk'].mean()),
            'cohens_d': {c: eda.cohens_d(df[c].values, df['at_risk'].values)
                         for c in eda.CONCEPTS},
        }
    _dump('eda_stats.json', out)
    return out


def identity_overlap() -> dict:
    """How far the two UCI course files describe the same students."""
    por = pd.read_csv(os.path.join(DATA, 'student-por.csv'), sep=';')
    mat = pd.read_csv(os.path.join(DATA, 'student-mat.csv'), sep=';')
    kp, km = _keys(por), _keys(mat)
    merged = mat.merge(por, on=MERGE_KEYS, suffixes=('_mat', '_por'))
    out = {
        'n_por': int(len(por)), 'n_mat': int(len(mat)),
        'merge_pairs_cortez': int(len(merged)),
        'mat_rows_with_por_match': int(np.isin(km, kp).sum()),
        'por_rows_with_mat_match': int(np.isin(kp, km).sum()),
        'por_only_rows': int((~np.isin(kp, km)).sum()),
        'mat_only_rows': int((~np.isin(km, kp)).sum()),
        'share_mat_with_match': float(np.isin(km, kp).mean()),
        'g3_corr_matched': float(merged[['G3_mat', 'G3_por']].corr().iloc[0, 1]),
    }
    _dump('identity_overlap.json', out)
    return out


def cf_diagnostics() -> dict:
    mat = pd.read_csv(os.path.join(DATA, 'student-mat.csv'), sep=';')
    df = _build_classifier_dataframe(mat)
    model, _ = _train_classifier(df)
    X_full = df.drop(columns=['at_risk'])
    proba = model.predict_proba(X_full)[:, 1]
    pred = model.predict(X_full).flatten().astype(int)
    at_risk_idx_all = np.where(pred == 1)[0]
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(at_risk_idx_all, size=80, replace=False) \
        if len(at_risk_idx_all) > 80 else at_risk_idx_all

    deltas = pd.read_csv(os.path.join(V6, 'counterfactual', 'cf_deltas.csv'))
    covered = set(deltas['_orig_idx'].astype(int))
    sampled_set = set(int(i) for i in sampled)
    consistent = covered.issubset(sampled_set)
    uncovered = sorted(sampled_set - covered)
    cov = sorted(covered)

    def profile(idx):
        sub = mat.iloc[idx]
        return {
            'n': int(len(idx)),
            'failures_mean': float(sub['failures'].mean()),
            'share_with_prior_failure': float((sub['failures'] > 0).mean()),
            'absences_mean': float(sub['absences'].mean()),
            'absences_median': float(sub['absences'].median()),
            'studytime_mean': float(sub['studytime'].mean()),
            'goout_mean': float(sub['goout'].mean()),
            'Walc_mean': float(sub['Walc'].mean()),
            'higher_no_share': float((sub['higher'] == 'no').mean()),
            'age_mean': float(sub['age'].mean()),
            'predicted_risk_mean': float(proba[idx].mean()),
            'actual_at_risk_share': float((sub['G3'] < 10).mean()),
            'already_at_floor_share': float(
                ((sub['absences'] == 0) & (sub['studytime'] == 4)).mean()),
        }

    out = {
        'n_predicted_at_risk_in_sample': int(len(at_risk_idx_all)),
        'n_sampled': int(len(sampled)),
        'archived_cover_idx_subset_of_rebuilt_sample': bool(consistent),
        'n_covered': len(cov),
        'n_uncovered': len(uncovered),
        'uncovered_idx': uncovered,
        'covered_profile': profile(cov) if consistent else None,
        'uncovered_profile': profile(uncovered) if consistent else None,
        'note': ('The archived run did not log why a student got no '
                 'counterfactual (DiCE returned none, an exception, or the '
                 '30 s per-student timeout); only the profiles are compared.'),
    }

    # k = 3 on the archived deltas, same preprocessing as generate_archetypes
    delta_cols = [c for c in deltas.columns if c not in ('_orig_idx', '_archetype')]
    Z = StandardScaler().fit_transform(deltas[delta_cols].fillna(0.0).values)
    sil = {}
    for k in range(2, 6):
        lbl = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(Z)
        sil[str(k)] = float(silhouette_score(Z, lbl))
    lbl3 = KMeans(n_clusters=3, random_state=SEED, n_init=20).fit_predict(Z)
    k3 = []
    for c in range(3):
        m = deltas[lbl3 == c]
        md = m[delta_cols].mean()
        top = sorted([(f, float(md[f])) for f in delta_cols if abs(md[f]) > 0.05],
                     key=lambda kv: -abs(kv[1]))[:3]
        k3.append({'n': int(len(m)), 'top_mean_deltas': top,
                   'from_k2_archetype': m['_archetype'].value_counts().to_dict()})
    out['silhouette_recomputed'] = sil
    out['k3_clusters'] = sorted(k3, key=lambda d: -d['n'])

    # Worked example: medoid of the attendance archetype (cluster_id 1)
    att = deltas[deltas['_archetype'] == 1]
    Za = Z[deltas['_archetype'].values == 1]
    medoid_pos = int(np.argmin(((Za - Za.mean(axis=0)) ** 2).sum(axis=1)))
    row = att.iloc[medoid_pos]
    i = int(row['_orig_idx'])
    stu = mat.iloc[i]
    changed = {c: float(row[c]) for c in delta_cols if abs(float(row[c])) > 1e-6}
    out['worked_example'] = {
        'orig_idx': i,
        'features': {k: (stu[k].item() if hasattr(stu[k], 'item') else stu[k])
                     for k in ['sex', 'age', 'absences', 'goout', 'studytime',
                               'freetime', 'Walc', 'Dalc', 'failures', 'Medu',
                               'Fedu', 'higher', 'G3']},
        'predicted_risk_probability': float(proba[i]),
        'counterfactual_changes': changed,
        'sparsity': int(len(changed)),
    }
    # Same comparison for every seeded DiCE run of src/cf_seed_sweep.py
    seeds_dir = os.path.join(RESULTS, 'results_v7', 'extras', 'cf_seeds')
    per_seed = {}
    if os.path.isdir(seeds_dir):
        for sd in sorted(os.listdir(seeds_dir), key=int):
            f = os.path.join(seeds_dir, sd, 'cf_deltas.csv')
            if not os.path.exists(f):
                continue
            cov_s = set(pd.read_csv(f)['_orig_idx'].astype(int))
            unc_s = sorted(sampled_set - cov_s)
            per_seed[sd] = {'n_uncovered': len(unc_s),
                            'uncovered_share_with_prior_failure':
                                float((mat.iloc[unc_s]['failures'] > 0).mean()),
                            'covered_share_with_prior_failure':
                                float((mat.iloc[sorted(cov_s)]['failures'] > 0).mean()),
                            'uncovered_absences_median':
                                float(mat.iloc[unc_s]['absences'].median()),
                            'uncovered_in_archived_uncovered':
                                len(set(unc_s) & set(uncovered))}
    out['per_seed_uncovered'] = per_seed
    _dump('cf_diagnostics.json', out)
    return out


def main():
    print("[extras] EDA statistics")
    e = eda_stats()
    print({k: (v['n_at_risk'], round(v['at_risk_pct'], 1),
               {c: round(d, 2) for c, d in v['cohens_d'].items()}) for k, v in e.items()})
    print("[extras] identity overlap")
    o = identity_overlap()
    print(f"  {o}")
    print("[extras] exact-permutation SHAP stability")
    s = shap_stability_exact()
    m = s['spearman_matrix']
    print(f"  rho Por~Math={m[0][1]:.3f} Por~xAPI={m[0][2]:.3f} Math~xAPI={m[1][2]:.3f}")
    print(f"  exact p two-sided: {np.round(np.array(s['spearman_p_matrix']), 4).tolist()}")
    print(f"  critical |rho| (alpha .05) = {s['critical_abs_rho_alpha05']:.3f}")
    print("[extras] Portuguese-only sensitivity")
    p = shap_por_only_sensitivity()
    print(f"  n={p['n_portuguese_only']} rho={p['rho_por_only_vs_math']:.3f} "
          f"p={p['p_two_sided_exact']:.3f}")
    print("[extras] counterfactual diagnostics")
    c = cf_diagnostics()
    print(f"  consistent={c['archived_cover_idx_subset_of_rebuilt_sample']} "
          f"covered={c['n_covered']} uncovered={c['n_uncovered']}")


if __name__ == '__main__':
    main()
