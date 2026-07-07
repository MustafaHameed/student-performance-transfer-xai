"""DiCE counterfactual archetypes for at-risk student intervention.

For each predicted at-risk student in the target (Math) test set, generate
the minimal feature changes that flip the prediction from "at-risk" to
"pass". Counterfactuals are constrained to MUTABLE features (study time,
absences, going out, etc.) and forbidden from changing IMMUTABLE features
(sex, age, parental jobs, school).

The Δ-vectors (counterfactual minus original) are then clustered into k=5
intervention archetypes, each of which is auto-named by its dominant Δ
direction. Validity, sparsity, and proximity metrics are reported.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

SEED = 42

# UCI raw column mutability split. studytime / freetime / goout / Dalc /
# Walc / absences / romantic / internet / paid / activities are all
# plausibly intervenable; demographics, family, prior-failure, school,
# parental-job features are not.
MUTABLE_FEATURES = [
    'studytime', 'freetime', 'goout', 'Dalc', 'Walc', 'absences',
    'romantic', 'internet', 'paid', 'activities', 'higher',
]
IMMUTABLE_FEATURES = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'failures', 'schoolsup', 'famsup', 'nursery',
    'famrel', 'health',
]

# Direction-of-improvement constraints. 'up' = a realistic intervention
# can only INCREASE this feature (e.g. raise study time). 'down' = can
# only DECREASE (e.g. cut absences, reduce alcohol). Categorical/binary
# features without a clear direction are left unconstrained.
MONOTONIC_DIRECTION = {
    'studytime': 'up',
    'absences': 'down',
    'goout': 'down',
    'Dalc': 'down',
    'Walc': 'down',
    'freetime': 'down',
}

# Map every mutable Δ feature onto an educationally-meaningful intervention
# category. Cluster names are derived by majority-vote over the absolute
# Δ contributions in each cluster.
ARCHETYPE_CATEGORY_MAP = {
    'absences': 'Attendance-focused',
    'goout': 'Lifestyle-moderation',
    'Dalc': 'Lifestyle-moderation',
    'Walc': 'Lifestyle-moderation',
    'freetime': 'Lifestyle-moderation',
    'romantic_changed': 'Lifestyle-moderation',
    'studytime': 'Study-intensity',
    'paid_changed': 'Study-intensity',
    'internet_changed': 'Study-intensity',
    'higher_changed': 'Study-intensity',
    'activities_changed': 'Study-intensity',
}


@dataclass
class CounterfactualMetrics:
    n_at_risk: int
    n_with_cf: int
    coverage: float          # fraction of at-risk students with >= 1 valid CF
    validity: float          # fraction of generated CFs that actually flip
    mean_sparsity: float     # mean number of features changed per CF
    mean_proximity: float    # mean L1 distance per CF (normalized)


@dataclass
class Archetype:
    cluster_id: int
    name: str
    n_students: int
    dominant_features: list[tuple[str, float]]   # (feat_name, mean_delta)
    example_cf: dict | None                       # example counterfactual


def _build_classifier_dataframe(df_raw: pd.DataFrame,
                                pass_threshold: int = 10) -> pd.DataFrame:
    """Drop G1/G2/G3 (and any subject indicator), add binary at-risk target."""
    df = df_raw.copy()
    df['at_risk'] = (df['G3'].astype(float) < pass_threshold).astype(int)
    drop = [c for c in ['G1', 'G2', 'G3', 'subject'] if c in df.columns]
    return df.drop(columns=drop)


def _train_classifier(df: pd.DataFrame, target: str = 'at_risk'):
    from catboost import CatBoostClassifier
    cat_features = [c for c in df.columns
                    if c != target and df[c].dtype == object]
    X = df.drop(columns=[target])
    y = df[target].values
    model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                               cat_features=cat_features, random_seed=SEED,
                               verbose=False, allow_writing_files=False)
    model.fit(X, y)
    return model, cat_features


def _monotonic_permitted_range(query_row: pd.Series,
                               continuous_features: list[str],
                               feature_min_max: dict[str, tuple[float, float]]
                               ) -> dict[str, list[float]]:
    """For each monotonically-constrained feature, build a [low, high]
    range that only allows movement in the realistic-intervention direction
    relative to the student's current value.
    """
    permitted = {}
    for feat, direction in MONOTONIC_DIRECTION.items():
        if feat not in continuous_features:
            continue
        cur = float(query_row[feat])
        lo_global, hi_global = feature_min_max[feat]
        if direction == 'up':
            permitted[feat] = [cur, hi_global]
        else:
            permitted[feat] = [lo_global, cur]
    return permitted


def _categorize_archetype(mean_deltas: pd.Series, delta_cols: list[str]) -> str:
    """Sum |mean Δ| within each ARCHETYPE_CATEGORY_MAP category. The
    category that captures the largest absolute mass becomes the cluster name.
    Falls back to 'Mixed-intervention' if no mapped feature contributes."""
    weights: dict[str, float] = {}
    for c in delta_cols:
        cat = ARCHETYPE_CATEGORY_MAP.get(c)
        if cat is None:
            continue
        v = float(mean_deltas[c])
        if abs(v) < 1e-6:
            continue
        weights[cat] = weights.get(cat, 0.0) + abs(v)
    if not weights:
        return 'Mixed-intervention'
    return max(weights, key=weights.get)


def _format_recipe(top_features: list[tuple[str, float]]) -> str:
    """Render the top-3 Δ features as a short human-readable intervention
    recipe (e.g. 'reduce absences by 5.9 days; reduce going-out by 1.2'). """
    pretty = {
        'absences': ('absences', 'days'),
        'studytime': ('study time', 'levels'),
        'goout': ('going-out', 'levels'),
        'Dalc': ('weekday alcohol', 'levels'),
        'Walc': ('weekend alcohol', 'levels'),
        'freetime': ('free time', 'levels'),
    }
    parts = []
    for feat, val in top_features:
        if feat.endswith('_changed'):
            parts.append(f"toggle {feat[:-len('_changed')]}")
            continue
        label, unit = pretty.get(feat, (feat, ''))
        verb = 'reduce' if val < 0 else 'raise'
        parts.append(f"{verb} {label} by {abs(val):.1f}{(' ' + unit) if unit else ''}")
    return '; '.join(parts) if parts else 'no dominant change'


def generate_archetypes(df_target_raw: pd.DataFrame,
                        n_archetypes: int = 3,
                        cf_per_student: int = 1,
                        max_at_risk: int | None = None,
                        method: str = 'genetic',
                        enforce_monotonic: bool = True,
                        kmeans_n_init: int = 20
                        ) -> tuple[CounterfactualMetrics,
                                   list[Archetype],
                                   pd.DataFrame]:
    """Train a classifier on the target, generate counterfactuals for
    predicted-at-risk students, cluster Δ-vectors into archetypes.

    method: DiCE backend ('genetic' = realistic but slower; 'random' = fast
            but produces non-monotonic CFs).
    enforce_monotonic: if True, restrict each constrained feature to its
            realistic-intervention direction per query.

    Returns (metrics, archetypes, deltas_df).
    """
    import dice_ml

    df = _build_classifier_dataframe(df_target_raw)
    model, cat_features = _train_classifier(df)

    feature_names = [c for c in df.columns if c != 'at_risk']
    continuous_features = [c for c in feature_names if c not in cat_features]

    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features,
                     outcome_name='at_risk')
    m = dice_ml.Model(model=model, backend='sklearn',
                      model_type='classifier')
    explainer = dice_ml.Dice(d, m, method=method)

    X_full = df.drop(columns=['at_risk'])
    preds = model.predict(X_full)
    at_risk_mask = preds.flatten() == 1
    at_risk_idx = np.where(at_risk_mask)[0]

    if max_at_risk is not None and len(at_risk_idx) > max_at_risk:
        rng = np.random.default_rng(SEED)
        at_risk_idx = rng.choice(at_risk_idx, size=max_at_risk, replace=False)

    actually_immutable = [c for c in IMMUTABLE_FEATURES if c in feature_names]

    deltas = []
    n_with_cf = 0
    n_valid = 0
    n_attempted = 0
    sparsities = []
    proximities = []

    feature_min_max = {c: (float(X_full[c].astype(float).min()),
                           float(X_full[c].astype(float).max()))
                       for c in continuous_features}
    feature_ranges = {c: lo_hi[1] - lo_hi[0]
                      for c, lo_hi in feature_min_max.items()}

    cf_examples_per_cluster: dict[int, dict] = {}

    features_to_vary = [c for c in MUTABLE_FEATURES if c in feature_names]

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
    import time as _time

    PER_STUDENT_TIMEOUT_S = 30  # cap per-student CF search to avoid hangs

    def _gen_for(query, permitted):
        return explainer.generate_counterfactuals(
            query, total_CFs=cf_per_student, desired_class=0,
            features_to_vary=features_to_vary,
            permitted_range=permitted, verbose=False)

    n_timeouts = 0
    for i in at_risk_idx:
        query = X_full.iloc[[i]]
        n_attempted += 1

        permitted = _monotonic_permitted_range(query.iloc[0],
                                                continuous_features,
                                                feature_min_max) \
            if enforce_monotonic else None

        t0 = _time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_gen_for, query, permitted)
                try:
                    res = fut.result(timeout=PER_STUDENT_TIMEOUT_S)
                except FTimeout:
                    n_timeouts += 1
                    fut.cancel()
                    continue
        except Exception:
            continue

        cf_df = res.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            continue

        cf_row = cf_df.iloc[0]
        if cf_row['at_risk'] != 0:
            continue

        n_valid += 1
        n_with_cf += 1

        delta = {}
        sparsity = 0
        proximity = 0.0
        for c in continuous_features:
            orig = float(query[c].iloc[0])
            new = float(cf_row[c])
            d_v = new - orig
            delta[c] = d_v
            if abs(d_v) > 1e-6:
                sparsity += 1
                proximity += abs(d_v) / max(feature_ranges[c], 1e-6)
        for c in cat_features:
            orig = str(query[c].iloc[0])
            new = str(cf_row[c])
            delta[c + '_changed'] = 1.0 if orig != new else 0.0
            if orig != new:
                sparsity += 1
                proximity += 1.0
        delta['_orig_idx'] = i
        deltas.append(delta)
        sparsities.append(sparsity)
        proximities.append(proximity)

    if not deltas:
        return (CounterfactualMetrics(0, 0, 0.0, 0.0, 0.0, 0.0), [],
                pd.DataFrame())

    deltas_df = pd.DataFrame(deltas)
    delta_cols = [c for c in deltas_df.columns if c != '_orig_idx']
    cluster_input = deltas_df[delta_cols].fillna(0.0).values
    scaler = StandardScaler()
    cluster_input_scaled = scaler.fit_transform(cluster_input)

    k = min(n_archetypes, len(deltas_df))
    km = KMeans(n_clusters=k, random_state=SEED, n_init=kmeans_n_init)
    labels = km.fit_predict(cluster_input_scaled)
    deltas_df['_archetype'] = labels

    # Silhouette scores for k = 2..5 (used to validate choice of k)
    silhouette_scores: dict[int, float] = {}
    for k_try in range(2, 6):
        if k_try >= len(deltas_df):
            break
        km_try = KMeans(n_clusters=k_try, random_state=SEED, n_init=kmeans_n_init)
        lbl_try = km_try.fit_predict(cluster_input_scaled)
        if len(set(lbl_try)) > 1:
            silhouette_scores[k_try] = float(silhouette_score(
                cluster_input_scaled, lbl_try))

    # Disambiguate clusters that map to the same category by appending a
    # sub-letter ordered by cluster size.
    raw_named = []
    for cid in range(k):
        members = deltas_df[deltas_df['_archetype'] == cid]
        if len(members) == 0:
            continue
        mean_deltas = members[delta_cols].mean()
        ranked = sorted(
            [(c, float(mean_deltas[c]))
             for c in delta_cols if abs(mean_deltas[c]) > 0.05],
            key=lambda kv: -abs(kv[1]),
        )
        top_3 = ranked[:3]
        category = _categorize_archetype(mean_deltas, delta_cols)
        raw_named.append((cid, members, top_3, category))

    # Resolve duplicate names: largest cluster keeps the bare category name;
    # subsequent clusters with the same category are suffixed (-A, -B, ...).
    cat_counts: dict[str, int] = {}
    raw_named.sort(key=lambda r: -len(r[1]))
    archetypes = []
    for cid, members, top_3, category in raw_named:
        seen = cat_counts.get(category, 0)
        cat_counts[category] = seen + 1
        if seen == 0 and cat_counts.get(category, 0) <= 1 and \
                sum(1 for r in raw_named if r[3] == category) == 1:
            name = category
        else:
            suffix = chr(ord('A') + seen)
            name = f"{category}-{suffix}"

        example_idx = int(members['_orig_idx'].iloc[0])
        recipe = _format_recipe(top_3)
        example = {
            'orig_index_in_target': example_idx,
            'top_changes': top_3,
            'recipe': recipe,
        }

        archetypes.append(Archetype(
            cluster_id=cid, name=name, n_students=int(len(members)),
            dominant_features=top_3, example_cf=example,
        ))

    metrics = CounterfactualMetrics(
        n_at_risk=int(len(at_risk_idx)),
        n_with_cf=n_with_cf,
        coverage=float(n_with_cf / max(n_attempted, 1)),
        validity=float(n_valid / max(n_attempted, 1)),
        mean_sparsity=float(np.mean(sparsities)) if sparsities else 0.0,
        mean_proximity=float(np.mean(proximities)) if proximities else 0.0,
    )
    return metrics, archetypes, deltas_df, silhouette_scores


def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base), 'data')
    df_mat = pd.read_csv(os.path.join(data_dir, 'student-mat.csv'), sep=';')
    print(f"Math dataset: {df_mat.shape}, at-risk rate "
          f"(G3<10): {(df_mat['G3'] < 10).mean():.3f}")
    metrics, archetypes, deltas_df, silhouette_scores = generate_archetypes(
        df_mat, n_archetypes=3, cf_per_student=1, max_at_risk=20,
        method='random', enforce_monotonic=True)
    print(f"\nMetrics: {metrics}")
    print(f"Silhouette scores by k: {silhouette_scores}")
    print(f"\nArchetypes ({len(archetypes)}):")
    for a in archetypes:
        print(f"  [{a.cluster_id}] {a.name} -- {a.n_students} students")
        for feat, val in a.dominant_features:
            print(f"      {feat:25s}  {val:+.3f}")


if __name__ == '__main__':
    main()
