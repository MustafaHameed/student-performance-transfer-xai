"""UCI Student Performance <-> xAPI-Edu concept-level feature crosswalk.

Both datasets are projected into a shared 8-feature concept space so a
classifier trained on one can be evaluated on the other for external
validation. All concepts are scaled to [0, 1] except the binary risk
target.

Concepts (8):
  is_male                 binary
  high_absence            binary  (UCI: absences > 7; xAPI: Above-7)
  family_engagement       0..1     (parental support / survey participation)
  engagement_score        0..1     (study time / behavioral engagement)
  higher_ed_intent        0..1     (UCI 'higher'; xAPI: HighSchool stage proxy)
  prior_failure_risk      0..1     (UCI failures/4; xAPI: low parent-satisfaction)
  parental_education      0..1     (UCI Medu+Fedu/8; xAPI: survey-yes proxy)
  romantic_or_distraction 0..1     (UCI 'romantic'; xAPI: low Discussion proxy)

Target:
  risk = 1 if at-risk else 0
  UCI:   G3 < 10
  xAPI:  Class == 'L'
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CONCEPT_FEATURES = [
    'is_male',
    'high_absence',
    'family_engagement',
    'engagement_score',
    'higher_ed_intent',
    'prior_failure_risk',
    'parental_education',
    'romantic_or_distraction',
]


def _binary(series: pd.Series, true_value) -> np.ndarray:
    return (series == true_value).astype(np.float32).values


def _normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


def uci_to_concepts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Project a raw UCI dataframe (with original string-valued cols and G3)
    into the shared concept space. Returns (X_concept, y_risk).
    """
    cols = {c: df[c] for c in df.columns}

    is_male = _binary(cols['sex'], 'M')
    high_absence = (cols['absences'].astype(float).values > 7).astype(np.float32)

    famsup = _binary(cols['famsup'], 'yes')
    schoolsup = _binary(cols['schoolsup'], 'yes')
    family_engagement = (famsup + schoolsup) / 2.0

    engagement_score = _normalize(cols['studytime'].astype(float).values, 1, 4)

    higher_ed_intent = _binary(cols['higher'], 'yes')

    prior_failure_risk = _normalize(cols['failures'].astype(float).values, 0, 4)

    medu = _normalize(cols['Medu'].astype(float).values, 0, 4)
    fedu = _normalize(cols['Fedu'].astype(float).values, 0, 4)
    parental_education = (medu + fedu) / 2.0

    romantic_or_distraction = _binary(cols['romantic'], 'yes')

    X = np.column_stack([
        is_male, high_absence, family_engagement, engagement_score,
        higher_ed_intent, prior_failure_risk, parental_education,
        romantic_or_distraction,
    ]).astype(np.float32)

    y = (cols['G3'].astype(float).values < 10).astype(np.int64)
    return X, y


def xapi_to_concepts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Project the raw xAPI dataframe into the shared concept space."""
    cols = {c: df[c] for c in df.columns}

    is_male = _binary(cols['gender'], 'M')
    high_absence = _binary(cols['StudentAbsenceDays'], 'Above-7')

    parent_yes = _binary(cols['ParentAnsweringSurvey'], 'Yes')
    parent_good = _binary(cols['ParentschoolSatisfaction'], 'Good')
    family_engagement = (parent_yes + parent_good) / 2.0

    behaviors = np.column_stack([
        cols['raisedhands'].astype(float).values,
        cols['VisITedResources'].astype(float).values,
        cols['AnnouncementsView'].astype(float).values,
        cols['Discussion'].astype(float).values,
    ])
    engagement_score = _normalize(behaviors.mean(axis=1), 0, 100)

    higher_ed_intent = _binary(cols['StageID'], 'HighSchool')

    # Above-7 absence + low parent satisfaction stand in as a coarse failure-risk
    prior_failure_risk = ((1.0 - parent_good) + (1.0 - engagement_score)) / 2.0
    prior_failure_risk = prior_failure_risk.astype(np.float32)

    parental_education = parent_yes  # weak proxy: parents engaged with school

    # Low Discussion engagement stands in as a "distraction" proxy
    discussion_norm = _normalize(cols['Discussion'].astype(float).values, 0, 100)
    romantic_or_distraction = (1.0 - discussion_norm).astype(np.float32)

    X = np.column_stack([
        is_male, high_absence, family_engagement, engagement_score,
        higher_ed_intent, prior_failure_risk, parental_education,
        romantic_or_distraction,
    ]).astype(np.float32)

    y = (cols['Class'] == 'L').astype(np.int64).values
    return X, y


def to_concept_space(df: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray]:
    source = source.lower()
    if source in ('uci', 'portuguese', 'math', 'por', 'mat'):
        return uci_to_concepts(df)
    if source == 'xapi':
        return xapi_to_concepts(df)
    raise ValueError(f"Unknown source '{source}'")


def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base), 'data')

    df_por = pd.read_csv(os.path.join(data_dir, 'student-por.csv'), sep=';')
    X_por, y_por = to_concept_space(df_por, 'uci')
    print(f"UCI-Por concepts: {X_por.shape}, risk rate: {y_por.mean():.3f}")

    df_xapi = pd.read_csv(os.environ.get('XAPI_CSV', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xAPI-Edu-Data.csv')))
    X_x, y_x = to_concept_space(df_xapi, 'xapi')
    print(f"xAPI concepts:    {X_x.shape}, risk rate: {y_x.mean():.3f}")

    print(f"Feature names: {CONCEPT_FEATURES}")


if __name__ == '__main__':
    main()
