"""xAPI-Edu (Kalboard 360) loader.

Loads the xAPI-Edu-Data.csv dataset and produces a clean numeric feature
matrix plus a binary at-risk target. Class labels: L=at-risk, M/H=not at-risk.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


XAPI_DEFAULT_PATH = os.environ.get(
    'XAPI_CSV',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'data', 'xAPI-Edu-Data.csv'))

CATEGORICAL_COLS = [
    'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
    'SectionID', 'Topic', 'Semester', 'Relation',
    'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays',
]

NUMERIC_COLS = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']


class XapiDataset:
    def __init__(self, csv_path: str = XAPI_DEFAULT_PATH):
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_encoders = {}

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"xAPI dataset not found at {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def preprocess(self, risk_label: str = 'L') -> tuple[np.ndarray, np.ndarray]:
        """Encode categoricals and produce binary at-risk target.

        risk_label: which Class value(s) are treated as at-risk.
            'L' -> only Low is at-risk (default; matches plan §C exp 7)
        """
        if self.df is None:
            self.load()
        df = self.df.copy()

        X_cat = pd.DataFrame()
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        X_num = df[NUMERIC_COLS].copy().astype(np.float32)
        X = pd.concat([X_cat, X_num], axis=1)
        self.feature_names = X.columns.tolist()
        self.X = X.values.astype(np.float32)

        self.y = (df['Class'] == risk_label).astype(np.int64).values
        return self.X, self.y


def main():
    ds = XapiDataset()
    X, y = ds.preprocess()
    print(f"xAPI X shape: {X.shape}, y shape: {y.shape}, at-risk: {int(y.sum())}/{len(y)}")
    print(f"Features: {ds.feature_names}")


if __name__ == '__main__':
    main()
