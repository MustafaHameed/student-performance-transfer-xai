"""
Data Loader Module for UCI Student Performance Dataset

This module handles:
1. Dataset download from UCI repository
2. Data preprocessing and encoding
3. Train/validation/test splitting
4. Feature engineering for deep learning models
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class StudentPerformanceDataset:
    """
    Handler for UCI Student Performance Dataset
    
    Dataset contains 649 instances with 30 features predicting
    student grades (G1, G2, G3) in Math and Portuguese courses.
    
    Attributes:
        df: Combined dataframe with all features
        X: Feature matrix
        y: Target variable (G3 - final grade)
        feature_names: List of feature column names
        categorical_features: List of categorical feature indices
        numerical_features: List of numerical feature indices
    """
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Create data directory if not exists
        os.makedirs(data_dir, exist_ok=True)

    def _cache_path_for_mode(self, subject_mode):
        cache_names = {
            'portuguese': 'student_performance.csv',
            'math': 'student_performance_math.csv',
            'both': 'student_performance_both.csv',
        }
        return os.path.join(self.data_dir, cache_names[subject_mode])

    def _resolve_subject_mode(self, use_both_subjects, subject_subset):
        if subject_subset is None:
            return 'both' if use_both_subjects else 'portuguese'

        subject_mode = str(subject_subset).strip().lower()
        aliases = {
            'por': 'portuguese',
            'portuguese': 'portuguese',
            'pt': 'portuguese',
            'math': 'math',
            'mat': 'math',
            'both': 'both',
            'combined': 'both',
            'all': 'both',
        }
        if subject_mode not in aliases:
            raise ValueError(
                "subject_subset must be one of 'portuguese', 'math', or 'both'"
            )
        return aliases[subject_mode]

    def _load_cached_dataset(self, subject_mode):
        cache_path = self._cache_path_for_mode(subject_mode)
        if os.path.exists(cache_path):
            self.df = pd.read_csv(cache_path)
            return self.df

        raw_subject_files = [
            os.path.join(self.data_dir, 'student-mat.csv'),
            os.path.join(self.data_dir, 'student-por.csv'),
        ]
        if all(os.path.exists(path) for path in raw_subject_files):
            return self._load_from_raw_subject_files(subject_mode, save_cache=True)

        return None

    def _load_from_raw_subject_files(self, subject_mode, save_cache=False):
        df_math = pd.read_csv(os.path.join(self.data_dir, 'student-mat.csv'), sep=';')
        df_por = pd.read_csv(os.path.join(self.data_dir, 'student-por.csv'), sep=';')

        df_math['subject'] = 'Math'
        df_por['subject'] = 'Portuguese'

        if subject_mode == 'math':
            self.df = df_math.reset_index(drop=True)
        elif subject_mode == 'portuguese':
            self.df = df_por.reset_index(drop=True)
        else:
            self.df = pd.concat([df_math, df_por], axis=0, ignore_index=True)

        if save_cache:
            self.df.to_csv(self._cache_path_for_mode(subject_mode), index=False)

        return self.df
    
    def download_and_load(self, use_both_subjects=False, include_g1_g2=False,
                          subject_subset=None, refresh=False):
        """
        Download UCI Student Performance Dataset
        
        Parameters:
            use_both_subjects: Backward-compatible alias. If True, load both
                              Math and Portuguese datasets.
            include_g1_g2: If True, include G1 and G2 as features (easier prediction)
                          If False, predict G3 without prior grades (early intervention)
            subject_subset: Explicit subject mode: 'portuguese', 'math', or 'both'.
            refresh: If True, ignore cached CSVs and re-fetch from source.
        
        Returns:
            DataFrame with loaded data
        """
        subject_mode = self._resolve_subject_mode(use_both_subjects, subject_subset)

        if not refresh:
            cached_df = self._load_cached_dataset(subject_mode)
            if cached_df is not None:
                print(f"Loaded cached {subject_mode} dataset from disk")
                print(f"Shape: {cached_df.shape}")
                return cached_df

        print(f"Loading UCI Student Performance Dataset ({subject_mode})...")
        
        try:
            if subject_mode == 'portuguese':
                from ucimlrepo import fetch_ucirepo
                student_performance = fetch_ucirepo(id=320)

                X = student_performance.data.features
                y = student_performance.data.targets

                self.df = pd.concat([X, y], axis=1)
                cache_path = self._cache_path_for_mode(subject_mode)
                self.df.to_csv(cache_path, index=False)

                print("Dataset loaded successfully!")
                print(f"Shape: {self.df.shape}")
                print(f"Features: {X.shape[1]}")
                print(f"Target columns: {y.columns.tolist()}")
                print(f"Data saved to {cache_path}")
            else:
                self._download_alternative()
                self._load_from_raw_subject_files(subject_mode, save_cache=True)
                print("Dataset loaded successfully!")
                print(f"Shape: {self.df.shape}")
                print(f"Data saved to {self._cache_path_for_mode(subject_mode)}")
            
        except Exception as e:
            print(f"Error downloading from UCI: {e}")
            print("Attempting alternative download method...")
            self._download_alternative()
            self._load_from_raw_subject_files(subject_mode, save_cache=True)
        
        return self.df
    
    def _download_alternative(self):
        """Alternative download using direct URL"""
        import urllib.request
        import zipfile
        
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        zip_path = os.path.join(self.data_dir, "student_performance.zip")
        
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        return zip_path
    
    def get_feature_info(self):
        """Get detailed information about dataset features"""
        
        feature_info = {
            'Demographics': {
                'school': 'Student school (binary: GP or MS)',
                'sex': 'Student sex (binary: F or M)',
                'age': 'Student age (numeric: 15-22)',
                'address': 'Home address type (binary: U - urban or R - rural)',
                'famsize': 'Family size (binary: LE3 - <=3 or GT3 - >3)',
                'Pstatus': 'Parent cohabitation status (binary: T - together or A - apart)'
            },
            'Parental_Background': {
                'Medu': "Mother's education (0-4: none to higher education)",
                'Fedu': "Father's education (0-4: none to higher education)",
                'Mjob': "Mother's job (teacher, health, services, at_home, other)",
                'Fjob': "Father's job (teacher, health, services, at_home, other)"
            },
            'School_Related': {
                'reason': 'Reason to choose school (home, reputation, course, other)',
                'guardian': 'Student guardian (mother, father, other)',
                'traveltime': 'Home to school travel time (1-4: <15min to >1hr)',
                'studytime': 'Weekly study time (1-4: <2hrs to >10hrs)',
                'failures': 'Number of past class failures (0-4)',
                'schoolsup': 'Extra educational support (binary: yes or no)',
                'famsup': 'Family educational support (binary: yes or no)',
                'paid': 'Extra paid classes (binary: yes or no)',
                'activities': 'Extra-curricular activities (binary: yes or no)',
                'nursery': 'Attended nursery school (binary: yes or no)',
                'higher': 'Wants higher education (binary: yes or no)'
            },
            'Lifestyle': {
                'internet': 'Internet access at home (binary: yes or no)',
                'romantic': 'In a romantic relationship (binary: yes or no)',
                'famrel': 'Quality of family relationships (1-5: very bad to excellent)',
                'freetime': 'Free time after school (1-5: very low to very high)',
                'goout': 'Going out with friends (1-5: very low to very high)',
                'Dalc': 'Workday alcohol consumption (1-5: very low to very high)',
                'Walc': 'Weekend alcohol consumption (1-5: very low to very high)',
                'health': 'Current health status (1-5: very bad to very good)',
                'absences': 'Number of school absences (0-93)'
            },
            'Target_Variables': {
                'G1': 'First period grade (0-20)',
                'G2': 'Second period grade (0-20)',
                'G3': 'Final grade - TARGET (0-20)'
            }
        }
        
        return feature_info
    
    def preprocess(self, target='G3', include_prior_grades=False, classification=False,
                   pass_threshold=10, scale_numeric=True):
        """
        Preprocess data for model training
        
        Parameters:
            target: Target variable ('G3' for final grade)
            include_prior_grades: Include G1, G2 as features
            classification: If True, convert to binary classification (pass/fail)
            pass_threshold: Grade threshold for passing (default: 10)
            scale_numeric: If True, standardize numerical columns. Tree models
                          can set this to False to retain native numeric scale.
        
        Returns:
            X: Preprocessed feature matrix
            y: Target variable
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call download_and_load() first.")
        
        df = self.df.copy()
        self.categorical_features = []
        self.numerical_features = []
        self.label_encoders = {}
        
        # Define categorical and numerical columns
        categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                           'Mjob', 'Fjob', 'reason', 'guardian',
                           'schoolsup', 'famsup', 'paid', 'activities',
                           'nursery', 'higher', 'internet', 'romantic']
        
        numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                         'failures', 'famrel', 'freetime', 'goout',
                         'Dalc', 'Walc', 'health', 'absences']
        
        # Add prior grades if requested
        if include_prior_grades:
            numerical_cols.extend(['G1', 'G2'])
        
        # Check for subject column
        if 'subject' in df.columns:
            categorical_cols.append('subject')
        
        # Filter to available columns
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        numerical_cols = [c for c in numerical_cols if c in df.columns]
        
        # Encode categorical variables
        X_cat = pd.DataFrame()
        for col in categorical_cols:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Process numerical variables
        X_num = df[numerical_cols].copy()
        
        # Combine features
        X = pd.concat([X_cat, X_num], axis=1)
        self.feature_names = X.columns.tolist()
        
        # Store feature indices
        self.categorical_features = list(range(len(categorical_cols)))
        self.numerical_features = list(range(len(categorical_cols), len(self.feature_names)))
        
        # Get target variable
        y = df[target].values
        
        # Convert to classification if requested
        if classification:
            y = (y >= pass_threshold).astype(int)
        
        # Scale numerical features when requested
        X_processed = X.copy()
        if scale_numeric and numerical_cols:
            X_processed[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        self.X = X_processed.values.astype(np.float32)
        self.y = y.astype(np.float32) if not classification else y.astype(np.int64)
        
        print(f"\nPreprocessing complete:")
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Categorical features: {len(self.categorical_features)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Scaled numerical features: {scale_numeric}")
        
        return self.X, self.y
    
    def get_categorical_dims(self):
        """Get dimensions for categorical embeddings"""
        if self.df is None:
            return None
        
        cat_dims = []
        for idx in self.categorical_features:
            col = self.feature_names[idx]
            cat_dims.append(len(self.label_encoders[col].classes_))
        
        return cat_dims
    
    def create_folds(self, n_folds=5, random_state=42):
        """
        Create stratified k-fold splits for cross-validation
        
        Parameters:
            n_folds: Number of folds
            random_state: Random seed for reproducibility
        
        Returns:
            List of (train_idx, val_idx) tuples
        """
        if self.X is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        # For regression, create bins for stratification
        y_binned = pd.qcut(self.y, q=5, labels=False, duplicates='drop')
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(skf.split(self.X, y_binned))
        
        return folds
    
    def get_train_val_test_split(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if self.X is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Test data loading functionality"""
    
    # Initialize dataset handler
    dataset = StudentPerformanceDataset(data_dir='./data')
    
    # Download and load data
    df = dataset.download_and_load()
    
    # Display basic info
    print("\n" + "="*50)
    print("Dataset Overview")
    print("="*50)
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
    
    # Preprocess data
    X, y = dataset.preprocess(target='G3', include_prior_grades=False)
    
    # Create train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_train_val_test_split()
    
    # Print feature info
    print("\n" + "="*50)
    print("Feature Information")
    print("="*50)
    feature_info = dataset.get_feature_info()
    for category, features in feature_info.items():
        print(f"\n{category}:")
        for feat, desc in features.items():
            print(f"  - {feat}: {desc}")
    
    return dataset


if __name__ == "__main__":
    main()
