from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datasets.loader import DatasetObj

class DataPreprocessor:
    def __init__(self, feature_types: Dict[str, str], method: str = 'standard'):
        self.feature_types = feature_types
        self.method = method
        self.pipeline = None
        self.feature_names_out = None

    def fit(self, X: pd.DataFrame):
        numeric_features = [f for f, t in self.feature_types.items() if t == 'numeric']
        categorical_features = [f for f, t in self.feature_types.items() if t in ['categorical', 'binary']]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=False
        )

        self.pipeline.fit(X)
        
        # Capture output feature names
        # Note: get_feature_names_out() is available in scikit-learn >= 1.0
        if hasattr(self.pipeline, 'get_feature_names_out'):
            self.feature_names_out = self.pipeline.get_feature_names_out()
        else:
            self.feature_names_out = numeric_features + categorical_features # Approximate fallback

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        X_array = self.pipeline.transform(X)
        
        # Convert back to DataFrame
        if self.feature_names_out is not None:
             columns = self.feature_names_out
        else:
             columns = [f"feat_{i}" for i in range(X_array.shape[1])]
             
        return pd.DataFrame(X_array, columns=columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
