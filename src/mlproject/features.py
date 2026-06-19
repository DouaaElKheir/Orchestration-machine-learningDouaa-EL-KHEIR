"""Pre-processing des features Adult Income."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Pipeline adapte au dataset Adult Income (numeriques + categorielles)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = [("num", numeric_pipeline, NUMERIC_FEATURES)]
    if CATEGORICAL_FEATURES:
        transformers.append(("cat", categorical_pipeline, CATEGORICAL_FEATURES))
    return ColumnTransformer(transformers=transformers)
