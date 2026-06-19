"""Chargement et decoupage des donnees Adult Income."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    INCOME_MAP,
    MISSING_VALUE,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    TEST_SIZE,
)


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {sorted(missing)}")


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le CSV brut Adult Income (cible 0/1, '?' -> 'Unknown')."""
    prepared = df.copy()
    for col in CATEGORICAL_FEATURES:
        prepared[col] = prepared[col].replace(MISSING_VALUE, "Unknown")
    if prepared[TARGET].dtype == object:
        prepared[TARGET] = prepared[TARGET].map(INCOME_MAP)
    if prepared[TARGET].isna().any():
        raise ValueError(f"Valeurs inattendues dans la colonne cible '{TARGET}'")
    return prepared


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset prepare, ou le genere depuis adult.csv si absent."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    elif path == DATA_PATH and RAW_DATA_PATH.exists():
        df = prepare_raw(pd.read_csv(RAW_DATA_PATH))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    else:
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. Placez adult.csv dans data/ puis lancez 'make data'."
        )

    _validate_columns(df)
    return df


def split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
