"""Configuration centrale du projet Adult Income.

data.py et features.py lisent toutes leurs colonnes via ces constantes.
Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- Dataset Adult Income -----------------------------------------------------
# Problematique : predire si le revenu annuel depasse 50K$ (1) ou non (0)

RAW_DATA_PATH = ROOT / "data" / "adult.csv"
DATA_PATH = ROOT / "data" / "adult_prepared.csv"
MODEL_DIR = ROOT / "models"

TARGET = "income"
MISSING_VALUE = "?"

# Mapping de la cible brute (adult.csv) vers 0/1 (adult_prepared.csv)
INCOME_MAP: dict[str, int] = {"<=50K": 0, ">50K": 1}

NUMERIC_FEATURES: list[str] = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- MLflow -------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "adult-income-baseline")
MLFLOW_EXPERIMENT_MODELS = os.getenv(
    "MLFLOW_EXPERIMENT_MODELS", "adult-income-compare-models"
)
MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION",
    "Suivi de la baseline Adult Income (regression logistique).",
)
MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project": "adult-income",
    "task": "binary-classification",
    "target": TARGET,
    "team": "esgi-iabd",
}
MLFLOW_EXPERIMENT_MODELS_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_MODELS_DESCRIPTION",
    "Comparaison de modeles Adult Income (RF, XGBoost, LightGBM).",
)
MLFLOW_EXPERIMENT_MODELS_TAGS: dict[str, str] = {
    "project": "adult-income",
    "task": "automl-comparison",
    "target": TARGET,
    "team": "esgi-iabd",
}
MLFLOW_EXPERIMENT_OPTUNA = os.getenv(
    "MLFLOW_EXPERIMENT_OPTUNA", "adult-income-optuna"
)
MLFLOW_EXPERIMENT_OPTUNA_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_OPTUNA_DESCRIPTION",
    "Optimisation Optuna (TPE) des modeles Adult Income.",
)
MLFLOW_EXPERIMENT_OPTUNA_TAGS: dict[str, str] = {
    "project": "adult-income",
    "task": "optuna-optimization",
    "target": TARGET,
    "team": "esgi-iabd",
}
MLFLOW_EXPERIMENT_EVAL = os.getenv("MLFLOW_EXPERIMENT_EVAL", "adult-income-evaluation")
MLFLOW_EXPERIMENT_EVAL_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_EVAL_DESCRIPTION",
    "Evaluation automatisee et porte qualite des modeles Adult Income.",
)
MLFLOW_EXPERIMENT_EVAL_TAGS: dict[str, str] = {
    "project": "adult-income",
    "task": "model-evaluation",
    "target": TARGET,
    "team": "esgi-iabd",
}
MODEL_NAME = os.getenv("MODEL_NAME", "adult-income-classifier")

# --- Baseline (choix issu du notebook 01_baseline.ipynb) ----------------------
BASELINE_C = 1.0
BASELINE_MAX_ITER = 1000
BASELINE_CLASS_WEIGHT = "balanced"

# --- Seuils de validation (S11) -----------------------------------------------
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.85"))
EVAL_F1_MIN = float(os.getenv("EVAL_F1_MIN", "0.65"))
