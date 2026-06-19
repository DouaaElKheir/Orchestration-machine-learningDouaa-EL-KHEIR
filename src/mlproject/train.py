"""Entrainement du modele de classification (baseline) avec suivi MLflow.

Modele : LogisticRegression (voir notebooks/01_baseline.ipynb).
Seance 5 — TP MLflow Tracking : params, metriques, modele et artefacts.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import (
    BASELINE_C,
    BASELINE_CLASS_WEIGHT,
    BASELINE_MAX_ITER,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MODEL_DIR,
)
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment


def build_model(
    c: float = BASELINE_C,
    max_iter: int = BASELINE_MAX_ITER,
    class_weight: str | None = BASELINE_CLASS_WEIGHT,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "clf",
                LogisticRegression(C=c, max_iter=max_iter, class_weight=class_weight),
            ),
        ]
    )


def _log_confusion_matrix(y_test, preds) -> None:
    ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["<=50K", ">50K"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "confusion.png"
        plt.savefig(path)
        mlflow.log_artifact(str(path), artifact_path="plots")
    plt.close()


def train(
    c: float = BASELINE_C,
    max_iter: int = BASELINE_MAX_ITER,
    class_weight: str | None = BASELINE_CLASS_WEIGHT,
) -> dict:
    setup_experiment(
        experiment_name=MLFLOW_EXPERIMENT,
        description=MLFLOW_EXPERIMENT_DESCRIPTION,
        tags=MLFLOW_EXPERIMENT_TAGS,
    )

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    with mlflow.start_run(run_name=f"logreg-c{c}"):
        log_dataset(df, context="training", name="adult-income")
        model = build_model(c=c, max_iter=max_iter, class_weight=class_weight)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        mlflow.log_params({
            "c": c,
            "max_iter": max_iter,
            "class_weight": class_weight or "none",
            "model": "logreg",
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        _log_confusion_matrix(y_test, preds)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=BASELINE_C)
    parser.add_argument("--max-iter", type=int, default=BASELINE_MAX_ITER)
    parser.add_argument("--class-weight", type=str, default=BASELINE_CLASS_WEIGHT)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter, class_weight=args.class_weight)


if __name__ == "__main__":
    main()
