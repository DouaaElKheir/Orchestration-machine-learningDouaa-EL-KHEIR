"""Outils d'evaluation partages : graphiques loggues comme artefacts MLflow."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def log_shap_summary(pipeline: Pipeline, x_test, name: str, max_samples: int = 200) -> None:
    """Logger un summary plot SHAP comme artefact MLflow ``shap_summary.png``."""
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    sample = transformed[:max_samples]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample)
    except Exception:  # pragma: no cover - modeles non supportes par TreeExplainer
        logger.warning("SHAP TreeExplainer indisponible pour %s, artefact ignore", name)
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Importance des variables (SHAP) : {name}")
    mlflow.log_figure(fig, "shap_summary.png")
    plt.close(fig)


def log_classification_artifacts(y_true, y_pred, title: str = "model") -> None:
    """Logger matrice de confusion + rapport de classification dans MLflow."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"Matrice de confusion : {title}")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    mlflow.log_dict(report_dict, "classification_report.json")
    mlflow.log_text(classification_report(y_true, y_pred), "classification_report.txt")
