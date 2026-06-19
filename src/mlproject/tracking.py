"""Configuration partagee du suivi MLflow.

Centralise la configuration du tracking pour eviter la duplication dans les
scripts d'entrainement, et ajoute la tracabilite des donnees (dataset lineage).
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from mlproject.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment(
    experiment_name: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Configurer le tracking MLflow et les metadonnees d'une experience."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name)
    client = mlflow.MlflowClient()

    note = description if description is not None else MLFLOW_EXPERIMENT_DESCRIPTION
    if note:
        client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", note)

    all_tags = dict(MLFLOW_EXPERIMENT_TAGS)
    if tags:
        all_tags.update(tags)
    for key, value in all_tags.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)

    logger.info("MLflow configure pour l'experience '%s'", experiment_name)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant."""
    dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
        df, source=str(DATA_PATH), targets=TARGET, name=name
    )
    mlflow.log_input(dataset, context=context)
