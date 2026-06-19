"""DAG Airflow — ré-entraînement planifié du modèle Adult Income.

Pipeline : préparation des données → entraînement → contrôle qualité.
Les métriques transitent via XCom entre les tâches train et check_quality.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Seuil minimum de F1 pour valider le modèle
QUALITY_THRESHOLD = 0.70


# ── Tâche 1 — Préparation des données (TODO S17-1) ───────────────────────────

def task_prepare_data(**context) -> None:
    """Régénère adult_prepared.csv à partir du CSV brut."""
    from scripts.prepare_data import prepare
    prepare()
    print("[OK] Données préparées avec succès.")


# ── Tâche 2 — Entraînement + push XCom (TODO S17-2) ─────────────────────────

def task_train(**context) -> None:
    """Entraîne le modèle et pousse les métriques dans XCom."""
    from mlproject.train import train
    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])
    print(f"[OK] Entraînement terminé — f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")


# ── Tâche 3 — Contrôle qualité (TODO S17-3) ──────────────────────────────────

def task_check_quality(**context) -> None:
    """Échoue si le F1 est sous le seuil, empêchant la livraison d'un mauvais modèle."""
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    roc_auc = context["ti"].xcom_pull(task_ids="train", key="roc_auc")
    print(f"[INFO] Métriques reçues — f1={f1:.3f}  roc_auc={roc_auc:.3f}")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Contrôle qualité échoué : f1={f1:.3f} < seuil={QUALITY_THRESHOLD}. "
            "Le modèle n'est pas livré."
        )
    print(f"[OK] Qualité validée : f1={f1:.3f} ≥ seuil={QUALITY_THRESHOLD}")


# ── Définition du DAG (TODO S17-4) ───────────────────────────────────────────

default_args = {
    "owner": "douaa",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="model_retraining",
    description="Ré-entraînement planifié du modèle Adult Income",
    # Tous les 1er du mois à 3h
    schedule="0 3 1 * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "retrain", "adult-income"],
) as dag:

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    check = PythonOperator(
        task_id="check_quality",
        python_callable=task_check_quality,
    )

    # ── Ordre des tâches (TODO S17-5) ────────────────────────────────────────
    prepare >> train_task >> check
