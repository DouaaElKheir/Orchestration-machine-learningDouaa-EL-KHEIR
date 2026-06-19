"""DAG Airflow — prévisions quotidiennes envoyées à l'API.

Pipeline : échantillonnage des données → envoi des prévisions à l'API.
Simule un trafic de production et alimente la boucle de feedback.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configuration
API_URL = "http://api:8000"
N_PREDICTIONS = 10  # Nombre de prévisions à envoyer par jour


# ── Tâche 1 — Échantillonnage des données (TODO S17-6) ───────────────────────────

def task_send_predictions(**context) -> None:
    """Échantillonne des données et envoie des prévisions à l'API."""
    # Charger les données préparées
    df = pd.read_csv("data/adult_prepared.csv")
    
    # Retirer la cible pour les prévisions
    if "target" in df.columns:
        features = df.drop(columns=["target"])
    else:
        features = df
    
    # Échantillonner (TODO S17-6)
    sample = features.sample(n=min(N_PREDICTIONS, len(features)))
    
    # Envoyer les prévisions à l'API (TODO S17-7)
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        # Vérifier que l'API est joignable
        client.get("/health").raise_for_status()
        
        # Envoyer chaque prédiction
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())  # types JSON natifs (pas de numpy)
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            print(f"[OK] Prédiction envoyée : {payload}")
    
    print(f"[OK] {len(sample)} prévisions envoyées à l'API avec succès.")


# ── Définition du DAG (TODO S17-8) ───────────────────────────────────────────────

default_args = {
    "owner": "douaa",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="daily_predictions",
    description="Prévisions quotidiennes envoyées à l'API",
    # Tous les 1er du mois à 10h
    schedule="0 10 1 * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "predictions", "adult-income"],
) as dag:

    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
