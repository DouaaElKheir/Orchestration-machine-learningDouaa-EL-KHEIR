"""Entraînement local du modèle sans dépendance MLflow.

Ce script produit un modèle scikit-learn exporté vers models/model.joblib.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import f1_score, roc_auc_score

from mlproject.config import MODEL_DIR
from mlproject.data import load_data, split
from mlproject.train import build_model


def train_local(c: float = 1.0, max_iter: int = 1000, class_weight: str | None = "balanced") -> dict[str, float]:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    model = build_model(c=c, max_iter=max_iter, class_weight=class_weight)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"[OK] modèle sauvegardé dans {MODEL_DIR / 'model.joblib'}")
    print(f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}")
    return metrics


def main() -> None:
    train_local()


if __name__ == "__main__":
    main()
