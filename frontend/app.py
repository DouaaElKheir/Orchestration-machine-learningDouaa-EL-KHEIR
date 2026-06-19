import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5001")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
GITHUB_URL = os.getenv(
    "GITHUB_URL",
    "https://github.com/DouaaElKheir/Orchestration-machine-learningDouaa-EL-KHEIR",
)

st.set_page_config(
    page_title="DOUAA EL KHEIR - Adult Income Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLES = """
<style>
    .stApp {
        background: linear-gradient(135deg, #fafafa 0%, #f0f4f8 45%, #e2e8f0 100%);
        color: #0f172a;
        min-height: 100vh;
    }

    .css-1lcbmhc.e1fqkh3o3 {
        background: transparent;
    }

    .stHeader, .stAppHeader, .stMarkdown {
        color: #0f172a !important;
    }

    .stButton>button {
        background-color: #111827;
        color: #f8fafc;
        border-radius: 999px;
        padding: 0.95rem 1.8rem;
        font-weight: 700;
        box-shadow: 0 16px 40px rgba(17, 24, 39, 0.18);
    }

    .stButton>button:hover {
        background-color: #0f172a;
    }

    .stSidebar {
        background: #ffffff !important;
        border-right: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
    }

    .stSidebar .css-1d391kg, .stSidebar .css-1v3fvcr, .stSidebar .css-1d0g9qv {
        background-color: transparent !important;
    }

    .stSidebar h2, .stSidebar h3, .stSidebar label, .stSidebar span {
        color: #0f172a !important;
    }

    .stApp div[role="main"] {
        padding: 2.5rem 3rem 3rem 3rem;
    }

    .stMetric {
        background: #ffffff !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        border-radius: 1rem;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0f172a !important;
    }

    .stDivider {
        border-color: rgba(15, 23, 42, 0.12) !important;
    }

    .stDataFrame, .stTable {
        border-radius: 1.25rem;
        overflow: hidden;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }

    .css-1d391kg, .css-1v3fvcr, .css-1d0g9qv, .css-qbe2hs, .css-1pqoj0f {
        background-color: rgba(255, 255, 255, 0.96) !important;
        border-radius: 1.4rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(15, 23, 42, 0.08);
        color: #0f172a !important;
    }

    .stTextInput>div>div>input,
    .stSelectbox>div>div>div>select,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        color: #0f172a !important;
        background-color: rgba(248, 250, 252, 0.96) !important;
        border: 1px solid rgba(15, 23, 42, 0.1) !important;
    }

    .stApp a, .stApp a:hover, .stApp a:visited {
        color: #1d4ed8 !important;
    }

    .hero-card {
        background: linear-gradient(180deg, #ffe4f0 0%, #ffffff 100%);
        border-radius: 2rem;
        padding: 2rem;
        box-shadow: 0 35px 80px rgba(219, 39, 119, 0.12);
        border: 1px solid rgba(249, 115, 210, 0.18);
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.75rem 1.25rem;
        border-radius: 999px;
        font-weight: 700;
        text-decoration: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        font-size: 0.95rem;
    }

    .hero-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.15);
    }

    .hero-chip-api {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        color: #9f1239 !important;
        border: 2px solid rgba(156, 163, 175, 0.24);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.15);
    }

    .hero-chip-mlflow {
        background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
        color: #9d174d !important;
        border: 2px solid rgba(190, 18, 93, 0.16);
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.15);
    }

    .hero-chip-airflow {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af !important;
        border: 2px solid rgba(59, 130, 246, 0.24);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15);
    }

    .hero-chip-github {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        color: #3730a3 !important;
        border: 2px solid rgba(99, 102, 241, 0.32);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15);
    }

    .sidebar-card {
        background: #fff7ed !important;
        border: 1px solid rgba(251, 146, 60, 0.18) !important;
        border-radius: 1.5rem !important;
        padding: 1rem !important;
        box-shadow: 0 20px 40px rgba(249, 115, 22, 0.08) !important;
    }

    .css-xqmnk1 {
        color: #0f172a !important;
    }
</style>
"""

st.markdown(PAGE_STYLES, unsafe_allow_html=True)

FEATURES = {
    "age": {
        "label": "Âge",
        "min": 17,
        "max": 90,
        "step": 1,
        "value": 38,
    },
    "fnlwgt": {
        "label": "Poids final (fnlwgt)",
        "min": 10000,
        "max": 650000,
        "step": 1000,
        "value": 150000,
    },
    "education.num": {
        "label": "Niveau d'éducation (num)",
        "min": 1,
        "max": 16,
        "step": 1,
        "value": 13,
    },
    "capital.gain": {
        "label": "Capital gagné",
        "min": 0,
        "max": 100000,
        "step": 100,
        "value": 0,
    },
    "capital.loss": {
        "label": "Capital perdu",
        "min": 0,
        "max": 10000,
        "step": 10,
        "value": 0,
    },
    "hours.per.week": {
        "label": "Heures par semaine",
        "min": 1,
        "max": 99,
        "step": 1,
        "value": 40,
    },
}

CATEGORICAL_OPTIONS = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
        "Unknown",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
        "Unknown",
    ],
    "marital.status": [
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Widowed",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
        "Unknown",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
    ],
    "sex": ["Male", "Female"],
    "native.country": [
        "United-States",
        "Mexico",
        "Philippines",
        "Germany",
        "Puerto-Rico",
        "Canada",
        "El-Salvador",
        "India",
        "Cuba",
        "England",
        "Jamaica",
        "South",
        "China",
        "Japan",
        "Italy",
        "Greece",
        "Vietnam",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
        "Unknown",
    ],
}

SECTIONS = ["Accueil", "Prédiction", "Modèle", "Architecture", "Évaluation", "Historique"]

if "history" not in st.session_state:
    st.session_state.history = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL
if "mlflow_url" not in st.session_state:
    st.session_state.mlflow_url = MLFLOW_URL
if "airflow_url" not in st.session_state:
    st.session_state.airflow_url = AIRFLOW_URL
if "last_request" not in st.session_state:
    st.session_state.last_request = {}


def api_health() -> str:
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=6)
        if response.ok and response.json().get("status") == "ok":
            return "connecté"
    except requests.RequestException:
        return "indisponible"
    return "indisponible"


def predict_remote(payload: dict) -> dict | None:
    try:
        response = requests.post(
            f"{st.session_state.api_url}/predict", json=payload, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(
            "Impossible de joindre l'API. Vérifiez que le service est démarré et que l'URL est correcte."
        )
        st.write(f"Détail : {error}")
        return None


def render_header() -> None:
    api_link = API_PUBLIC_URL
    mlflow_link = st.session_state.mlflow_url
    airflow_link = st.session_state.airflow_url
    github_link = GITHUB_URL

    st.markdown(
        f"""
        <div class='hero-card'>
            <div style='display:flex; flex-wrap:wrap; gap:1.5rem; align-items:center;'>
                <div style='flex:1; min-width:280px;'>
                    <p style='margin:0; color:#be185d; font-size:0.9rem; font-weight:800; letter-spacing:0.18em; text-transform:uppercase;'>🚀 MLOps Studio</p>
                    <h1 style='margin:0.7rem 0 0 0; color:#111827; font-size:3rem; line-height:1.05;'>Adult Income</h1>
                    <p style='margin:1rem 0 0 0; color:#475569; font-size:1.05rem; max-width:680px;'>Interface designée pour tester le modèle, analyser l'évaluation, et accéder rapidement à l'API, MLflow, Airflow et GitHub.</p>
                    <div style='display:flex; flex-wrap:wrap; gap:0.75rem; margin-top:1.5rem;'>
                        <a class='hero-chip hero-chip-api' href='{api_link}' target='_blank'>🔧 API</a>
                        <a class='hero-chip hero-chip-mlflow' href='{mlflow_link}' target='_blank'>📊 MLflow</a>
                        <a class='hero-chip hero-chip-airflow' href='{airflow_link}' target='_blank'>⚙️ Airflow</a>
                        <a class='hero-chip hero-chip-github' href='{github_link}' target='_blank'>🐙 GitHub</a>
                    </div>
                </div>
                <div style='display:grid; gap:1rem; min-width:260px;'>
                    <div style='background:#ffffff; border-radius:1.4rem; padding:1.3rem; box-shadow:0 24px 50px rgba(15,23,42,0.08);'>
                        <div style='font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.12em;'>🟢 API status</div>
                        <div style='margin-top:0.65rem; font-size:2rem; font-weight:800; color:#111827;'>{api_health()}</div>
                    </div>
                    <div style='background:#ffffff; border-radius:1.4rem; padding:1.3rem; box-shadow:0 24px 50px rgba(15,23,42,0.08);'>
                        <div style='font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.12em;'>🎯 Seuil</div>
                        <div style='margin-top:0.65rem; font-size:2rem; font-weight:800; color:#111827;'>{int(st.session_state.threshold * 100)}%</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    render_header()
    api_status = api_health()
    connected = api_status == "connecté"

    st.write(
        "Cette application expose un modèle de classification binaire capable de prédire si un revenu est supérieur à 50K."
    )
    st.write(
        "Le frontend Streamlit appelle une API FastAPI qui charge le modèle entraîné depuis `models/model.joblib`."
    )

    st.markdown("---")
    st.markdown("## Observations clés")
    st.markdown(
        "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:1rem; margin-bottom:1.5rem;'>"
        "<div style='background:#ffffff; color:#0f172a; padding:1.3rem; border-radius:1.5rem; border:1px solid rgba(15,23,42,0.08); box-shadow:0 20px 40px rgba(15,23,42,0.06);'>"
        "<div style='font-size:0.85rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b;'>API</div>"
        f"<div style='font-size:2.3rem; font-weight:800; margin-top:0.6rem;'>{'Connectée' if connected else 'Indisponible'}</div>"
        "<div style='color:#64748b; margin-top:0.5rem;'>Statut du service</div>"
        "</div>"
        "<div style='background:#0f172a; color:#f8fafc; padding:1.3rem; border-radius:1.5rem; box-shadow:0 20px 40px rgba(15,23,42,0.18);'>"
        "<div style='font-size:0.85rem; text-transform:uppercase; letter-spacing:0.12em; color:#cbd5e1;'>Seuil</div>"
        f"<div style='font-size:2.3rem; font-weight:800; margin-top:0.6rem;'>{int(st.session_state.threshold * 100)} %</div>"
        "<div style='color:#cbd5e1; margin-top:0.5rem;'>Seuil de décision</div>"
        "</div>"
        "<div style='background:#ffffff; color:#0f172a; padding:1.3rem; border-radius:1.5rem; border:1px solid rgba(15,23,42,0.08); box-shadow:0 20px 40px rgba(15,23,42,0.06);'>"
        "<div style='font-size:0.85rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b;'>Prédictions</div>"
        f"<div style='font-size:2.3rem; font-weight:800; margin-top:0.6rem;'>{len(st.session_state.history)}</div>"
        "<div style='color:#64748b; margin-top:0.5rem;'>Demandes traitées</div>"
        "</div>"
        "<div style='background:#0f172a; color:#f8fafc; padding:1.3rem; border-radius:1.5rem; box-shadow:0 20px 40px rgba(15,23,42,0.18);'>"
        "<div style='font-size:0.85rem; text-transform:uppercase; letter-spacing:0.12em; color:#cbd5e1;'>Probabilité moyenne</div>"
        f"<div style='font-size:2.3rem; font-weight:800; margin-top:0.6rem;'>{(pd.DataFrame(st.session_state.history)['probability'].mean() if st.session_state.history else 0.0):.2%}</div>"
        "<div style='color:#cbd5e1; margin-top:0.5rem;'>Risque estimé</div>"
        "</div>"
        "</div>", unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## Résumé opérationnel")
    st.markdown(
        "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:1rem;'>"
        "<div style='background:#f8fafc; color:#0f172a; padding:1.25rem; border-radius:1.25rem; border:1px solid rgba(15,23,42,0.08);'>"
        "<div style='font-size:0.85rem; color:#475569;'>Observations</div>"
        "<div style='font-size:1.9rem; font-weight:700; margin-top:0.5rem;'>569</div>"
        "</div>"
        "<div style='background:#f8fafc; color:#0f172a; padding:1.25rem; border-radius:1.25rem; border:1px solid rgba(15,23,42,0.08);'>"
        "<div style='font-size:0.85rem; color:#475569;'>Variables</div>"
        "<div style='font-size:1.9rem; font-weight:700; margin-top:0.5rem;'>30</div>"
        "</div>"
        "<div style='background:#f8fafc; color:#0f172a; padding:1.25rem; border-radius:1.25rem; border:1px solid rgba(15,23,42,0.08);'>"
        "<div style='font-size:0.85rem; color:#475569;'>Type</div>"
        "<div style='font-size:1.9rem; font-weight:700; margin-top:0.5rem;'>Classification</div>"
        "</div>"
        "<div style='background:#f8fafc; color:#0f172a; padding:1.25rem; border-radius:1.25rem; border:1px solid rgba(15,23,42,0.08);'>"
        "<div style='font-size:0.85rem; color:#475569;'>Cible</div>"
        "<div style='font-size:1.9rem; font-weight:700; margin-top:0.5rem;'>&lt;=50K / &gt;50K</div>"
        "</div>"
        "</div>", unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:1.5rem;'>"
        "<div style='background:#ffffff; color:#0f172a; padding:1.4rem; border-radius:1.4rem; box-shadow:0 24px 50px rgba(15,23,42,0.08);'>"
        "<h3 style='margin-bottom:0.75rem;'>Interprétation métier</h3>"
        "<ul style='margin:0; padding-left:1.25rem; color:#334155;'>"
        "<li><strong>0</strong> : revenu prédit comme <=50K</li>"
        "<li><strong>1</strong> : revenu prédit comme >50K</li>"
        "<li>La probabilité indique le risque estimé de revenu élevé.</li>"
        "</ul>"
        "</div>"
        "<div style='background:#0f172a; color:#f8fafc; padding:1.4rem; border-radius:1.4rem; box-shadow:0 24px 50px rgba(15,23,42,0.18);'>"
        "<h3 style='margin-bottom:0.75rem;'>Service API</h3>"
        f"<p style='font-size:1.15rem; margin:0.4rem 0 0.3rem 0;'><strong>Status :</strong> {api_status}</p>"
        "<p style='color:#cbd5e1; margin:0;'>FastAPI répond sur /health et /predict.</p>"
        "</div>"
        "</div>", unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## Pipeline MLOps du projet")
    st.markdown(
        "<div style='display:flex; flex-wrap:wrap; gap:0.75rem;'>"
        "<span style='background:#ffffff; color:#0f172a; padding:0.85rem 1rem; border-radius:999px; border:1px solid rgba(15,23,42,0.08);'>Dataset</span>"
        "<span style='background:#eef2ff; color:#1d4ed8; padding:0.85rem 1rem; border-radius:999px;'>Préparation</span>"
        "<span style='background:#e0f2fe; color:#0369a1; padding:0.85rem 1rem; border-radius:999px;'>Entraînement</span>"
        "<span style='background:#dcfce7; color:#166534; padding:0.85rem 1rem; border-radius:999px;'>Modèle</span>"
        "<span style='background:#fef3c7; color:#92400e; padding:0.85rem 1rem; border-radius:999px;'>API</span>"
        "<span style='background:#fef2f2; color:#991b1b; padding:0.85rem 1rem; border-radius:999px;'>Frontend</span>"
        "<span style='background:#f3e8ff; color:#6d28d9; padding:0.85rem 1rem; border-radius:999px;'>Utilisateurs</span>"
        "</div>", unsafe_allow_html=True,
    )


def render_prediction() -> None:
    st.header("Prédiction")
    st.write(
        "Complétez le profil ci-dessous, puis cliquez sur **Lancer la prédiction** pour obtenir une estimation professionnelle."
    )

    with st.form(key="prediction_form"):
        columns = st.columns([1, 1])
        inputs: dict[str, object] = {}

        for idx, (key, metadata) in enumerate(FEATURES.items()):
            container = columns[idx % 2]
            inputs[key] = container.number_input(
                metadata["label"],
                min_value=metadata["min"],
                max_value=metadata["max"],
                step=metadata["step"],
                value=metadata["value"],
            )

        for key, options in CATEGORICAL_OPTIONS.items():
            inputs[key] = st.selectbox(key.replace(".", " ").title(), options, index=0)

        submit = st.form_submit_button("Lancer la prédiction")

    if submit:
        if api_health() != "connecté":
            st.warning(
                "L'API est actuellement injoignable. Vérifiez l'URL dans les paramètres de la page Accueil."
            )
            return

        result = predict_remote(inputs)
        if result is not None:
            score = result.get("probability", 0.0)
            label = "> 50K $" if score >= st.session_state.threshold else "<= 50K $"
            st.session_state.last_request = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": label,
                "probability": score,
                **inputs,
            }
            st.session_state.history.insert(0, st.session_state.last_request)

            st.success(f"Résultat : **{label}**")
            st.metric("Probabilité", f"{score:.2%}")
            st.progress(score)

            st.markdown("### Profil soumis")
            st.table(pd.DataFrame([inputs]).T.rename(columns={0: "Valeur"}))



def render_history() -> None:
    st.header("Historique")
    st.write("Suivez les dernières requêtes de prédiction et comparez les résultats.")

    if not st.session_state.history:
        st.info("Aucune prédiction enregistrée pour le moment.")
        return

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("Effacer l'historique"):
        st.session_state.history = []
        st.success("Historique effacé.")

    st.divider()
    st.markdown(
        """
        *Les données de l'historique sont conservées localement pendant la session.*
        """
    )


def render_model() -> None:
    st.header("Modèle")
    st.write(
        "Présentation du modèle utilisé pour la prédiction de revenu."
    )
    st.markdown(
        """
        - Type : modèle de classification binaire.
        - Jeu de données : Adult Income.
        - Objectif : prédire si le revenu est supérieur à 50K.
        - Variables utilisées : âge, poids final, niveau d'éducation, gains/pertes de capital, heures travaillées, et attributs catégoriels.
        """
    )
    st.markdown("### Statut du modèle")
    st.write("Ce modèle est mis à jour localement et utilisé pour alimenter le service de prédiction FastAPI.")


def render_evaluation() -> None:
    st.header("📊 Évaluation")
    st.write("Visualisations d'évaluation et diagnostics du modèle avec graphiques détaillés.")

    # If we have in-session history, show quick visuals
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)
        
        with col1:
            if "probability" in df.columns:
                st.subheader("📈 Distribution des probabilités")
                probs = df["probability"].dropna()
                if not probs.empty:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(probs, bins=20, color="#fb7185", edgecolor="#7f1d1d", alpha=0.7)
                    ax.set_xlabel("Probabilité", fontsize=10)
                    ax.set_ylabel("Effectif", fontsize=10)
                    ax.set_title("Distribution des scores", fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.info("Pas de probabilités disponibles dans l'historique.")

        with col2:
            if "status" in df.columns:
                st.subheader("📊 Répartition des statuts")
                counts = df["status"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ['#22c55e', '#ef4444']
                counts.plot(kind='bar', ax=ax, color=colors[:len(counts)])
                ax.set_xlabel("Statut", fontsize=10)
                ax.set_ylabel("Nombre", fontsize=10)
                ax.set_title("Répartition des prédictions", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # Additional metrics
        st.subheader("🎯 Métriques de performance")
        if "probability" in df.columns and "status" in df.columns:
            probs = df["probability"].dropna()
            if not probs.empty:
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Moyenne", f"{probs.mean():.2%}")
                with metric_col2:
                    st.metric("Médiane", f"{probs.median():.2%}")
                with metric_col3:
                    st.metric("Min", f"{probs.min():.2%}")
                with metric_col4:
                    st.metric("Max", f"{probs.max():.2%}")

        # Time series if we have timestamps
        if "timestamp" in df.columns and "probability" in df.columns:
            st.subheader("📉 Évolution des probabilités dans le temps")
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy = df_copy.sort_values('timestamp')
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_copy['timestamp'], df_copy['probability'], marker='o', linewidth=2, markersize=4, color='#8b5cf6')
            ax.set_xlabel("Temps", fontsize=10)
            ax.set_ylabel("Probabilité", fontsize=10)
            ax.set_title("Évolution des prédictions", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        st.subheader("📋 Aperçu des dernières requêtes")
        st.dataframe(df.head(50), use_container_width=True)
        return

    # Otherwise try to load a local evaluation sample and offer richer metrics
    st.info("Aucune prédiction en session. Chargement d'un échantillon d'évaluation si disponible.")
    try:
        sample = pd.read_csv("data/adult_prepared.csv", nrows=500)
        st.write("Aperçu des données d'évaluation")
        st.dataframe(sample.head())

        if "target" in sample.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Distribution de la cible")
                target_counts = sample["target"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = ['#3b82f6', '#ef4444']
                target_counts.plot(kind='bar', ax=ax, color=colors[:len(target_counts)])
                ax.set_xlabel("Classe", fontsize=10)
                ax.set_ylabel("Effectif", fontsize=10)
                ax.set_title("Distribution des classes", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)

            with col2:
                st.subheader("🎯 Statistiques descriptives")
                st.write(sample.describe())

            if st.button("🚀 Générer prédictions via l'API (nécessite API accessible)"):
                if api_health() != "connecté":
                    st.error("L'API n'est pas accessible. Vérifiez l'URL dans la configuration.")
                else:
                    n = min(200, len(sample))
                    X = sample.iloc[:n]
                    probs = []
                    y_true = X["target"].astype(int).tolist()
                    progress = st.progress(0)
                    for i, (_, row) in enumerate(X.iterrows()):
                        payload = {}
                        for feat in FEATURES.keys():
                            if feat in row.index:
                                payload[feat] = row[feat]
                        for cat in CATEGORICAL_OPTIONS.keys():
                            if cat in row.index:
                                payload[cat] = row[cat]

                        res = predict_remote(payload)
                        if res is None:
                            st.warning("Prédiction interrompue — vérifiez l'API.")
                            break
                        probs.append(res.get("probability", 0.0))
                        progress.progress(int((i + 1) / n * 100))

                    if len(probs) == len(y_true) and len(probs) > 1:
                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_true, probs)
                        roc_auc = auc(fpr, tpr)
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        ax1.plot(fpr, tpr, color="#8b5cf6", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
                        ax1.plot([0, 1], [0, 1], color="#94a3b8", lw=1, linestyle="--")
                        ax1.set_xlabel("False Positive Rate", fontsize=10)
                        ax1.set_ylabel("True Positive Rate", fontsize=10)
                        ax1.set_title("Courbe ROC", fontsize=12, fontweight='bold')
                        ax1.legend(loc="lower right")
                        ax1.grid(True, alpha=0.3)
                        st.pyplot(fig1)

                        # Confusion Matrix
                        thresh = st.session_state.threshold
                        y_pred = [1 if p >= thresh else 0 for p in probs]
                        cm = confusion_matrix(y_true, y_pred)
                        fig2, ax2 = plt.subplots(figsize=(5, 4))
                        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax2.figure.colorbar(im, ax=ax2)
                        ax2.set_title("Matrice de confusion", fontsize=12, fontweight='bold')
                        ax2.set_xlabel("Prédit", fontsize=10)
                        ax2.set_ylabel("Vrai", fontsize=10)
                        thresh_val = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax2.text(j, i, format(cm[i, j], 'd'),
                                        ha="center", va="center",
                                        color="white" if cm[i, j] > thresh_val else "black")
                        st.pyplot(fig2)

                        # Precision-Recall Curve
                        from sklearn.metrics import precision_recall_curve, average_precision_score
                        precision, recall, _ = precision_recall_curve(y_true, probs)
                        avg_precision = average_precision_score(y_true, probs)
                        fig3, ax3 = plt.subplots(figsize=(6, 4))
                        ax3.plot(recall, precision, color="#10b981", lw=2, label=f"AP = {avg_precision:.2f}")
                        ax3.set_xlabel("Recall", fontsize=10)
                        ax3.set_ylabel("Precision", fontsize=10)
                        ax3.set_title("Courbe Precision-Recall", fontsize=12, fontweight='bold')
                        ax3.legend(loc="lower left")
                        ax3.grid(True, alpha=0.3)
                        st.pyplot(fig3)

                        # Metrics Summary
                        st.subheader("📊 Résumé des métriques")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("AUC-ROC", f"{roc_auc:.3f}")
                        with metric_col2:
                            st.metric("Avg Precision", f"{avg_precision:.3f}")
                        with metric_col3:
                            st.metric("Total prédictions", len(probs))
                        with metric_col4:
                            st.metric("Seuil utilisé", f"{thresh:.2f}")
                    else:
                        st.warning("Impossible de calculer les métriques : prédictions incomplètes.")
    except Exception as exc:
        st.warning(f"Aucun fichier d'évaluation disponible localement: {exc}")


def render_architecture() -> None:
    st.header("Architecture")
    st.write(
        "Visualisation du flux de données, de l'entraînement jusqu'au déploiement de l'API et du frontend."
    )

    st.markdown(
        "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:1rem; margin-bottom:1.5rem;'>"
        "<div style='background:#ffffff; color:#0f172a; padding:1.3rem; border-radius:1.5rem; border:1px solid rgba(15,23,42,0.08); box-shadow:0 20px 40px rgba(15,23,42,0.06);'>"
        "<h3 style='margin-top:0;'>Données</h3>"
        "<p style='margin:0.5rem 0 0 0; color:#475569;'>Adult Income CSV propre et encodé.</p>"
        "</div>"
        "<div style='background:#eff6ff; color:#1d4ed8; padding:1.3rem; border-radius:1.5rem; box-shadow:0 20px 40px rgba(59,130,246,0.08);'>"
        "<h3 style='margin-top:0;'>Préparation</h3>"
        "<p style='margin:0.5rem 0 0 0; color:#1e40af;'>Nettoyage, encodage et normalisation des variables.</p>"
        "</div>"
        "<div style='background:#ecfdf5; color:#166534; padding:1.3rem; border-radius:1.5rem; box-shadow:0 20px 40px rgba(16,185,129,0.08);'>"
        "<h3 style='margin-top:0;'>Entraînement</h3>"
        "<p style='margin:0.5rem 0 0 0; color:#14532d;'>Optuna, GridSearchCV et suivi MLflow.</p>"
        "</div>"
        "</div>", unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='background:#0f172a; color:#f8fafc; padding:1.5rem; border-radius:1.5rem; box-shadow:0 24px 50px rgba(15,23,42,0.18); margin-bottom:1.5rem;'>"
        "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:1rem;'>"
        "<div>"
        "<h3 style='margin-top:0;'>Modèle</h3>"
        "<p style='margin:0.5rem 0 0 0; color:#cbd5e1;'>Pipeline scikit-learn sauvegardé sous `models/model.joblib`.</p>"
        "</div>"
        "<div>"
        "<h3 style='margin-top:0;'>API</h3>"
        "<p style='margin:0.5rem 0 0 0; color:#cbd5e1;'>FastAPI expose `/health` et `/predict` avec Uvicorn.</p>"
        "</div>"
        "<div>"
        "<h3 style='margin-top:0;'>Frontend</h3>"
        "<p style='margin:0.5rem 0 0 0; color:#cbd5e1;'>Streamlit affiche les métriques, les prédictions et l'historique.</p>"
        "</div>"
        "</div>"
        "</div>", unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Diagramme de flux")
    st.markdown(
        "<div style='display:flex; flex-wrap:wrap; gap:0.75rem; align-items:center;'>"
        "<div style='flex:1; min-width:180px; background:#eef2ff; color:#3730a3; padding:1rem 1.1rem; border-radius:1rem; text-align:center;'>Dataset</div>"
        "<div style='font-size:1.7rem; color:#0f172a;'>→</div>"
        "<div style='flex:1; min-width:180px; background:#e0f2fe; color:#0369a1; padding:1rem 1.1rem; border-radius:1rem; text-align:center;'>Préparation</div>"
        "<div style='font-size:1.7rem; color:#0f172a;'>→</div>"
        "<div style='flex:1; min-width:180px; background:#dcfce7; color:#166534; padding:1rem 1.1rem; border-radius:1rem; text-align:center;'>Entraînement</div>"
        "<div style='font-size:1.7rem; color:#0f172a;'>→</div>"
        "<div style='flex:1; min-width:180px; background:#fef3c7; color:#92400e; padding:1rem 1.1rem; border-radius:1rem; text-align:center;'>API</div>"
        "<div style='font-size:1.7rem; color:#0f172a;'>→</div>"
        "<div style='flex:1; min-width:180px; background:#fef2f2; color:#991b1b; padding:1rem 1.1rem; border-radius:1rem; text-align:center;'>Frontend</div>"
        "</div>", unsafe_allow_html=True,
    )


def main() -> None:
    st.sidebar.title("DOUAA EL KHEIR")
    st.sidebar.markdown("---")
    st.sidebar.title("Configuration")
    st.sidebar.markdown("### Liens rapides")
    st.sidebar.markdown(
        f"""
        <div class='sidebar-card'>
            <p style='margin:0 0 0.75rem 0; font-weight:700; color:#981b4a;'>🔗 Accès direct</p>
            <a class='hero-chip hero-chip-api' href='{API_PUBLIC_URL}' target='_blank'>🔧 API</a>
            <a class='hero-chip hero-chip-mlflow' href='{st.session_state.mlflow_url}' target='_blank'>📊 MLflow</a>
            <a class='hero-chip hero-chip-airflow' href='{st.session_state.airflow_url}' target='_blank'>⚙️ Airflow</a>
            <a class='hero-chip hero-chip-github' href='{GITHUB_URL}' target='_blank'>🐙 GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Paramètres de service")
    st.sidebar.text_input("URL de l'API", value=st.session_state.api_url, key="api_url")
    st.sidebar.text_input("URL de MLflow", value=st.session_state.mlflow_url, key="mlflow_url")
    st.sidebar.metric("Statut API", api_health())
    st.sidebar.slider(
        "Seuil de décision >50K",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.threshold,
        step=0.01,
        key="threshold",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("- Modèle chargé : Oui\n- Nombre de variables : 30")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(SECTIONS)
    with tab1:
        render_home()
    with tab2:
        render_prediction()
    with tab3:
        render_model()
    with tab4:
        render_architecture()
    with tab5:
        render_evaluation()
    with tab6:
        render_history()


if __name__ == "__main__":
    main()