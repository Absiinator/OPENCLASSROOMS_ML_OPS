"""
Dashboard Streamlit - Home Credit Scoring
=========================================

Interface utilisateur pour le scoring de crédit.
Version refactorisée avec modules séparés.

Fonctionnalités (selon exigences) :
- Visualiser le score, probabilité et interprétation intelligible
- Informations descriptives du client
- Comparaison avec l'ensemble des clients via graphiques
- Accessibilité WCAG (contrastes, labels, navigation)
- Rapport de Data Drift (Evidently)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

# Import des modules locaux
import api_client
import constants

check_api_health = api_client.check_api_health
get_model_info = api_client.get_model_info
predict_client = api_client.predict_client
explain_prediction = api_client.explain_prediction
get_feature_importance = api_client.get_feature_importance
API_URL = api_client.API_URL

from constants import (
    REQUIRED_FEATURES,
    FEATURE_EXPLANATIONS,
    MODEL_CONFIG,
    get_default_features,
    calculate_ratios,
    get_risk_category,
    get_risk_color
)

# ============================================
# Configuration de la page
# ============================================
st.set_page_config(
    page_title="Home Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS pour l'accessibilité WCAG
# ============================================
st.markdown("""
<style>
    /* Contrastes élevés pour l'accessibilité WCAG AA */
    .stMetric label { color: #1a1a1a !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0a0a0a !important; font-size: 1.5rem !important; }
    
    /* Focus visible pour navigation clavier */
    button:focus, input:focus, select:focus, a:focus {
        outline: 3px solid #005fcc !important;
        outline-offset: 2px !important;
    }
    
    /* Indicateurs visuels clairs avec patterns */
    .risk-low { background-color: #d4edda !important; color: #155724 !important; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #28a745; }
    .risk-high { background-color: #f8d7da !important; color: #721c24 !important; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# ============================================
# Variables globales
# ============================================
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
GITHUB_REPO_URL = os.getenv(
    "GITHUB_REPO_URL",
    "https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS"
).rstrip("/")
if GITHUB_REPO_URL.endswith(".git"):
    GITHUB_REPO_URL = GITHUB_REPO_URL[:-4]

# Chemins pour données et rapports
if os.path.exists("/app/data"):
    PROJECT_ROOT = "/app"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DRIFT_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "evidently_full_report.html")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "application_train.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "reports", "feature_importance.csv")
EXIGENCE_PATH = os.path.join(PROJECT_ROOT, "exigence.txt")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "model_config.json")


def github_blob(path: str) -> str:
    """Construit un lien GitHub vers un fichier du repo."""
    clean_path = path.lstrip("/")
    return f"{GITHUB_REPO_URL}/blob/main/{clean_path}"


def github_tree(path: str) -> str:
    """Construit un lien GitHub vers un dossier du repo."""
    clean_path = path.lstrip("/")
    return f"{GITHUB_REPO_URL}/tree/main/{clean_path}"


def resolve_github_link(target: str) -> str:
    """Résout un lien GitHub à partir d'un chemin interne ou d'une URL complète."""
    if target.startswith("http://") or target.startswith("https://"):
        return target
    if target.startswith("dir:"):
        return github_tree(target.replace("dir:", "", 1))
    return github_blob(target)


# ============================================
# Fonctions utilitaires
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    """Charge les données de référence pour comparaison (cache 1h)."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, nrows=10000)
            return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_top_features_from_report(top_n: int = 15) -> List[str]:
    """Charge les features les plus importantes depuis le rapport (notebooks)."""
    candidates = []
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        candidates.append(FEATURE_IMPORTANCE_PATH)

    # Chemin issu du run MLflow (si disponible)
    run_id = None
    if os.path.exists(MODEL_CONFIG_PATH):
        try:
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                run_id = json.load(f).get("run_id")
        except Exception:
            run_id = None
    if run_id:
        candidates.append(os.path.join(
            PROJECT_ROOT,
            "notebooks",
            "mlruns",
            "446811177754564983",
            run_id,
            "artifacts",
            "feature_importance.csv"
        ))

    # Chemin explicite connu (fallback)
    candidates.append(os.path.join(
        PROJECT_ROOT,
        "notebooks",
        "mlruns",
        "446811177754564983",
        "169b2338d7224da2801ea8dda14d64e3",
        "artifacts",
        "feature_importance.csv"
    ))

    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "feature" in df.columns and "importance" in df.columns:
                    df = df.sort_values("importance", ascending=False)
                    return df["feature"].head(top_n).tolist()
            except Exception:
                continue
    return []


CE_EVIDENCE = [
    {
        "title": "Stratégie de modélisation",
        "items": [
            {
                "id": "CE1",
                "text": "Variables catégorielles transformées (encodage).",
                "links": [
                    ("Notebook 02", "notebooks/02_Preprocessing_Features.ipynb"),
                    ("Préprocessing", "src/preprocessing.py")
                ],
            },
            {
                "id": "CE2",
                "text": "Création de nouvelles variables (feature engineering).",
                "links": [
                    ("Notebook 02", "notebooks/02_Preprocessing_Features.ipynb"),
                    ("Rapport features", "reports/new_features_correlations.png")
                ],
            },
            {
                "id": "CE3",
                "text": "Transformations mathématiques selon les distributions.",
                "links": [
                    ("Notebook 02", "notebooks/02_Preprocessing_Features.ipynb")
                ],
            },
            {
                "id": "CE4",
                "text": "Normalisation lorsque nécessaire.",
                "links": [
                    ("Notebook 02", "notebooks/02_Preprocessing_Features.ipynb"),
                    ("Préprocessing", "src/preprocessing.py")
                ],
            },
            {
                "id": "CE5",
                "text": "Stratégie d’élaboration du modèle alignée au besoin métier.",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb"),
                    ("Support soutenance", "presentation_outline.txt")
                ],
            },
            {
                "id": "CE6",
                "text": "Choix de la variable cible pertinente.",
                "links": [
                    ("Notebook 01", "notebooks/01_EDA.ipynb")
                ],
            },
            {
                "id": "CE7",
                "text": "Vérification du data leakage.",
                "links": [
                    ("Notebook 01", "notebooks/01_EDA.ipynb"),
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb")
                ],
            },
            {
                "id": "CE8",
                "text": "Tests d’algorithmes (linéaire & non‑linéaire).",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb")
                ],
            },
        ],
    },
    {
        "title": "Évaluation des performances",
        "items": [
            {
                "id": "CE1",
                "text": "Métrique adaptée + score métier (FN/FP).",
                "links": [
                    ("Metrics", "src/metrics.py"),
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb"),
                    ("Rapport métriques", "reports/metrics_report.txt")
                ],
            },
            {
                "id": "CE2",
                "text": "Autres indicateurs (ROC, confusion, etc.).",
                "links": [
                    ("ROC", "reports/roc_curve.png"),
                    ("Confusion", "reports/confusion_matrix.png"),
                    ("Rapport métriques", "reports/metrics_report.txt")
                ],
            },
            {
                "id": "CE3",
                "text": "Séparation train/test pour l’évaluation.",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb"),
                    ("Entraînement", "src/train.py")
                ],
            },
            {
                "id": "CE4",
                "text": "Modèle de référence (Dummy).",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb")
                ],
            },
            {
                "id": "CE5",
                "text": "Prise en compte du déséquilibre des classes.",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb"),
                    ("Entraînement", "src/train.py")
                ],
            },
            {
                "id": "CE6",
                "text": "Optimisation des hyper‑paramètres.",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb")
                ],
            },
            {
                "id": "CE7",
                "text": "Validation croisée (Grid/Random Search).",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb")
                ],
            },
            {
                "id": "CE8",
                "text": "Résultats présentés du simple au complexe + choix final.",
                "links": [
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb"),
                    ("Support soutenance", "presentation_outline.txt")
                ],
            },
            {
                "id": "CE9",
                "text": "Feature importance globale et locale.",
                "links": [
                    ("Importance globale", "reports/feature_importance.csv"),
                    ("Top 30", "reports/feature_importance_top30.png"),
                    ("Dashboard (local)", "streamlit_app/app.py")
                ],
            },
        ],
    },
    {
        "title": "Pipeline d’entraînement & registry",
        "items": [
            {
                "id": "CE1",
                "text": "Pipeline d’entraînement reproductible.",
                "links": [
                    ("Entraînement", "src/train.py"),
                    ("Préprocessing", "src/preprocessing.py"),
                    ("Notebook 03", "notebooks/03_Model_Training_MLflow.ipynb")
                ],
            },
            {
                "id": "CE2",
                "text": "Sérialisation & stockage des modèles.",
                "links": [
                    ("Modèles", "models/model_config.json"),
                    ("Registry MLflow", "dir:notebooks/mlruns")
                ],
            },
            {
                "id": "CE3",
                "text": "Mesures & résultats formalisés pour comparaison.",
                "links": [
                    ("Rapport métriques", "reports/metrics_report.txt"),
                    ("Registry MLflow", "dir:notebooks/mlruns")
                ],
            },
        ],
    },
    {
        "title": "Versioning du code",
        "items": [
            {
                "id": "CE1",
                "text": "Repo versionné Git + GitHub.",
                "links": [
                    ("GitHub", "https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS")
                ],
            },
            {
                "id": "CE2",
                "text": "Historique de commits (versions distinctes).",
                "links": [
                    ("Commits", "https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/commits/main")
                ],
            },
            {
                "id": "CE3",
                "text": "Packages & versions listés.",
                "links": [
                    ("environment.yml", "environment.yml"),
                    ("pyproject.toml", "pyproject.toml"),
                    ("API requirements", "api/requirements.txt")
                ],
            },
            {
                "id": "CE4",
                "text": "Fichier introductif & structure du projet.",
                "links": [
                    ("README", "README.md"),
                    ("README API", "api/README.md"),
                    ("README Dashboard", "streamlit_app/README.md")
                ],
            },
            {
                "id": "CE5",
                "text": "Scripts commentés pour réutilisation.",
                "links": [
                    ("Code source", "dir:src"),
                    ("API", "api/main.py")
                ],
            },
        ],
    },
    {
        "title": "Déploiement continu de l’API",
        "items": [
            {
                "id": "CE1",
                "text": "Pipeline CI/CD défini.",
                "links": [
                    ("Workflow", ".github/workflows/ci-cd.yml"),
                    ("Blueprint", "render.yaml")
                ],
            },
            {
                "id": "CE2",
                "text": "API déployée renvoie une prédiction.",
                "links": [
                    ("API", "api/main.py"),
                    ("README API", "api/README.md")
                ],
            },
            {
                "id": "CE3",
                "text": "Déploiement Cloud automatisé.",
                "links": [
                    ("render.yaml", "render.yaml"),
                    ("Guide Render", "RENDER_SETUP.md")
                ],
            },
            {
                "id": "CE4",
                "text": "Tests unitaires automatisés.",
                "links": [
                    ("Tests", "dir:tests"),
                    ("Workflow", ".github/workflows/ci-cd.yml")
                ],
            },
            {
                "id": "CE5",
                "text": "API indépendante du dashboard.",
                "links": [
                    ("API", "dir:api"),
                    ("Dashboard", "dir:streamlit_app"),
                    ("Blueprint", "render.yaml")
                ],
            },
        ],
    },
    {
        "title": "Suivi de performance & drift",
        "items": [
            {
                "id": "CE1",
                "text": "Stratégie de suivi (data drift).",
                "links": [
                    ("Notebook 04", "notebooks/04_Drift_Evidently.ipynb")
                ],
            },
            {
                "id": "CE2",
                "text": "Simulation + rapport Evidently.",
                "links": [
                    ("Rapport", "reports/evidently_full_report.html"),
                    ("Notebook 04", "notebooks/04_Drift_Evidently.ipynb")
                ],
            },
            {
                "id": "CE3",
                "text": "Analyse stabilité + actions proposées.",
                "links": [
                    ("Notebook 04", "notebooks/04_Drift_Evidently.ipynb"),
                    ("Rapport", "reports/evidently_full_report.html")
                ],
            },
        ],
    },
]


def render_ce_checklist() -> None:
    """Affiche la checklist CE avec preuves GitHub."""
    st.markdown("### ✅ Conformité CE (preuves GitHub)")
    with st.expander("Voir la checklist CE", expanded=False):
        for section in CE_EVIDENCE:
            st.markdown(f"**{section['title']}**")
            for item in section["items"]:
                links = []
                for label, target in item.get("links", []):
                    url = resolve_github_link(target)
                    links.append(f"[{label}]({url})")
                links_md = " • ".join(links) if links else "—"
                st.markdown(f"- **{item['id']}** {item['text']} — {links_md}")


def interpret_score(probability: float, threshold: float) -> dict:
    """Génère une interprétation textuelle pour non-experts."""
    if probability < threshold:
        decision = "ACCORDÉ"
        if probability < threshold * 0.5:
            confidence = "très élevée"
            explanation = "Profil très favorable. Risque de défaut minimal."
        else:
            confidence = "modérée"
            explanation = "Profil acceptable avec quelques points de vigilance."
    else:
        decision = "REFUSÉ"
        if probability > threshold * 1.5:
            confidence = "très élevée"
            explanation = "Profil à risque significatif. Défaut de paiement probable."
        else:
            confidence = "modérée"
            explanation = "Profil légèrement au-dessus du seuil de risque."
    
    return {
        "decision": decision,
        "confidence": confidence,
        "explanation": explanation,
        "probability_text": f"{probability*100:.1f}%",
        "threshold_text": f"{threshold*100:.1f}%"
    }


def get_feature_explanation(feature_name: str) -> str:
    """Retourne une explication en langage naturel d'une feature."""
    return FEATURE_EXPLANATIONS.get(feature_name, get_feature_label(feature_name))


FEATURE_LABELS = {k: v.get("label", k) for k, v in REQUIRED_FEATURES.items()}

# Libellés explicites pour les colonnes "brutes" du dataset
FEATURE_LABEL_OVERRIDES = {
    "SK_ID_CURR": "ID client",
    "TARGET": "Défaut (cible)",
    "NAME_CONTRACT_TYPE": "Type de contrat",
    "CODE_GENDER": "Genre",
    "FLAG_OWN_CAR": "Possède une voiture",
    "FLAG_OWN_REALTY": "Propriétaire immobilier",
    "NAME_TYPE_SUITE": "Type d’accompagnement",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_EDUCATION_TYPE": "Niveau d’éducation",
    "NAME_FAMILY_STATUS": "Statut familial",
    "NAME_HOUSING_TYPE": "Type de logement",
    "REGION_POPULATION_RELATIVE": "Population relative de la région",
    "DAYS_BIRTH": "Âge (jours)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours)",
    "DAYS_REGISTRATION": "Jours depuis inscription",
    "DAYS_ID_PUBLISH": "Jours depuis publication ID",
    "OWN_CAR_AGE": "Âge du véhicule (années)",
    "FLAG_MOBIL": "Téléphone mobile fourni",
    "FLAG_EMP_PHONE": "Téléphone employeur fourni",
    "FLAG_WORK_PHONE": "Téléphone travail fourni",
    "FLAG_CONT_MOBILE": "Téléphone mobile joignable",
    "FLAG_PHONE": "Téléphone fixe fourni",
    "FLAG_EMAIL": "Email fourni",
    "CNT_FAM_MEMBERS": "Nombre de membres du foyer",
    "REGION_RATING_CLIENT": "Note de la région",
    "REGION_RATING_CLIENT_W_CITY": "Note région (avec ville)",
    "WEEKDAY_APPR_PROCESS_START": "Jour de la semaine de la demande",
    "HOUR_APPR_PROCESS_START": "Heure de la demande",
    "REG_REGION_NOT_LIVE_REGION": "Région d’enregistrement ≠ résidence",
    "REG_REGION_NOT_WORK_REGION": "Région d’enregistrement ≠ travail",
    "LIVE_REGION_NOT_WORK_REGION": "Région résidence ≠ travail",
    "REG_CITY_NOT_LIVE_CITY": "Ville d’enregistrement ≠ résidence",
    "REG_CITY_NOT_WORK_CITY": "Ville d’enregistrement ≠ travail",
    "LIVE_CITY_NOT_WORK_CITY": "Ville résidence ≠ travail",
    "ORGANIZATION_TYPE": "Type d’organisation",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "DAYS_LAST_PHONE_CHANGE": "Jours depuis dernier changement de téléphone",
    "FONDKAPREMONT_MODE": "Fonds de rénovation (mode)",
    "HOUSETYPE_MODE": "Type de maison (mode)",
    "WALLSMATERIAL_MODE": "Matériau des murs (mode)",
    "EMERGENCYSTATE_MODE": "État d’urgence (mode)",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Observations (30 jours)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Défauts (30 jours)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Observations (60 jours)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Défauts (60 jours)",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Demandes bureau de crédit (heure)",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Demandes bureau de crédit (jour)",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Demandes bureau de crédit (semaine)",
    "AMT_REQ_CREDIT_BUREAU_MON": "Demandes bureau de crédit (mois)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Demandes bureau de crédit (trimestre)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Demandes bureau de crédit (année)",
    "PREV_SK_ID_PREV_COUNT": "Demandes précédentes - Nombre de dossiers",
    "partial_payment_rate": "Taux de paiements partiels",
    "CC_CNT_DRAWINGS_CURRENT_MAX": "Carte de crédit - Nombre de tirages (max)",
    "CC_AMT_BALANCE_MEAN": "Carte de crédit - Solde (moyenne)",
    "PREV_AMT_APPLICATION_MEAN": "Demandes précédentes - Montant demandé (moyenne)",
    "PREV_CREDIT_APPLICATION_RATIO_MEAN": "Demandes précédentes - Ratio crédit/demande (moyenne)",
    "BUREAU_AMT_CREDIT_SUM_LIMIT_MAX": "Bureau - Limite de crédit cumulée (max)",
    "BUREAU_DAYS_CREDIT_MAX": "Bureau - Jours depuis crédit (max)",
    "BUREAU_DAYS_CREDIT_MIN": "Bureau - Jours depuis crédit (min)",
    "NONLIVINGAREA_AVG": "Surface non habitable (moyenne)"
}

FEATURE_LABELS.update(FEATURE_LABEL_OVERRIDES)

STAT_SUFFIXES = {
    "AVG": "moyenne",
    "MEAN": "moyenne",
    "MEDI": "médiane",
    "MEDIAN": "médiane",
    "MODE": "mode",
    "MAX": "max",
    "MIN": "min",
    "STD": "écart type",
    "SUM": "somme",
    "COUNT": "nb",
}

PREFIX_LABELS = {
    "BUREAU": "Bureau",
    "PREV": "Demandes précédentes",
    "POS": "POS Cash",
    "CC": "Carte de crédit",
    "INSTAL": "Remboursements",
}

TOKEN_LABELS = {
    "AMT": "Montant",
    "CNT": "Nombre",
    "DAYS": "Jours",
    "FLAG": "Indicateur",
    "EXT": "Score externe",
    "SOURCE": "Source",
    "REGION": "Région",
    "CITY": "Ville",
    "CREDIT": "Crédit",
    "INCOME": "Revenu",
    "ANNUITY": "Annuité",
    "GOODS": "Bien",
    "PRICE": "Prix",
    "BALANCE": "Solde",
    "LIMIT": "Limite",
    "CURRENT": "Actuel",
    "DRAWINGS": "Tirages",
    "PAYMENT": "Paiement",
    "APPLICATION": "Demande",
    "APP": "Demande",
    "PARTIAL": "Partiel",
    "RATE": "Taux",
    "RATIO": "Ratio",
    "SUM": "Somme",
    "RATING": "Note",
    "WEEKDAY": "Jour de semaine",
    "HOUR": "Heure",
    "NAME": "",
    "TYPE": "Type",
    "SUITE": "Accompagnement",
    "EDUCATION": "Éducation",
    "FAMILY": "Familial",
    "HOUSING": "Logement",
    "ORGANIZATION": "Organisation",
    "AGE": "Âge",
    "OWN": "Possession",
    "CAR": "Voiture",
    "REALTY": "Immobilier",
    "MOBIL": "Mobile",
    "EMAIL": "Email",
    "PHONE": "Téléphone",
    "EMP": "Emploi",
    "WORK": "Travail",
    "CONTACT": "Contact",
    "LIVE": "Résidence",
    "NOT": "≠",
    "APARTMENTS": "Appartements",
    "BASEMENTAREA": "Surface sous-sol",
    "COMMONAREA": "Surface commune",
    "LANDAREA": "Surface terrain",
    "LIVINGAPARTMENTS": "Appartements habitables",
    "LIVINGAREA": "Surface habitable",
    "NONLIVINGAPARTMENTS": "Appartements non habitables",
    "NONLIVINGAREA": "Surface non habitable",
    "TOTALAREA": "Surface totale",
    "FLOORSMAX": "Étages max",
    "FLOORSMIN": "Étages min",
    "ENTRANCES": "Entrées",
    "ELEVATORS": "Ascenseurs",
    "YEARS": "Années",
    "BEGINEXPLUATATION": "Début exploitation",
    "BUILD": "Construction",
    "FONDKAPREMONT": "Fonds de rénovation",
    "HOUSETYPE": "Type de maison",
    "WALLSMATERIAL": "Matériau des murs",
    "EMERGENCYSTATE": "État d’urgence",
    "OBS": "Observations",
    "DEF": "Défauts",
    "SOCIAL": "Social",
    "CIRCLE": "Cercle",
    "DOCUMENT": "Document",
    "REQ": "Demandes",
    "BUREAU": "Bureau",
    "QRT": "Trimestre",
    "YEAR": "Année",
    "MON": "Mois",
    "WEEK": "Semaine",
    "DAY": "Jour",
    "ID": "ID",
    "SK": "ID",
    "CURR": "Courant",
    "PREV": "Précédent"
}


def _humanize_feature_name(feature_name: str) -> str:
    """Transforme un nom de colonne en libellé explicite."""
    name = re.sub(r"_x$|_y$", "", feature_name, flags=re.IGNORECASE)
    name = re.sub(r"__+", "_", name)

    # Cas spécifiques (ex: FLAG_DOCUMENT_3)
    doc_match = re.match(r"FLAG_DOCUMENT_(\d+)", name)
    if doc_match:
        return f"Document {doc_match.group(1)} fourni"

    parts = [p for p in name.split("_") if p]
    if not parts:
        return feature_name

    group_label = None
    first = parts[0].upper()
    if first in PREFIX_LABELS:
        group_label = PREFIX_LABELS[first]
        parts = parts[1:]

    stat_suffix = None
    if parts:
        last = parts[-1].upper()
        if last in STAT_SUFFIXES:
            stat_suffix = STAT_SUFFIXES[last]
            parts = parts[:-1]

    label_parts = []
    for part in parts:
        upper = part.upper()
        if upper in TOKEN_LABELS:
            token_label = TOKEN_LABELS[upper]
            if token_label:
                label_parts.append(token_label)
            continue
        if upper.isdigit():
            label_parts.append(upper)
            continue
        label_parts.append(part.replace("-", " ").title())

    label = " ".join(label_parts).strip() if label_parts else feature_name
    if group_label:
        label = f"{group_label} - {label}"
    if stat_suffix:
        label = f"{label} ({stat_suffix})"
    return label


def get_feature_label(feature_name: str) -> str:
    """Retourne un libellé explicite pour une feature."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    normalized = re.sub(r"_x$|_y$", "", feature_name, flags=re.IGNORECASE)
    if normalized in FEATURE_LABELS:
        return FEATURE_LABELS[normalized]
    return _humanize_feature_name(feature_name)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_reference_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calcule les valeurs par défaut (médiane/mode) des features de référence."""
    if df is None or df.empty:
        return {"numeric": {}, "categorical": {}}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_medians = df[numeric_cols].median(numeric_only=True).to_dict()

    cat_cols = [c for c in df.columns if c not in numeric_cols]
    cat_modes = {}
    for col in cat_cols:
        try:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                cat_modes[col] = mode.iloc[0]
        except Exception:
            continue

    return {"numeric": numeric_medians, "categorical": cat_modes}


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_categories(df: pd.DataFrame, column: str, max_items: int = 20) -> List[str]:
    """Retourne les catégories les plus fréquentes pour une colonne."""
    if df is None or df.empty or column not in df.columns:
        return []
    series = df[column].dropna().astype(str)
    if series.empty:
        return []
    return series.value_counts().head(max_items).index.tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_numeric_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calcule min/max et quantiles pour les sliders numériques."""
    if df is None or df.empty:
        return {}
    ranges = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        min_val = float(series.min())
        max_val = float(series.max())
        p5 = float(series.quantile(0.05))
        p95 = float(series.quantile(0.95))
        sample = series.head(200)
        is_int = bool(((sample % 1) == 0).all())
        ranges[col] = {
            "min": min_val,
            "max": max_val,
            "p5": p5,
            "p95": p95,
            "is_int": is_int
        }
    return ranges


def is_bounded_numeric(feature_name: str, range_info: Dict[str, Any]) -> bool:
    """Détermine si une variable numérique doit être affichée en slider."""
    if not range_info:
        return False
    min_val = range_info.get("min")
    max_val = range_info.get("max")
    if min_val is None or max_val is None:
        return False
    if pd.isna(min_val) or pd.isna(max_val):
        return False
    name = feature_name.upper()
    # 0/1, pourcentages ou ratios
    if 0 <= min_val and max_val <= 1:
        return True
    if any(tok in name for tok in ["RATIO", "RATE", "PERCENT", "SHARE"]) and 0 <= min_val and max_val <= 100:
        return True
    # Petites échelles discrètes
    if range_info.get("is_int") and 0 <= min_val and max_val <= 5:
        return True
    return False


def detect_multitag(values: List[str]) -> Optional[str]:
    """Détecte un séparateur multi-tags si présent dans les valeurs."""
    delimiters = ["|", ";", ",", " / "]
    for d in delimiters:
        with_delim = [v for v in values if d in v]
        if len(with_delim) < 3:
            continue
        tags = []
        for val in with_delim:
            tags.extend(split_multitag_value(val, d))
        unique_tags = set(tags)
        avg_tags = len(tags) / max(len(with_delim), 1)
        # Heuristique: plusieurs valeurs et vrai découpage en tags
        if avg_tags >= 1.5 and len(unique_tags) >= 3:
            return d
    for val in values:
        if (val.startswith("[") and val.endswith("]")) or (val.startswith("(") and val.endswith(")")):
            return ","
    return None


def split_multitag_value(value: str, delimiter: str) -> List[str]:
    """Découpe une valeur multi-tags en liste de tags."""
    if value is None:
        return []
    val = str(value).strip()
    if (val.startswith("[") and val.endswith("]")) or (val.startswith("(") and val.endswith(")")):
        val = val[1:-1]
    parts = [p.strip() for p in val.split(delimiter)]
    return [p for p in parts if p]


def render_numeric_inputs(
    features_list: List[str],
    ref_stats: Dict[str, Dict[str, Any]],
    numeric_ranges: Dict[str, Dict[str, Any]],
    key_prefix: str,
    columns: int = 2,
    show_raw_for_bounded: bool = False
) -> None:
    """Affiche des inputs numériques (slider si borné, sinon input)."""
    if not features_list:
        return
    cols = st.columns(columns)
    for idx, feat in enumerate(sorted(features_list, key=get_feature_label)):
        default_val = ref_stats["numeric"].get(feat, 0.0)
        if default_val is None or pd.isna(default_val):
            default_val = 0.0
        current_val = st.session_state.client_features.get(feat, default_val)
        range_info = numeric_ranges.get(feat, {})
        with cols[idx % columns]:
            if is_bounded_numeric(feat, range_info):
                min_val = range_info.get("min", default_val)
                max_val = range_info.get("max", default_val)
                if min_val == max_val:
                    min_val, max_val = default_val - 1, default_val + 1
                step = 1.0 if range_info.get("is_int") else 0.01
                value = st.slider(
                    get_feature_label(feat),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(current_val) if current_val is not None and not pd.isna(current_val) else float(default_val),
                    step=step,
                    help=get_feature_explanation(feat),
                    key=f"{key_prefix}_slider_{feat}"
                )
                if show_raw_for_bounded:
                    value = st.number_input(
                        f"{get_feature_label(feat)} (valeur)",
                        value=float(value),
                        help=get_feature_explanation(feat),
                        key=f"{key_prefix}_input_{feat}"
                    )
            else:
                value = st.number_input(
                    get_feature_label(feat),
                    value=float(current_val) if current_val is not None and not pd.isna(current_val) else float(default_val),
                    help=get_feature_explanation(feat),
                    key=f"{key_prefix}_input_{feat}"
                )
            st.session_state.client_features[feat] = value


def render_categorical_inputs(
    features_list: List[str],
    ref_data: pd.DataFrame,
    ref_stats: Dict[str, Dict[str, Any]],
    key_prefix: str,
    columns: int = 2
) -> None:
    """Affiche des inputs catégoriels (selectbox ou multi-tags)."""
    if not features_list:
        return
    cols = st.columns(columns)
    for idx, feat in enumerate(sorted(features_list, key=get_feature_label)):
        default_val = ref_stats["categorical"].get(feat, "MISSING")
        if default_val is None or pd.isna(default_val):
            default_val = "MISSING"
        choices = get_top_categories(ref_data, feat, max_items=40)
        default_val = str(default_val)
        if default_val not in choices:
            choices = [default_val] + choices
        if "MISSING" not in choices:
            choices.append("MISSING")
        current_val = st.session_state.client_features.get(feat, default_val)
        current_val = str(current_val) if current_val is not None else default_val
        if current_val not in choices:
            choices = [current_val] + choices
        delimiter = detect_multitag(choices)
        with cols[idx % columns]:
            if delimiter:
                with st.expander(get_feature_label(feat), expanded=False):
                    st.caption("Sélection multiple possible")
                    tag_set = set()
                    for val in choices:
                        tag_set.update(split_multitag_value(val, delimiter))
                    tags = sorted(tag_set)
                    col_a, col_b = st.columns(2)
                    if col_a.button("Tout sélectionner", key=f"{key_prefix}_{feat}_select_all"):
                        for tag in tags:
                            st.session_state[f"{key_prefix}_{feat}_tag_{tag}"] = True
                    if col_b.button("Tout désélectionner", key=f"{key_prefix}_{feat}_clear_all"):
                        for tag in tags:
                            st.session_state[f"{key_prefix}_{feat}_tag_{tag}"] = False
                    selected = []
                    for tag in tags:
                        key = f"{key_prefix}_{feat}_tag_{tag}"
                        if key not in st.session_state:
                            st.session_state[key] = tag in split_multitag_value(current_val, delimiter)
                        if st.checkbox(tag, key=key):
                            selected.append(tag)
                    if selected:
                        st.session_state.client_features[feat] = delimiter.join(selected)
                    else:
                        st.session_state.client_features[feat] = "MISSING"
            else:
                st.session_state.client_features[feat] = st.selectbox(
                    get_feature_label(feat),
                    options=choices,
                    index=choices.index(current_val),
                    help=get_feature_explanation(feat),
                    key=f"{key_prefix}_select_{feat}"
                )


def check_mlflow_health(url: str) -> bool:
    """Vérifie si l'UI MLflow est accessible."""
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def create_gauge_chart(probability: float, threshold: float) -> go.Figure:
    """Crée une jauge visuelle du score de risque."""
    color = get_risk_color(probability)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 40}},
        delta={'reference': threshold * 100, 'position': "bottom", 'relative': False},
        title={'text': "Probabilité de défaut", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},
                {'range': [20, 40], 'color': '#fff3cd'},
                {'range': [40, 60], 'color': '#ffeaa7'},
                {'range': [60, 80], 'color': '#f8d7da'},
                {'range': [80, 100], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "#1a1a1a"}
    )
    return fig


def create_comparison_chart(
    client_value: float,
    feature_name: str,
    reference_data: pd.DataFrame,
    group_filter: Optional[str] = None
) -> Optional[go.Figure]:
    """Crée un histogramme de comparaison client vs population."""
    if feature_name not in reference_data.columns:
        return None

    data = reference_data[feature_name].dropna()

    if group_filter and group_filter != "Tous les clients":
        if "TARGET" in reference_data.columns:
            if group_filter == "Clients sans défaut (TARGET=0)":
                data = reference_data[reference_data["TARGET"] == 0][feature_name].dropna()
            elif group_filter == "Clients en défaut (TARGET=1)":
                data = reference_data[reference_data["TARGET"] == 1][feature_name].dropna()

    if data.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        name="Distribution",
        opacity=0.7,
        marker_color="#4169E1"
    ))

    label = get_feature_label(feature_name)
    fig.add_vline(
        x=client_value,
        line_dash="dash",
        line_color="#C41E3A",
        line_width=3,
        annotation_text=f"Client: {client_value:.2f}",
        annotation_position="top"
    )

    percentile = (data < client_value).mean() * 100
    fig.update_layout(
        title=f"Distribution de {label} (client au {percentile:.0f}e percentile)",
        xaxis_title=label,
        yaxis_title="Nombre de clients",
        height=400,
        showlegend=False
    )

    return fig


def create_radar_comparison(
    client_features: Dict[str, float],
    reference_data: pd.DataFrame,
    selected_features: List[str]
) -> Optional[go.Figure]:
    """Crée un graphique radar pour comparer plusieurs features."""
    normalized_client = []
    normalized_mean = []
    valid_features = []

    for feat in selected_features:
        if feat in reference_data.columns and feat in client_features:
            if client_features.get(feat) is None or pd.isna(client_features.get(feat)):
                continue
            ref_data = reference_data[feat].dropna()
            if ref_data.empty:
                continue
            min_val, max_val = ref_data.min(), ref_data.max()
            if max_val > min_val:
                client_norm = (client_features[feat] - min_val) / (max_val - min_val)
                mean_norm = (ref_data.mean() - min_val) / (max_val - min_val)
            else:
                client_norm, mean_norm = 0.5, 0.5
            normalized_client.append(client_norm)
            normalized_mean.append(mean_norm)
            valid_features.append(feat)

    if not normalized_client or len(valid_features) < 3:
        return None
    labels = [get_feature_label(f) for f in valid_features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_mean + [normalized_mean[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.3)',
        line_color='#4169E1',
        name='Moyenne population'
    ))
    fig.add_trace(go.Scatterpolar(
        r=normalized_client + [normalized_client[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(196, 30, 58, 0.3)',
        line_color='#C41E3A',
        name='Client actuel'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title={'text': "Comparaison multi-critères", 'font': {'size': 16}},
        height=500,
        paper_bgcolor='white'
    )

    return fig


# ============================================
# Interface utilisateur principale
# ============================================

def render_sidebar(reference_data: pd.DataFrame):
    """Affiche la sidebar avec navigation, docs et statuts."""
    with st.sidebar:
        st.title("🏦 Home Credit")
        st.divider()

        # Navigation
        st.header("📍 Navigation")
        nav_options = [
            ("🎯 Scoring", "scoring"),
            ("📊 Dataset", "dataset"),
            ("📈 Data Drift", "drift"),
            ("📖 Documentation", "docs")
        ]
        if "current_page" not in st.session_state:
            st.session_state.current_page = "scoring"
        for label, page_key in nav_options:
            btn_type = "primary" if st.session_state.current_page == page_key else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state.current_page = page_key
                st.rerun()

        st.divider()

        # Documentation
        st.header("🔗 Documentation")
        st.link_button("Swagger", f"{API_URL}/docs", use_container_width=True)
        st.link_button("MLflow UI", MLFLOW_URL, use_container_width=True)
        st.link_button("README Projet", github_blob("README.md"), use_container_width=True)
        st.link_button("Guide Render", github_blob("RENDER_SETUP.md"), use_container_width=True)
        st.link_button("README API", github_blob("api/README.md"), use_container_width=True)
        st.link_button("README Dashboard", github_blob("streamlit_app/README.md"), use_container_width=True)
        st.link_button("Notebooks (dossier)", github_tree("notebooks"), use_container_width=True)
        with st.expander("Notebooks (liens directs)", expanded=False):
            st.link_button("01_EDA", github_blob("notebooks/01_EDA.ipynb"), use_container_width=True)
            st.link_button("02_Preprocessing", github_blob("notebooks/02_Preprocessing_Features.ipynb"), use_container_width=True)
            st.link_button("03_Model_Training", github_blob("notebooks/03_Model_Training_MLflow.ipynb"), use_container_width=True)
            st.link_button("04_Drift", github_blob("notebooks/04_Drift_Evidently.ipynb"), use_container_width=True)
        st.link_button("Repo GitHub", GITHUB_REPO_URL, use_container_width=True)

        st.divider()

        # État des services
        st.header("🏥 État")
        api_ok = check_api_health()
        mlflow_ok = check_mlflow_health(MLFLOW_URL)
        drift_exists = os.path.exists(DRIFT_REPORT_PATH)
        st.write(f"{'✅' if api_ok else '⚠️'} API: {'OK' if api_ok else 'Hors ligne'}")
        st.write(f"{'✅' if mlflow_ok else '⚠️'} MLflow: {'OK' if mlflow_ok else 'Hors ligne'}")
        st.write(f"{'✅' if drift_exists else '⚠️'} Drift: {'OK' if drift_exists else 'Absent'}")

        st.divider()

        # Modèle
        st.header("🤖 Modèle")
        model_info = get_model_info()
        if model_info:
            st.write(f"Nom: **{model_info.get('model_name', 'N/A')}**")
            st.write(f"Version: **{model_info.get('version', 'N/A')}**")
            training_date = model_info.get("training_date") or "N/A"
            st.write(f"Date modèle: **{training_date}**")
        else:
            st.caption("Infos indisponibles")

        st.divider()
        st.caption("v1.0.0 • Home Credit Scoring")


def render_prediction_tab():
    """Onglet principal: Scoring et prédiction."""
    st.header("🎯 Scoring de crédit")
    
    # Vérifier l'API (ne bloque pas la comparaison)
    api_ok = check_api_health()
    if not api_ok:
        st.warning("⚠️ API indisponible. La comparaison reste possible, la prédiction est désactivée.")
        st.info(f"API_URL: {API_URL}")

    ref_data = load_reference_data()
    ref_stats = compute_reference_stats(ref_data)
    numeric_ranges = compute_numeric_ranges(ref_data)
    top_from_report = load_top_features_from_report(25)
    important_features = set(top_from_report + list(REQUIRED_FEATURES.keys()))
    
    st.markdown("### Saisie des informations client")
    
    # Initialiser les features avec les valeurs par défaut
    if 'client_features' not in st.session_state:
        st.session_state.client_features = get_default_features()
    
    # Formulaire de saisie en colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Finances")
        st.session_state.client_features["AMT_INCOME_TOTAL"] = st.number_input(
            "Revenu annuel (€)",
            min_value=0.0, max_value=10000000.0,
            value=float(st.session_state.client_features.get("AMT_INCOME_TOTAL", 150000)),
            step=10000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_INCOME_TOTAL", "")
        )
        st.session_state.client_features["AMT_CREDIT"] = st.number_input(
            "Montant crédit (€)",
            min_value=0.0, max_value=5000000.0,
            value=float(st.session_state.client_features.get("AMT_CREDIT", 500000)),
            step=10000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_CREDIT", "")
        )
        st.session_state.client_features["AMT_ANNUITY"] = st.number_input(
            "Annuité (€/an)",
            min_value=0.0, max_value=500000.0,
            value=float(st.session_state.client_features.get("AMT_ANNUITY", 25000)),
            step=1000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_ANNUITY", "")
        )
        st.session_state.client_features["AMT_GOODS_PRICE"] = st.number_input(
            "Prix du bien (€)",
            min_value=0.0, max_value=5000000.0,
            value=float(st.session_state.client_features.get("AMT_GOODS_PRICE", 450000)),
            step=10000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_GOODS_PRICE", "")
        )
    
    with col2:
        st.markdown("#### 👤 Personnel")
        age_years = st.slider(
            "Âge (années)",
            min_value=18, max_value=80,
            value=35,
            help="Converti en jours pour le modèle"
        )
        st.session_state.client_features["DAYS_BIRTH"] = -age_years * 365
        
        exp_years = st.slider(
            "Ancienneté emploi (années)",
            min_value=0, max_value=50,
            value=5,
            help="Converti en jours pour le modèle"
        )
        st.session_state.client_features["DAYS_EMPLOYED"] = -exp_years * 365
        
        st.session_state.client_features["CNT_CHILDREN"] = st.number_input(
            "Nombre d'enfants",
            min_value=0, max_value=20,
            value=int(st.session_state.client_features.get("CNT_CHILDREN", 1))
        )
        
        gender = st.selectbox("Genre", ["Homme", "Femme"])
        st.session_state.client_features["CODE_GENDER_M"] = 1 if gender == "Homme" else 0
        
        st.session_state.client_features["FLAG_OWN_CAR"] = 1 if st.checkbox("Possède une voiture", value=True) else 0
        st.session_state.client_features["FLAG_OWN_REALTY"] = 1 if st.checkbox("Propriétaire immobilier", value=True) else 0
    
    with col3:
        st.markdown("#### 📊 Scores externes")
        st.session_state.client_features["EXT_SOURCE_1"] = st.slider(
            "Score externe 1",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.client_features.get("EXT_SOURCE_1", 0.5)),
            step=0.01,
            help=FEATURE_EXPLANATIONS.get("EXT_SOURCE_1", "")
        )
        st.session_state.client_features["EXT_SOURCE_2"] = st.slider(
            "Score externe 2",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.client_features.get("EXT_SOURCE_2", 0.6)),
            step=0.01,
            help=FEATURE_EXPLANATIONS.get("EXT_SOURCE_2", "")
        )
        st.session_state.client_features["EXT_SOURCE_3"] = st.slider(
            "Score externe 3",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.client_features.get("EXT_SOURCE_3", 0.55)),
            step=0.01,
            help=FEATURE_EXPLANATIONS.get("EXT_SOURCE_3", "")
        )
        st.session_state.client_features["REGION_RATING_CLIENT"] = st.selectbox(
            "Note région",
            options=[1, 2, 3],
            index=1,
            help="1=faible risque, 3=risque élevé"
        )

    # Variables complémentaires (optionnelles)
    numeric_cols = ref_data.select_dtypes(include=[np.number]).columns.tolist() if not ref_data.empty else []
    categorical_cols = [c for c in ref_data.columns if c not in numeric_cols] if not ref_data.empty else []
    exclude_cols = ["SK_ID_CURR", "TARGET", "index"]
    extra_numeric = [f for f in numeric_cols if f not in exclude_cols and f not in REQUIRED_FEATURES]
    extra_categorical = [f for f in categorical_cols if f not in exclude_cols and f not in REQUIRED_FEATURES]
    if extra_numeric or extra_categorical:
        important_numeric = [f for f in extra_numeric if f in important_features]
        other_numeric = [f for f in extra_numeric if f not in important_features]
        important_cat = [f for f in extra_categorical if f in important_features]
        other_cat = [f for f in extra_categorical if f not in important_features]

        if important_numeric:
            st.subheader("⭐ Variables numériques importantes (supplémentaires)")
            render_numeric_inputs(
                important_numeric,
                ref_stats,
                numeric_ranges,
                key_prefix="important_num",
                show_raw_for_bounded=True
            )

        if important_cat:
            st.subheader("⭐ Variables catégorielles importantes (supplémentaires)")
            render_categorical_inputs(
                important_cat,
                ref_data,
                ref_stats,
                key_prefix="important_cat"
            )

        if other_numeric or other_cat:
            with st.expander("➕ Variables avancées (moins importantes)", expanded=False):
                st.caption("Toutes les variables avancées sont affichées pour modification directe.")
                if other_numeric:
                    st.markdown("**Variables numériques avancées**")
                    render_numeric_inputs(
                        other_numeric,
                        ref_stats,
                        numeric_ranges,
                        key_prefix="other_num"
                    )

                if other_cat:
                    st.markdown("**Variables catégorielles avancées**")
                    render_categorical_inputs(
                        other_cat,
                        ref_data,
                        ref_stats,
                        key_prefix="other_cat"
                    )
    
    # Calculer les ratios automatiquement
    features = calculate_ratios(st.session_state.client_features)
    st.session_state.current_features = features
    
    st.markdown("---")

    # Comparaison population (sans prédiction)
    render_comparison_section(features, ref_data, show_header=True)

    st.markdown("---")

    # Bouton de prédiction
    if st.button("🔮 Calculer le score", type="primary", use_container_width=True, disabled=not api_ok):
        with st.spinner("Calcul en cours..."):
            result = predict_client(features)
        
        if result and "error" not in result:
            # Afficher les résultats
            probability = result.get("probability", 0)
            threshold = result.get("threshold", 0.44)
            decision = result.get("decision", "")
            
            # Interprétation
            interpretation = interpret_score(probability, threshold)
            
            # Layout résultats
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                # Jauge de risque
                fig_gauge = create_gauge_chart(probability, threshold)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with res_col2:
                # Décision
                if decision == "ACCEPTÉ":
                    st.markdown(f'<div class="risk-low"><h2>✅ {decision}</h2></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-high"><h2>❌ {decision}</h2></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                ### Interprétation
                
                - **Probabilité de défaut**: {interpretation['probability_text']}
                - **Seuil de décision**: {interpretation['threshold_text']}
                - **Confiance**: {interpretation['confidence']}
                
                {interpretation['explanation']}
                """)
            
            # Stocker pour comparaison
            st.session_state.last_prediction = result
            st.session_state.last_features = features

            # Résumé descriptif du client
            st.subheader("👤 Résumé du profil client")
            profile_col1, profile_col2, profile_col3 = st.columns(3)
            with profile_col1:
                st.markdown("**Situation financière**")
                st.write(f"- Revenu: {features['AMT_INCOME_TOTAL']:,.0f} €")
                st.write(f"- Crédit demandé: {features['AMT_CREDIT']:,.0f} €")
                st.write(f"- Ratio crédit/revenu: {features['CREDIT_INCOME_RATIO']:.2f}")
            with profile_col2:
                st.markdown("**Situation personnelle**")
                age_years = abs(int(features['DAYS_BIRTH'])) // 365
                employed_years = abs(int(features['DAYS_EMPLOYED'])) // 365
                st.write(f"- Âge: {age_years} ans")
                st.write(f"- Ancienneté emploi: {employed_years} ans")
                st.write(f"- Enfants: {features['CNT_CHILDREN']}")
            with profile_col3:
                st.markdown("**Scores de crédit**")
                st.write(f"- Score moyen: {features['EXT_SOURCE_MEAN']:.2f}")
                st.write(f"- Propriétaire: {'Oui' if features['FLAG_OWN_REALTY'] else 'Non'}")
                st.write(f"- Véhicule: {'Oui' if features['FLAG_OWN_CAR'] else 'Non'}")
            
        else:
            status_code = result.get("status_code") if result else None
            detail = result.get("detail", "Erreur inconnue") if result else "Pas de réponse"
            if isinstance(detail, dict):
                detail = detail.get("detail", detail)
            detail = str(detail)
            if status_code == 404:
                st.error("❌ Endpoint API introuvable (404).")
                st.info(f"API_URL: {API_URL}")
                st.info(f"Endpoint attendu: {API_URL}/predict")
            else:
                st.error(f"❌ Erreur API ({status_code}): {detail}")


def render_comparison_section(
    features: Dict[str, float],
    ref_data: pd.DataFrame,
    show_header: bool = True
):
    """Section comparaison avec la population."""
    if show_header:
        st.header("📊 Comparaison avec la population")
    if ref_data.empty:
        st.warning("⚠️ Données de référence non disponibles.")
        return

    st.markdown("""
    **Comparez les caractéristiques du client avec l'ensemble de la population ou un groupe de clients similaires.**
    """)

    group_filter = st.selectbox(
        "🎯 Groupe de comparaison",
        ["Tous les clients", "Clients sans défaut (TARGET=0)", "Clients en défaut (TARGET=1)"],
        help="Sélectionnez le groupe avec lequel comparer le client"
    )

    numeric_cols = ref_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in ref_data.columns if c not in numeric_cols]
    exclude_cols = ["SK_ID_CURR", "TARGET", "index"]
    numeric_features = [f for f in numeric_cols if f not in exclude_cols]
    categorical_features = [f for f in categorical_cols if f not in exclude_cols]
    available_features = numeric_features + categorical_features

    if not available_features:
        st.warning("Aucune feature comparable disponible.")
        return

    top_from_report = load_top_features_from_report(50)
    priority_features = [f for f in top_from_report if f in available_features]
    for feat in REQUIRED_FEATURES.keys():
        if feat in available_features and feat not in priority_features:
            priority_features.append(feat)
    other_features = [f for f in available_features if f not in priority_features]
    available_features = priority_features + sorted(other_features, key=get_feature_label)
    default_feature = priority_features[0] if priority_features else (available_features[0] if available_features else None)

    tab1, tab2, tab3 = st.tabs([
        "🎯 Vue Radar",
        "📈 Comparaison détaillée",
        "📋 Statistiques complètes"
    ])

    with tab1:
        st.subheader("🎯 Comparaison multi-critères")
        # Utiliser les features les plus importantes issues du notebook (reports/feature_importance.csv)
        top_from_report = load_top_features_from_report(15)
        default_radar = [f for f in top_from_report if f in numeric_features][:6]
        if len(default_radar) < 3:
            default_radar = [f for f in [
                "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1",
                "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY",
                "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO"
            ] if f in numeric_features][:6]

        radar_features = st.multiselect(
            "Caractéristiques à comparer (3-8 recommandé)",
            numeric_features,
            default=default_radar,
            help="Choisissez jusqu'à 8 caractéristiques pour le radar",
            format_func=get_feature_label
        )

        if radar_features and len(radar_features) >= 3:
            fig_radar = create_radar_comparison(features, ref_data, radar_features)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("Radar indisponible. Vérifiez que les valeurs client sont renseignées.")
        else:
            st.info("Sélectionnez au moins 3 caractéristiques.")

    with tab2:
        st.subheader("📈 Comparaison détaillée par caractéristique")
        selected_feature = st.selectbox(
            "Sélectionnez une caractéristique",
            available_features,
            help="Voir la distribution et la position du client",
            format_func=get_feature_label,
            index=available_features.index(default_feature) if default_feature in available_features else 0
        )

        st.info(f"**{get_feature_label(selected_feature)}**: {get_feature_explanation(selected_feature)}")
        client_value = features.get(selected_feature)
        if selected_feature in numeric_features:
            if client_value is None or pd.isna(client_value):
                st.warning("Valeur client indisponible pour cette caractéristique. Ajoutez-la dans les variables complémentaires.")
            else:
                fig = create_comparison_chart(client_value, selected_feature, ref_data, group_filter)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    ref_col = ref_data[selected_feature].dropna()
                    percentile = (ref_col < client_value).mean() * 100 if len(ref_col) > 0 else 0
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Valeur client", f"{client_value:,.2f}")
                    with stat_col2:
                        st.metric("Moyenne population", f"{ref_col.mean():,.2f}")
                    with stat_col3:
                        st.metric("Percentile", f"{percentile:.0f}%")

                    if percentile < 25:
                        st.warning("⚠️ Client dans les 25% les plus bas.")
                    elif percentile > 75:
                        st.success("✅ Client dans les 25% les plus hauts.")
                    else:
                        st.info("ℹ️ Client dans la moyenne.")
                else:
                    st.warning("Impossible de créer le graphique.")
        else:
            if client_value is None or (isinstance(client_value, float) and pd.isna(client_value)):
                st.warning("Valeur client indisponible pour cette caractéristique. Ajoutez-la dans les variables complémentaires.")
            else:
                series = ref_data[selected_feature].dropna().astype(str)
                if series.empty:
                    st.info("Aucune donnée disponible pour cette caractéristique.")
                else:
                    unique_count = series.nunique()
                    if unique_count > 15:
                        st.info("Trop de catégories pour un graphique lisible.")
                    else:
                        counts = series.value_counts()
                        labels = list(counts.index)
                        values = list(counts.values)
                        client_str = str(client_value)
                        colors = ["#C41E3A" if label == client_str else "#4169E1" for label in labels]
                        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
                        fig.update_layout(
                            title=f"Répartition de {get_feature_label(selected_feature)}",
                            xaxis_title="Catégorie",
                            yaxis_title="Nombre de clients",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        client_pct = (series == client_str).mean() * 100
                        st.info(f"Valeur client: **{client_str}** • Fréquence: **{client_pct:.1f}%**")

    with tab3:
        st.subheader("📋 Statistiques complètes du client")
        numeric_data = []
        for feat in numeric_features:
            if feat in ref_data.columns:
                client_val = features.get(feat)
                ref_col = ref_data[feat].dropna()
                if len(ref_col) == 0:
                    continue
                percentile = (ref_col < client_val).mean() * 100 if client_val is not None and not pd.isna(client_val) else None
                numeric_data.append({
                    "Caractéristique": get_feature_label(feat),
                    "Colonne": feat,
                    "Valeur client": f"{client_val:,.2f}" if client_val is not None and not pd.isna(client_val) else "Non renseigné",
                    "Moyenne pop.": f"{ref_col.mean():,.2f}",
                    "Médiane pop.": f"{ref_col.median():,.2f}",
                    "Percentile": f"{percentile:.0f}%" if percentile is not None else "—",
                })

        if numeric_data:
            st.markdown("**Variables numériques**")
            df_numeric = pd.DataFrame(numeric_data)
            st.dataframe(df_numeric, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune statistique numérique disponible.")

        cat_data = []
        for feat in categorical_features:
            if feat in ref_data.columns:
                series = ref_data[feat].dropna().astype(str)
                if series.empty:
                    continue
                mode_val = series.mode(dropna=True)
                mode_val = mode_val.iloc[0] if not mode_val.empty else "—"
                mode_pct = (series == str(mode_val)).mean() * 100 if mode_val != "—" else 0
                client_val = features.get(feat)
                client_str = str(client_val) if client_val is not None and not (isinstance(client_val, float) and pd.isna(client_val)) else "Non renseigné"
                client_pct = (series == client_str).mean() * 100 if client_str != "Non renseigné" else None
                cat_data.append({
                    "Caractéristique": get_feature_label(feat),
                    "Colonne": feat,
                    "Valeur client": client_str,
                    "Mode pop.": f"{mode_val}",
                    "Freq. mode": f"{mode_pct:.1f}%",
                    "Freq. client": f"{client_pct:.1f}%" if client_pct is not None else "—"
                })

        if cat_data:
            st.markdown("**Variables catégorielles**")
            df_cat = pd.DataFrame(cat_data)
            st.dataframe(df_cat, use_container_width=True, hide_index=True)


def render_drift_tab():
    """Onglet rapport de Data Drift (Evidently)."""
    st.header("📈 Analyse du Data Drift")

    st.markdown("""
    ## Rapport Evidently
    
    Le rapport de data drift permet de détecter les dérives entre:
    - **Données d'entraînement** (référence)
    - **Données de production** (nouvelles données)
    
    **Métriques surveillées**:
    - Distribution des features
    - Valeurs manquantes
    - Corrélations
    - Tests statistiques (Kolmogorov-Smirnov, Chi²)
    """)

    if os.path.exists(DRIFT_REPORT_PATH):
        try:
            with open(DRIFT_REPORT_PATH, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.markdown("### Rapport complet Evidently")
            st.components.v1.html(html_content, height=1200, scrolling=True)
        except Exception as e:
            st.error(f"Erreur lors du chargement du rapport: {e}")
    else:
        st.warning("📋 Rapport Evidently non disponible.")
        st.info(
            "Le rapport est généré par le notebook "
            f"[04_Drift_Evidently.ipynb]({github_blob('notebooks/04_Drift_Evidently.ipynb')})."
        )


def render_documentation_tab():
    """Onglet documentation."""
    st.header("📖 Documentation")
    
    st.markdown("""
    ## Guide d'utilisation
    
    ### 🎯 Onglet Scoring
    Saisissez les caractéristiques du client pour obtenir:
    - La probabilité de défaut de paiement
    - La décision (Accordé/Refusé)
    - Une interprétation pour non-experts
    
    ### 📊 Comparaison (dans l'onglet Scoring)
    Comparez le profil du client avec l'ensemble de la population:
    - Visualisation de la distribution
    - Position du client (percentile)

    ### 📊 Onglet Dataset
    Présente les statistiques clés du jeu de données (taille, taux de défaut, revenus, crédits, scores externes).
    
    ### 📈 Onglet Data Drift
    Rapport Evidently analysant la stabilité des données.
    
    ---
    
    ## Features importantes
    
    | Feature | Description | Impact |
    |---------|-------------|--------|
    | EXT_SOURCE_1/2/3 | Scores de crédit externes | Plus élevé = meilleur |
    | CREDIT_INCOME_RATIO | Ratio crédit/revenu | Plus bas = meilleur |
    | DAYS_BIRTH | Âge (jours négatifs) | Plus âgé = plus stable |
    | AMT_CREDIT | Montant du crédit | - |
    
    ---
    
    ## Seuil de décision
    
    **Seuil optimal: 0.44** (issu de l'optimisation du coût métier)
    
    - Coût Faux Négatif (FN): 10 (mauvais client accepté)
    - Coût Faux Positif (FP): 1 (bon client refusé)
    
    Le modèle minimise: `10 × FN + 1 × FP`
    """)


def render_dataset_tab():
    """Onglet analyse du dataset."""
    st.header("📊 Analyse du jeu de données")

    ref_data = load_reference_data()
    if ref_data.empty:
        st.warning("⚠️ Données de référence non disponibles.")
        return

    model_info = get_model_info()
    if model_info:
        st.markdown("### 🤖 Infos modèle")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Nom", model_info.get("model_name", "N/A"))
        col_m2.metric("Version", model_info.get("version", "N/A"))
        col_m3.metric("Date", model_info.get("training_date") or "N/A")
        col_m4, col_m5, col_m6 = st.columns(3)
        col_m4.metric("AUC", f"{model_info.get('auc', 0):.3f}" if model_info.get("auc") is not None else "N/A")
        col_m5.metric("Accuracy", f"{model_info.get('accuracy', 0):.3f}" if model_info.get("accuracy") is not None else "N/A")
        col_m6.metric("Coût métier", f"{model_info.get('business_cost', 0):,.0f}" if model_info.get("business_cost") is not None else "N/A")

    st.markdown("### Vue d'ensemble")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clients", f"{len(ref_data):,}")
    with col2:
        if "TARGET" in ref_data.columns:
            st.metric("Taux défaut", f"{ref_data['TARGET'].mean():.1%}")
        else:
            st.metric("Taux défaut", "N/A")
    with col3:
        st.metric("Colonnes", f"{ref_data.shape[1]:,}")

    st.markdown("### Structure & qualité")
    numeric_cols = ref_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in ref_data.columns if c not in numeric_cols]
    total_cells = ref_data.shape[0] * ref_data.shape[1]
    missing_cells = int(ref_data.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0
    col_q1, col_q2, col_q3 = st.columns(3)
    col_q1.metric("Variables numériques", f"{len(numeric_cols):,}")
    col_q2.metric("Variables catégorielles", f"{len(categorical_cols):,}")
    col_q3.metric("Valeurs manquantes", f"{missing_pct:.1f}%")

    if "TARGET" in ref_data.columns:
        st.markdown("### Cible (TARGET)")
        target_counts = ref_data["TARGET"].value_counts().sort_index()
        fig_target = go.Figure(
            data=[
                go.Bar(
                    x=["Non défaut (0)", "Défaut (1)"],
                    y=[target_counts.get(0, 0), target_counts.get(1, 0)],
                    marker_color=["#4169E1", "#C41E3A"]
                )
            ]
        )
        fig_target.update_layout(
            height=300,
            yaxis_title="Nombre de clients",
            showlegend=False
        )
        st.plotly_chart(fig_target, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Finances")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        if "AMT_INCOME_TOTAL" in ref_data.columns:
            st.metric("Revenu médian", f"{ref_data['AMT_INCOME_TOTAL'].median():,.0f}€")
        else:
            st.metric("Revenu médian", "N/A")
    with col_f2:
        if "AMT_CREDIT" in ref_data.columns:
            st.metric("Crédit médian", f"{ref_data['AMT_CREDIT'].median():,.0f}€")
        else:
            st.metric("Crédit médian", "N/A")
    with col_f3:
        if "AMT_ANNUITY" in ref_data.columns:
            st.metric("Annuité médiane", f"{ref_data['AMT_ANNUITY'].median():,.0f}€")
        else:
            st.metric("Annuité médiane", "N/A")

    st.markdown("---")
    st.markdown("### 📊 Scores externes")
    col_s1, col_s2, col_s3 = st.columns(3)
    if "EXT_SOURCE_1" in ref_data.columns:
        col_s1.metric("Score externe 1", f"{ref_data['EXT_SOURCE_1'].median():.3f}")
    else:
        col_s1.metric("Score externe 1", "N/A")
    if "EXT_SOURCE_2" in ref_data.columns:
        col_s2.metric("Score externe 2", f"{ref_data['EXT_SOURCE_2'].median():.3f}")
    else:
        col_s2.metric("Score externe 2", "N/A")
    if "EXT_SOURCE_3" in ref_data.columns:
        col_s3.metric("Score externe 3", f"{ref_data['EXT_SOURCE_3'].median():.3f}")
    else:
        col_s3.metric("Score externe 3", "N/A")

    st.markdown("---")
    st.markdown("### 📌 Variables importantes")
    top_features = load_top_features_from_report(15)
    if top_features:
        display_labels = [get_feature_label(f) for f in top_features]
        st.write(", ".join(display_labels))
    else:
        st.caption("Aucune importance de feature disponible.")

    st.markdown("---")
    render_ce_checklist()

    st.markdown("---")
    st.markdown("### ✅ Exigences (exigence.txt)")
    if os.path.exists(EXIGENCE_PATH):
        with st.expander("Voir toutes les exigences", expanded=False):
            try:
                with open(EXIGENCE_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                st.text(content)
            except Exception as e:
                st.error(f"Erreur de lecture: {e}")
    else:
        st.caption("Fichier exigence.txt non trouvé.")


# ============================================
# Point d'entrée principal
# ============================================

def main():
    """Fonction principale de l'application."""
    st.title("🏦 Home Credit - Outil de Scoring")
    st.markdown("""
    **Outil d'aide à la décision pour l'octroi de crédit**
    
    Cette application évalue le risque de défaut de paiement et fournit une interprétation
    claire du score pour chaque demande de crédit.
    """)

    reference_data = load_reference_data()
    render_sidebar(reference_data)

    current_page = st.session_state.get("current_page", "scoring")

    if current_page == "scoring":
        render_prediction_tab()
    elif current_page == "dataset":
        render_dataset_tab()
    elif current_page == "drift":
        render_drift_tab()
    elif current_page == "docs":
        render_documentation_tab()
    else:
        st.session_state.current_page = "scoring"
        render_prediction_tab()


if __name__ == "__main__":
    main()
