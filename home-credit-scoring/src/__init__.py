"""
Module src - Home Credit Scoring
================================

Ce module contient toutes les fonctionnalités pour le pipeline MLOps:
- preprocessing: Pipeline de prétraitement des données
- train: Entraînement avec tracking MLflow
- metrics: Métriques métier et optimisation du seuil
- inference: Chargement et prédiction du modèle
- feature_importance: Importance des features (globale et locale SHAP)
"""

from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR", 
    "MODELS_DIR",
    "REPORTS_DIR"
]
