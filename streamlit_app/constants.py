"""
Constantes et données de référence pour le Dashboard.
=====================================================

Contient les features requises, leurs explications, et valeurs par défaut.
Les valeurs par défaut sont issues de l'analyse des données d'entraînement.
"""

# ============================================
# 17 Features requises pour l'inférence
# ============================================
# Ces features sont transformées en 245 features engineered par le pipeline

REQUIRED_FEATURES = {
    # Finances
    "AMT_INCOME_TOTAL": {
        "label": "Revenu annuel total",
        "type": "float",
        "min": 0,
        "max": 10000000,
        "default": 150000.0,
        "unit": "€",
        "description": "Revenu total annuel du client en euros"
    },
    "AMT_CREDIT": {
        "label": "Montant du crédit",
        "type": "float",
        "min": 0,
        "max": 5000000,
        "default": 500000.0,
        "unit": "€",
        "description": "Montant total du crédit demandé"
    },
    "AMT_ANNUITY": {
        "label": "Annuité du crédit",
        "type": "float",
        "min": 0,
        "max": 500000,
        "default": 25000.0,
        "unit": "€/an",
        "description": "Montant de l'annuité à rembourser"
    },
    "AMT_GOODS_PRICE": {
        "label": "Prix du bien",
        "type": "float",
        "min": 0,
        "max": 5000000,
        "default": 450000.0,
        "unit": "€",
        "description": "Prix du bien financé par le crédit"
    },
    
    # Temporel (en jours négatifs depuis aujourd'hui)
    "DAYS_BIRTH": {
        "label": "Âge (en jours)",
        "type": "int",
        "min": -30000,
        "max": -6000,
        "default": -12775,  # ~35 ans
        "unit": "jours",
        "description": "Âge du client en jours (négatif). Ex: -12775 ≈ 35 ans"
    },
    "DAYS_EMPLOYED": {
        "label": "Ancienneté emploi (jours)",
        "type": "int",
        "min": -20000,
        "max": 0,
        "default": -1825,  # ~5 ans
        "unit": "jours",
        "description": "Ancienneté dans l'emploi actuel en jours (négatif). Ex: -1825 ≈ 5 ans"
    },
    
    # Personnel
    "CNT_CHILDREN": {
        "label": "Nombre d'enfants",
        "type": "int",
        "min": 0,
        "max": 20,
        "default": 1,
        "unit": "",
        "description": "Nombre d'enfants à charge"
    },
    "CODE_GENDER_M": {
        "label": "Genre (Homme)",
        "type": "int",
        "min": 0,
        "max": 1,
        "default": 1,
        "unit": "",
        "description": "1 = Homme, 0 = Femme"
    },
    "FLAG_OWN_CAR": {
        "label": "Possède une voiture",
        "type": "int",
        "min": 0,
        "max": 1,
        "default": 1,
        "unit": "",
        "description": "1 = Oui, 0 = Non"
    },
    "FLAG_OWN_REALTY": {
        "label": "Propriétaire immobilier",
        "type": "int",
        "min": 0,
        "max": 1,
        "default": 1,
        "unit": "",
        "description": "1 = Propriétaire, 0 = Locataire"
    },
    
    # Scores externes (sources de crédit externes)
    "EXT_SOURCE_1": {
        "label": "Score externe 1",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "unit": "",
        "description": "Score de crédit provenant d'une source externe 1 (0-1, plus élevé = meilleur)"
    },
    "EXT_SOURCE_2": {
        "label": "Score externe 2",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.6,
        "unit": "",
        "description": "Score de crédit provenant d'une source externe 2 (0-1, plus élevé = meilleur)"
    },
    "EXT_SOURCE_3": {
        "label": "Score externe 3",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.55,
        "unit": "",
        "description": "Score de crédit provenant d'une source externe 3 (0-1, plus élevé = meilleur)"
    },
    
    # Région
    "REGION_RATING_CLIENT": {
        "label": "Note de la région",
        "type": "int",
        "min": 1,
        "max": 3,
        "default": 2,
        "unit": "",
        "description": "Notation de la région du client (1=meilleur, 3=moins bon)"
    },
    
    # Ratios calculés (peuvent être calculés automatiquement si manquants)
    "CREDIT_INCOME_RATIO": {
        "label": "Ratio crédit/revenu",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 3.33,
        "unit": "",
        "description": "Ratio entre le montant du crédit et le revenu annuel"
    },
    "ANNUITY_INCOME_RATIO": {
        "label": "Ratio annuité/revenu",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.17,
        "unit": "",
        "description": "Ratio entre l'annuité et le revenu annuel"
    },
    "EXT_SOURCE_MEAN": {
        "label": "Moyenne scores externes",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.55,
        "unit": "",
        "description": "Moyenne des 3 scores externes"
    },
}


# ============================================
# Explications détaillées des features
# ============================================
FEATURE_EXPLANATIONS = {
    # Features principales
    "EXT_SOURCE_1": "Score de crédit externe #1 (agence de notation). Plus élevé = meilleur historique crédit.",
    "EXT_SOURCE_2": "Score de crédit externe #2 (autre agence). Plus élevé = meilleur profil financier.",
    "EXT_SOURCE_3": "Score de crédit externe #3 (source tierce). Plus élevé = risque plus faible.",
    "EXT_SOURCE_MEAN": "Moyenne des 3 scores externes. Indicateur synthétique de solvabilité.",
    
    "AMT_INCOME_TOTAL": "Revenu annuel total du client en euros.",
    "AMT_CREDIT": "Montant total du crédit demandé en euros.",
    "AMT_ANNUITY": "Montant de l'annuité (remboursement annuel) en euros.",
    "AMT_GOODS_PRICE": "Prix du bien financé par le crédit en euros.",
    
    "DAYS_BIRTH": "Âge du client en jours (valeur négative). -12775 ≈ 35 ans.",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi actuel en jours (négatif). -1825 ≈ 5 ans.",
    
    "CNT_CHILDREN": "Nombre d'enfants à charge du client.",
    "CODE_GENDER_M": "Genre du client (1=Homme, 0=Femme).",
    "FLAG_OWN_CAR": "Le client possède-t-il une voiture (1=Oui, 0=Non).",
    "FLAG_OWN_REALTY": "Le client est-il propriétaire immobilier (1=Oui, 0=Non).",
    
    "REGION_RATING_CLIENT": "Note de risque de la région (1=faible risque, 3=risque élevé).",
    
    "CREDIT_INCOME_RATIO": "Ratio crédit demandé / revenu annuel. Plus bas = meilleure capacité.",
    "ANNUITY_INCOME_RATIO": "Ratio annuité / revenu annuel. Plus bas = charge plus soutenable.",
}


# ============================================
# Valeurs par défaut d'un client type
# ============================================
def get_default_features() -> dict:
    """Retourne un dictionnaire avec les valeurs par défaut de toutes les features."""
    return {key: config["default"] for key, config in REQUIRED_FEATURES.items()}


def calculate_ratios(features: dict) -> dict:
    """
    Calcule automatiquement les ratios si non fournis.
    
    Args:
        features: Dictionnaire des features du client
    
    Returns:
        Dictionnaire avec les ratios calculés
    """
    result = features.copy()
    
    # CREDIT_INCOME_RATIO
    if "CREDIT_INCOME_RATIO" not in result or result.get("CREDIT_INCOME_RATIO") is None:
        income = result.get("AMT_INCOME_TOTAL", 1)
        credit = result.get("AMT_CREDIT", 0)
        result["CREDIT_INCOME_RATIO"] = credit / income if income > 0 else 0
    
    # ANNUITY_INCOME_RATIO
    if "ANNUITY_INCOME_RATIO" not in result or result.get("ANNUITY_INCOME_RATIO") is None:
        income = result.get("AMT_INCOME_TOTAL", 1)
        annuity = result.get("AMT_ANNUITY", 0)
        result["ANNUITY_INCOME_RATIO"] = annuity / income if income > 0 else 0
    
    # EXT_SOURCE_MEAN
    if "EXT_SOURCE_MEAN" not in result or result.get("EXT_SOURCE_MEAN") is None:
        sources = [
            result.get("EXT_SOURCE_1", 0.5),
            result.get("EXT_SOURCE_2", 0.5),
            result.get("EXT_SOURCE_3", 0.5)
        ]
        # Filtrer les None
        valid_sources = [s for s in sources if s is not None]
        result["EXT_SOURCE_MEAN"] = sum(valid_sources) / len(valid_sources) if valid_sources else 0.5
    
    return result


# ============================================
# Configuration du modèle (depuis model_config.json)
# ============================================
MODEL_CONFIG = {
    "optimal_threshold": 0.44,  # Seuil optimal issu de l'entraînement
    "cost_fn": 10,              # Coût d'un Faux Négatif
    "cost_fp": 1,               # Coût d'un Faux Positif
    "model_name": "home_credit_model",
    "auc": 0.768,
}


# ============================================
# Catégories de risque
# ============================================
def get_risk_category(probability: float) -> str:
    """Retourne la catégorie de risque basée sur la probabilité."""
    if probability < 0.2:
        return "Très faible"
    elif probability < 0.4:
        return "Faible"
    elif probability < 0.6:
        return "Modéré"
    elif probability < 0.8:
        return "Élevé"
    else:
        return "Très élevé"


def get_risk_color(probability: float) -> str:
    """Retourne la couleur associée au niveau de risque."""
    if probability < 0.4:
        return "#28a745"  # Vert
    elif probability < 0.6:
        return "#ffc107"  # Jaune/Orange
    else:
        return "#dc3545"  # Rouge
