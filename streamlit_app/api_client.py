"""
Client API pour le Dashboard Streamlit.
=======================================

Module dédié aux appels HTTP vers l'API FastAPI de scoring.
Utilise requests avec des appels JSON simples (sans Pydantic côté client).
"""

import requests
import os
from typing import Dict, Any, Optional

# Configuration API depuis variables d'environnement
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Timeout configurations (Render free tier peut être lent au démarrage)
TIMEOUT_HEALTH = 15
TIMEOUT_INFO = 30
TIMEOUT_PREDICT = 180  # 3 min pour cold start + calcul


def check_api_health() -> bool:
    """
    Vérifie si l'API est accessible.
    
    Returns:
        bool: True si l'API répond correctement, False sinon
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT_HEALTH)
        return response.status_code == 200
    except Exception:
        return False


def get_model_info() -> Optional[Dict[str, Any]]:
    """
    Récupère les informations du modèle déployé.
    
    Returns:
        Dict avec model_name, version, optimal_threshold, etc.
        None si erreur
    """
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=TIMEOUT_INFO)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict_client(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Effectue une prédiction pour un client.
    
    Utilise l'endpoint /predict avec le format attendu par l'API.
    
    Args:
        features: Dictionnaire des caractéristiques du client
            Clés requises (17 features):
            - AMT_INCOME_TOTAL: float (revenu annuel)
            - AMT_CREDIT: float (montant du crédit demandé)
            - AMT_ANNUITY: float (montant de l'annuité)
            - AMT_GOODS_PRICE: float (prix du bien)
            - DAYS_BIRTH: int (âge en jours, négatif, ex: -12000)
            - DAYS_EMPLOYED: int (ancienneté emploi en jours, négatif)
            - CNT_CHILDREN: int (nombre d'enfants)
            - CODE_GENDER_M: int (1=homme, 0=femme)
            - FLAG_OWN_CAR: int (1=oui, 0=non)
            - FLAG_OWN_REALTY: int (1=oui, 0=non)
            - EXT_SOURCE_1: float (score externe 1, 0-1)
            - EXT_SOURCE_2: float (score externe 2, 0-1)
            - EXT_SOURCE_3: float (score externe 3, 0-1)
            - REGION_RATING_CLIENT: int (note région, 1-3)
            - CREDIT_INCOME_RATIO: float (ratio crédit/revenu)
            - ANNUITY_INCOME_RATIO: float (ratio annuité/revenu)
            - EXT_SOURCE_MEAN: float (moyenne des scores externes)
    
    Returns:
        Dict avec:
            - probability: float (probabilité de défaut 0-1)
            - decision: str ("ACCEPTÉ" ou "REFUSÉ")
            - risk_category: str (niveau de risque)
            - threshold: float (seuil utilisé)
        Ou dict avec "error" si échec
    """
    try:
        endpoint = "/predict"
        url = f"{API_URL}{endpoint}"
        response = requests.post(
            url,
            json={"features": features},
            timeout=TIMEOUT_PREDICT,
            headers={"Content-Type": "application/json"}
        )

        # Fallback simple si l'API attend encore "data" (compatibilité)
        if response.status_code == 422:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            if "data" in str(detail).lower():
                response = requests.post(
                    url,
                    json={"data": features},
                    timeout=TIMEOUT_PREDICT,
                    headers={"Content-Type": "application/json"}
                )

        if response.status_code == 200:
            return response.json()

        # Extraire le détail de l'erreur
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        return {
            "error": True,
            "status_code": response.status_code,
            "detail": error_detail,
            "endpoint": endpoint
        }
            
    except requests.exceptions.ConnectionError:
        return {"error": True, "detail": "API non accessible"}
    except requests.exceptions.Timeout:
        return {"error": True, "detail": "Timeout de l'API"}
    except Exception as e:
        return {"error": True, "detail": str(e)}


def explain_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Obtient l'explication de la prédiction avec les features importantes.
    
    Args:
        features: Dictionnaire des caractéristiques du client (même format que predict)
    
    Returns:
        Dict avec probability, decision, top_features (liste des contributions)
        Ou dict avec "error" si échec
    """
    try:
        url = f"{API_URL}/predict/explain"
        response = requests.post(
            url,
            json={"features": features},
            timeout=TIMEOUT_PREDICT,
            headers={"Content-Type": "application/json"}
        )

        # Fallback simple si l'API attend encore "data" (compatibilité)
        if response.status_code == 422:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            if "data" in str(detail).lower():
                response = requests.post(
                    url,
                    json={"data": features},
                    timeout=TIMEOUT_PREDICT,
                    headers={"Content-Type": "application/json"}
                )

        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text
            return {
                "error": True,
                "status_code": response.status_code,
                "detail": error_detail
            }
            
    except requests.exceptions.ConnectionError:
        return {"error": True, "detail": "API non accessible pour les explications"}
    except requests.exceptions.Timeout:
        return {"error": True, "detail": "Timeout de l'API explications"}
    except Exception as e:
        return {"error": True, "detail": str(e)}


def get_feature_importance(top_n: int = 20) -> Optional[list]:
    """
    Récupère l'importance globale des features du modèle.
    
    Args:
        top_n: Nombre de features à retourner (défaut: 20)
    
    Returns:
        Liste de dict avec feature, importance, rank
        None si erreur
    """
    try:
        response = requests.get(
            f"{API_URL}/model/features",
            params={"top_n": top_n},
            timeout=TIMEOUT_INFO
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None
