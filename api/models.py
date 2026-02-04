"""
Schémas Pydantic pour l'API de scoring crédit.
==============================================

Définit les modèles de données pour les requêtes et réponses de l'API.
Compatible Pydantic v2 et FastAPI.
"""

from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional, List, Dict, Any
from enum import Enum


class RiskCategory(str, Enum):
    """Catégories de risque."""
    VERY_LOW = "Très faible"
    LOW = "Faible"
    MODERATE = "Modéré"
    HIGH = "Élevé"
    VERY_HIGH = "Très élevé"


class Decision(str, Enum):
    """Décision de crédit."""
    ACCEPTED = "ACCEPTÉ"
    REFUSED = "REFUSÉ"


class PredictionRequest(BaseModel):
    """Requête de prédiction simple avec features en dictionnaire.

    Format attendu (unique) :
    {
      "features": { ... }
    }
    Compatibilité: "data" est accepté comme alias.
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Features du client (format unique attendu)",
        validation_alias=AliasChoices("features", "data")
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "features": {
                        "AMT_INCOME_TOTAL": 150000.0,
                        "AMT_CREDIT": 500000.0,
                        "AMT_ANNUITY": 25000.0,
                        "AMT_GOODS_PRICE": 450000.0,
                        "DAYS_BIRTH": -12775,
                        "DAYS_EMPLOYED": -1825,
                        "CNT_CHILDREN": 1,
                        "CODE_GENDER_M": 1,
                        "FLAG_OWN_CAR": 1,
                        "FLAG_OWN_REALTY": 1,
                        "EXT_SOURCE_1": 0.5,
                        "EXT_SOURCE_2": 0.6,
                        "EXT_SOURCE_3": 0.55,
                        "REGION_RATING_CLIENT": 2,
                        "CREDIT_INCOME_RATIO": 3.33,
                        "ANNUITY_INCOME_RATIO": 0.17,
                        "EXT_SOURCE_MEAN": 0.55
                    }
                }
            ]
        }
    )

    def get_features_dict(self) -> Dict[str, Any]:
        """Retourne les features telles qu'envoyées."""
        return self.features


class ClientFeatures(BaseModel):
    """
    Features d'un client pour la prédiction.
    
    Contient les principales features utilisées par le modèle.
    Toutes les features sont optionnelles car le modèle gère les valeurs manquantes.
    """
    # Identifiant
    SK_ID_CURR: Optional[int] = Field(None, description="ID du client")
    
    # Type de contrat
    NAME_CONTRACT_TYPE: Optional[str] = Field(None, description="Type de contrat (Cash/Revolving)")
    
    # Informations personnelles
    CODE_GENDER: Optional[str] = Field(None, description="Genre (M/F)")
    FLAG_OWN_CAR: Optional[str] = Field(None, description="Possède une voiture (Y/N)")
    FLAG_OWN_REALTY: Optional[str] = Field(None, description="Possède un bien immobilier (Y/N)")
    CNT_CHILDREN: Optional[int] = Field(None, ge=0, description="Nombre d'enfants")
    
    # Revenus et crédit
    AMT_INCOME_TOTAL: Optional[float] = Field(None, ge=0, description="Revenu total")
    AMT_CREDIT: Optional[float] = Field(None, ge=0, description="Montant du crédit")
    AMT_ANNUITY: Optional[float] = Field(None, ge=0, description="Annuité")
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0, description="Prix des biens")
    
    # Informations professionnelles
    NAME_INCOME_TYPE: Optional[str] = Field(None, description="Type de revenu")
    NAME_EDUCATION_TYPE: Optional[str] = Field(None, description="Niveau d'éducation")
    NAME_FAMILY_STATUS: Optional[str] = Field(None, description="Statut familial")
    NAME_HOUSING_TYPE: Optional[str] = Field(None, description="Type de logement")
    OCCUPATION_TYPE: Optional[str] = Field(None, description="Type d'occupation")
    ORGANIZATION_TYPE: Optional[str] = Field(None, description="Type d'organisation")
    
    # Variables temporelles (en jours, valeurs négatives)
    DAYS_BIRTH: Optional[int] = Field(None, le=0, description="Âge en jours (négatif)")
    DAYS_EMPLOYED: Optional[int] = Field(None, description="Jours d'emploi (négatif)")
    DAYS_REGISTRATION: Optional[float] = Field(None, description="Jours depuis inscription")
    DAYS_ID_PUBLISH: Optional[int] = Field(None, description="Jours depuis publication ID")
    
    # Scores externes (très prédictifs)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1, description="Score externe 1 (0-1)")
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1, description="Score externe 2 (0-1)")
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1, description="Score externe 3 (0-1)")
    
    # Informations régionales
    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, description="Population relative région")
    REGION_RATING_CLIENT: Optional[int] = Field(None, ge=1, le=3, description="Note région (1-3)")
    
    # Famille
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=1, description="Nombre membres famille")
    
    # Autres features (le modèle peut gérer des features additionnelles)
    # Ajoutez d'autres features si nécessaire
    
    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100001,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 135000.0,
                "AMT_CREDIT": 568800.0,
                "AMT_ANNUITY": 20560.5,
                "AMT_GOODS_PRICE": 450000.0,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -19241,
                "DAYS_EMPLOYED": -2329,
                "EXT_SOURCE_1": 0.7526,
                "EXT_SOURCE_2": 0.7897,
                "EXT_SOURCE_3": 0.1595
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction pour un client."""
    client_id: Optional[int] = Field(None, description="ID du client")
    probability: float = Field(..., ge=0, le=1, description="Probabilité de défaut")
    prediction: int = Field(..., ge=0, le=1, description="Prédiction binaire (0/1)")
    decision: Decision = Field(..., description="Décision (ACCEPTÉ/REFUSÉ)")
    risk_category: RiskCategory = Field(..., description="Catégorie de risque")
    threshold: float = Field(..., description="Seuil de décision utilisé")
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_id": 100001,
                "probability": 0.15,
                "prediction": 0,
                "decision": "ACCEPTÉ",
                "risk_category": "Faible",
                "threshold": 0.35
            }
        }


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction pour plusieurs clients."""
    clients: List[ClientFeatures] = Field(..., description="Liste des clients")
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Seuil personnalisé")


class BatchPredictionResponse(BaseModel):
    """Réponse de prédiction pour plusieurs clients."""
    predictions: List[PredictionResponse] = Field(..., description="Liste des prédictions")
    total_clients: int = Field(..., description="Nombre total de clients")
    accepted_count: int = Field(..., description="Nombre de clients acceptés")
    refused_count: int = Field(..., description="Nombre de clients refusés")


class FeatureContribution(BaseModel):
    """Contribution d'une feature à la prédiction."""
    feature: str = Field(..., description="Nom de la feature")
    value: float = Field(..., description="Valeur de la feature")
    contribution: float = Field(..., description="Contribution à la prédiction")
    direction: str = Field(..., description="Direction de l'effet")


class ExplanationResponse(BaseModel):
    """Explication d'une prédiction."""
    client_id: Optional[int] = Field(None, description="ID du client")
    probability: float = Field(..., description="Probabilité de défaut")
    prediction: int = Field(..., description="Prédiction binaire")
    decision: Decision = Field(..., description="Décision")
    top_features: List[FeatureContribution] = Field(..., description="Features les plus influentes")


class FeatureImportance(BaseModel):
    """Importance d'une feature."""
    feature: str = Field(..., description="Nom de la feature")
    importance: float = Field(..., description="Score d'importance")
    rank: int = Field(..., description="Rang de la feature")


class ModelInfo(BaseModel):
    """Informations sur le modèle."""
    model_name: str = Field(..., description="Nom du modèle")
    version: str = Field(..., description="Version")
    optimal_threshold: float = Field(..., description="Seuil optimal")
    cost_fn: int = Field(..., description="Coût FN")
    cost_fp: int = Field(..., description="Coût FP")
    n_features: int = Field(..., description="Nombre de features")
    training_date: Optional[str] = Field(None, description="Date d'entraînement")


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str = Field(..., description="Statut de l'API")
    model_loaded: bool = Field(..., description="Modèle chargé")
    version: str = Field(..., description="Version de l'API")


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    detail: str = Field(..., description="Message d'erreur")
    error_code: Optional[str] = Field(None, description="Code d'erreur")
