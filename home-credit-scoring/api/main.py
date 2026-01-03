"""
API FastAPI pour le scoring crédit Home Credit.
================================================

Cette API expose le modèle de scoring crédit pour:
- Prédiction individuelle
- Prédiction batch
- Explication des prédictions
- Informations sur le modèle

Déployable sur Render, Railway ou tout cloud provider.
"""

import os
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    ClientFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    FeatureContribution,
    FeatureImportance,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    RiskCategory,
    Decision
)

# Configuration
API_VERSION = "1.0.0"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Variables globales pour le modèle
model = None
preprocessor = None
config = None


def get_risk_category(probability: float) -> RiskCategory:
    """Détermine la catégorie de risque selon la probabilité."""
    if probability < 0.2:
        return RiskCategory.VERY_LOW
    elif probability < 0.4:
        return RiskCategory.LOW
    elif probability < 0.6:
        return RiskCategory.MODERATE
    elif probability < 0.8:
        return RiskCategory.HIGH
    else:
        return RiskCategory.VERY_HIGH


def load_model_artifacts():
    """Charge le modèle et le préprocesseur."""
    global model, preprocessor, config
    
    import joblib
    
    model_path = MODEL_DIR / "lgbm_model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    config_path = MODEL_DIR / "model_config.json"
    
    # Vérifier si les fichiers existent
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Préprocesseur non trouvé: {preprocessor_path}")
    
    # Charger le modèle
    model = joblib.load(model_path)
    print(f"✅ Modèle chargé: {model_path}")
    
    # Charger le préprocesseur
    preprocessor = joblib.load(preprocessor_path)
    print(f"✅ Préprocesseur chargé: {preprocessor_path}")
    
    # Charger la configuration
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration chargée: seuil={config.get('optimal_threshold', 0.5):.3f}")
    else:
        config = {"optimal_threshold": 0.5, "cost_fn": 10, "cost_fp": 1}
        print("⚠️ Configuration par défaut utilisée")
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # Startup
    try:
        load_model_artifacts()
        print("🚀 API démarrée avec succès!")
    except Exception as e:
        print(f"⚠️ Erreur au chargement: {e}")
        print("   L'API démarre mais les prédictions ne fonctionneront pas.")
    
    yield
    
    # Shutdown
    print("👋 API arrêtée")


# Créer l'application FastAPI
app = FastAPI(
    title="Home Credit Scoring API",
    description="""
    API de scoring crédit pour le projet Home Credit.
    
    ## Fonctionnalités
    
    * **Prédiction** : Obtenir la probabilité de défaut d'un client
    * **Batch** : Prédire pour plusieurs clients en une requête
    * **Explication** : Comprendre les facteurs de risque
    * **Info** : Informations sur le modèle déployé
    
    ## Coût métier
    
    Le modèle est optimisé pour minimiser le coût métier:
    - Faux Négatif (défaut non détecté) : coût = 10
    - Faux Positif (bon client refusé) : coût = 1
    
    ## Seuil de décision
    
    Le seuil optimal est déterminé lors de l'entraînement pour minimiser le coût total.
    """,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, restreindre aux origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée de l'API."""
    return {
        "message": "Home Credit Scoring API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérification de l'état de l'API."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version=API_VERSION
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Obtenir les informations sur le modèle déployé."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return ModelInfo(
        model_name=config.get("model_name", "home_credit_model"),
        version=API_VERSION,
        optimal_threshold=config.get("optimal_threshold", 0.5),
        cost_fn=config.get("cost_fn", 10),
        cost_fp=config.get("cost_fp", 1),
        n_features=len(preprocessor.feature_names) if preprocessor else 0,
        training_date=config.get("training_date")
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    client: ClientFeatures,
    threshold: Optional[float] = Query(None, ge=0, le=1, description="Seuil personnalisé")
):
    """
    Prédit la probabilité de défaut pour un client.
    
    - **client**: Données du client
    - **threshold**: Seuil de décision personnalisé (optionnel, utilise l'optimal par défaut)
    
    Retourne la probabilité, la décision et la catégorie de risque.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Convertir en DataFrame
        client_dict = client.model_dump(exclude_none=True)
        df = pd.DataFrame([client_dict])
        
        # Prétraitement
        X = preprocessor.transform(df)
        
        # Prédiction
        probability = float(model.predict_proba(X)[0, 1])
        
        # Seuil
        used_threshold = threshold if threshold is not None else config.get("optimal_threshold", 0.5)
        prediction = 1 if probability >= used_threshold else 0
        
        return PredictionResponse(
            client_id=client.SK_ID_CURR,
            probability=round(probability, 4),
            prediction=prediction,
            decision=Decision.REFUSED if prediction == 1 else Decision.ACCEPTED,
            risk_category=get_risk_category(probability),
            threshold=used_threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Prédit la probabilité de défaut pour plusieurs clients.
    
    - **clients**: Liste des données clients
    - **threshold**: Seuil personnalisé (optionnel)
    
    Retourne les prédictions pour chaque client et un résumé.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    if len(request.clients) == 0:
        raise HTTPException(status_code=400, detail="Liste de clients vide")
    
    if len(request.clients) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 clients par requête")
    
    try:
        # Convertir en DataFrame
        clients_data = [c.model_dump(exclude_none=True) for c in request.clients]
        df = pd.DataFrame(clients_data)
        
        # Prétraitement
        X = preprocessor.transform(df)
        
        # Prédictions
        probabilities = model.predict_proba(X)[:, 1]
        
        # Seuil
        used_threshold = request.threshold if request.threshold is not None else config.get("optimal_threshold", 0.5)
        
        # Construire les réponses
        predictions = []
        for i, (client, proba) in enumerate(zip(request.clients, probabilities)):
            prediction = 1 if proba >= used_threshold else 0
            predictions.append(PredictionResponse(
                client_id=client.SK_ID_CURR,
                probability=round(float(proba), 4),
                prediction=prediction,
                decision=Decision.REFUSED if prediction == 1 else Decision.ACCEPTED,
                risk_category=get_risk_category(proba),
                threshold=used_threshold
            ))
        
        accepted = sum(1 for p in predictions if p.prediction == 0)
        refused = len(predictions) - accepted
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_clients=len(predictions),
            accepted_count=accepted,
            refused_count=refused
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction batch: {str(e)}")


@app.post("/predict/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_prediction(client: ClientFeatures):
    """
    Explique la prédiction pour un client avec les features les plus influentes.
    
    Utilise l'importance des features du modèle pour identifier
    les facteurs qui contribuent le plus à la décision.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Convertir en DataFrame
        client_dict = client.model_dump(exclude_none=True)
        df = pd.DataFrame([client_dict])
        
        # Prétraitement
        X = preprocessor.transform(df)
        
        # Prédiction
        probability = float(model.predict_proba(X)[0, 1])
        threshold = config.get("optimal_threshold", 0.5)
        prediction = 1 if probability >= threshold else 0
        
        # Feature importance (globale pour simplifier)
        feature_importances = model.feature_importances_
        feature_names = preprocessor.feature_names
        
        # Top features par importance
        sorted_indices = np.argsort(feature_importances)[::-1][:10]
        
        top_features = []
        for idx in sorted_indices:
            feature_name = feature_names[idx]
            feature_value = float(X[0, idx]) if not np.isnan(X[0, idx]) else 0.0
            importance = float(feature_importances[idx])
            
            # Déterminer la direction basée sur la valeur et l'importance
            direction = "augmente le risque" if feature_value > 0 else "diminue le risque"
            
            top_features.append(FeatureContribution(
                feature=feature_name,
                value=round(feature_value, 4),
                contribution=round(importance, 4),
                direction=direction
            ))
        
        return ExplanationResponse(
            client_id=client.SK_ID_CURR,
            probability=round(probability, 4),
            prediction=prediction,
            decision=Decision.REFUSED if prediction == 1 else Decision.ACCEPTED,
            top_features=top_features
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'explication: {str(e)}")


@app.get("/model/features", response_model=List[FeatureImportance], tags=["Model"])
async def get_feature_importance(top_n: int = Query(20, ge=1, le=100)):
    """
    Obtenir l'importance des features du modèle.
    
    - **top_n**: Nombre de features à retourner (1-100)
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        feature_importances = model.feature_importances_
        feature_names = preprocessor.feature_names
        
        # Trier par importance
        sorted_indices = np.argsort(feature_importances)[::-1][:top_n]
        
        result = []
        for rank, idx in enumerate(sorted_indices, 1):
            result.append(FeatureImportance(
                feature=feature_names[idx],
                importance=round(float(feature_importances[idx]), 4),
                rank=rank
            ))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/client/{client_id}", response_model=PredictionResponse, tags=["Client"])
async def get_client_prediction(
    client_id: int,
    threshold: Optional[float] = Query(None, ge=0, le=1)
):
    """
    Obtenir la prédiction pour un client par son ID.
    
    Note: Cette endpoint nécessite que les données du client soient disponibles
    dans la base de données ou le fichier de données.
    """
    # Cette implémentation est un placeholder
    # En production, vous chargeriez les données du client depuis une base de données
    raise HTTPException(
        status_code=501, 
        detail="Endpoint non implémenté. Utilisez /predict avec les données du client."
    )


# Gestion des erreurs
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erreur interne: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
