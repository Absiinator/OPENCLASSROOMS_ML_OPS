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

from fastapi import FastAPI, HTTPException, Query, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
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
DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# Variables globales pour le modèle
model = None
preprocessor = None
config = None


def get_model():
    """Dependency provider returning the current model (can be overridden in tests)."""
    return model


def get_preprocessor():
    """Dependency provider returning the current preprocessor (can be overridden in tests)."""
    return preprocessor


def get_explainer():
    """Dependency provider for explainer (placeholder, override in tests)."""
    return None


def get_model_info():
    """Dependency provider returning basic model metadata (overrideable in tests)."""
    return {
        'model_name': config.get('model_name', 'home_credit_model') if config else 'home_credit_model',
        'model_version': API_VERSION,
        'threshold': config.get('optimal_threshold', 0.5) if config else 0.5,
        'features': getattr(preprocessor, 'feature_names', []) if preprocessor else []
    }


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
async def health_check(model_dep = Depends(get_model)):
    """Vérification de l'état de l'API."""
    used_model = model_dep or model
    return HealthResponse(
        status="healthy" if used_model is not None else "degraded",
        model_loaded=used_model is not None,
        version=API_VERSION
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info_endpoint(info: dict = Depends(get_model_info)):
    """Obtenir les informations sur le modèle déployé (dépendance overrideable)."""
    return ModelInfo(
        model_name=info.get('model_name', 'home_credit_model'),
        version=info.get('model_version', API_VERSION),
        optimal_threshold=info.get('threshold', 0.5),
        cost_fn=config.get("cost_fn", 10) if config else 10,
        cost_fp=config.get("cost_fp", 1) if config else 1,
        n_features=len(info.get('features', [])),
        training_date=info.get('training_date')
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    payload: dict = Body(...),
    threshold: Optional[float] = Query(None, ge=0, le=1, description="Seuil personnalisé"),
    model_dep = Depends(get_model),
    preprocessor_dep = Depends(get_preprocessor)
):
    """
    Prédit la probabilité de défaut pour un client.
    
    - **client**: Données du client
    - **threshold**: Seuil de décision personnalisé (optionnel, utilise l'optimal par défaut)
    
    Retourne la probabilité, la décision et la catégorie de risque.
    """
    used_model = model_dep or model
    used_preprocessor = preprocessor_dep or preprocessor

    if used_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Accept both direct client fields or wrapper {'features': {...}}
        if isinstance(payload, dict) and 'features' in payload:
            client_dict = payload['features']
        else:
            client_dict = payload

        # Validate payload: if not wrapped and keys are not recognised features, return 400
        accepted_feature_names = []
        if used_preprocessor is not None:
            accepted_feature_names = getattr(used_preprocessor, 'feature_names', []) or []
        elif hasattr(used_model, 'feature_names_'):
            accepted_feature_names = getattr(used_model, 'feature_names_', []) or []

        if not (isinstance(payload, dict) and 'features' in payload):
            payload_keys = set(client_dict.keys()) if isinstance(client_dict, dict) else set()
            if not payload_keys:
                raise HTTPException(status_code=400, detail="Payload invalide: features manquantes")
            if accepted_feature_names:
                # if no intersection between provided keys and accepted feature names, consider invalid
                if payload_keys.isdisjoint(set(accepted_feature_names)):
                    raise HTTPException(status_code=400, detail="Payload invalide: features non reconnues")

        df = pd.DataFrame([client_dict])
        
        # Prétraitement
        if used_preprocessor is not None:
            X = used_preprocessor.transform(df)
        else:
            # fallback: pass raw values
            X = df.values

        # Prédiction
        probability = float(used_model.predict_proba(X)[0, 1])
        
        # Seuil
        if threshold is not None:
            used_threshold = threshold
        elif config is not None:
            used_threshold = config.get("optimal_threshold", 0.5)
        else:
            used_threshold = 0.5
            
        prediction = 1 if probability >= used_threshold else 0
        
        return PredictionResponse(
            client_id=client_dict.get('SK_ID_CURR') if isinstance(client_dict, dict) else None,
            probability=round(probability, 4),
            prediction=prediction,
            decision=Decision.REFUSED if prediction == 1 else Decision.ACCEPTED,
            risk_category=get_risk_category(probability),
            threshold=used_threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: dict = Body(...), model_dep = Depends(get_model), preprocessor_dep = Depends(get_preprocessor)):
    """
    Prédit la probabilité de défaut pour plusieurs clients.
    
    - **clients**: Liste des données clients
    - **threshold**: Seuil personnalisé (optionnel)
    
    Retourne les prédictions pour chaque client et un résumé.
    """
    used_model = model_dep or model
    used_preprocessor = preprocessor_dep or preprocessor

    if used_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    clients_list = []
    if isinstance(request, dict) and 'clients' in request:
        clients_list = request['clients']
    else:
        # allow direct BatchPredictionRequest-like dict
        clients_list = request.get('clients', []) if hasattr(request, 'get') else []

    if len(clients_list) == 0:
        raise HTTPException(status_code=400, detail="Liste de clients vide")
    if len(clients_list) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 clients par requête")
    
    try:
        # Convertir en DataFrame
        clients_data = []
        for c in clients_list:
            if isinstance(c, dict) and 'features' in c:
                clients_data.append(c['features'])
            elif isinstance(c, dict):
                clients_data.append(c)
            else:
                # try pydantic model dump
                try:
                    clients_data.append(c.model_dump(exclude_none=True))
                except Exception:
                    clients_data.append({})

        df = pd.DataFrame(clients_data)

        # Prétraitement
        if used_preprocessor is not None:
            X = used_preprocessor.transform(df)
        else:
            X = df.values

        # Prédictions
        probabilities = used_model.predict_proba(X)[:, 1]
        
        # Seuil
        if request.get('threshold') is not None:
            used_threshold = request['threshold']
        elif config is not None:
            used_threshold = config.get("optimal_threshold", 0.5)
        else:
            used_threshold = 0.5
        
        # Construire les réponses
        predictions = []
        for i, proba in enumerate(probabilities):
            # Récupérer le dictionnaire client original
            client_data = clients_data[i] if i < len(clients_data) else {}
            prediction = 1 if proba >= used_threshold else 0
            predictions.append(PredictionResponse(
                client_id=client_data.get('SK_ID_CURR') if isinstance(client_data, dict) else None,
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
async def explain_prediction(payload: dict = Body(...), model_dep = Depends(get_model), preprocessor_dep = Depends(get_preprocessor), explainer_dep = Depends(get_explainer)):
    """
    Explique la prédiction pour un client avec les features les plus influentes.
    
    Utilise l'importance des features du modèle pour identifier
    les facteurs qui contribuent le plus à la décision.
    """
    used_model = model_dep or model
    used_preprocessor = preprocessor_dep or preprocessor
    used_explainer = explainer_dep

    if used_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Accept wrapper {'features': {...}} or direct dict
        if isinstance(payload, dict) and 'features' in payload:
            client_dict = payload['features']
        else:
            client_dict = payload

        df = pd.DataFrame([client_dict])
        
        # Prétraitement
        if used_preprocessor is not None:
            X = used_preprocessor.transform(df)
        else:
            X = df.values

        # Prédiction
        probability = float(used_model.predict_proba(X)[0, 1])
        threshold = config.get("optimal_threshold", 0.5) if config is not None else 0.5
        prediction = 1 if probability >= threshold else 0

        # Feature importance (globale pour simplifier)
        feature_importances = getattr(used_model, 'feature_importances_', None)
        # try alternative attr names on mocks
        if feature_importances is None:
            feature_importances = getattr(used_model, 'feature_importances', None)

        feature_names = getattr(used_preprocessor, 'feature_names', None)
        if feature_names is None:
            feature_names = getattr(used_model, 'feature_names_', None) or getattr(used_model, 'feature_names', None)

        if feature_importances is None or feature_names is None:
            # cannot compute detailed explanation, fallback empty list
            return ExplanationResponse(
                client_id=client_dict.get('SK_ID_CURR') if isinstance(client_dict, dict) else None,
                probability=round(probability, 4),
                prediction=prediction,
                decision=Decision.REFUSED if prediction == 1 else Decision.ACCEPTED,
                top_features=[]
            )
        
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
async def get_feature_importance(top_n: int = Query(20, ge=1, le=100), model_dep = Depends(get_model), preprocessor_dep = Depends(get_preprocessor)):
    """
    Obtenir l'importance des features du modèle.
    
    - **top_n**: Nombre de features à retourner (1-100)
    """
    used_model = model_dep or model
    used_preprocessor = preprocessor_dep or preprocessor

    if used_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        feature_importances = getattr(used_model, 'feature_importances_', None) or getattr(used_model, 'feature_importances', None)
        feature_names = getattr(used_preprocessor, 'feature_names', None) or getattr(used_model, 'feature_names_', None) or getattr(used_model, 'feature_names', None)
        if feature_importances is None or feature_names is None:
            # fallback: return feature names with zero importance
            if feature_names is None:
                raise HTTPException(status_code=500, detail="Feature names indisponibles")
            result = []
            for rank, fname in enumerate(feature_names[:top_n], 1):
                result.append(FeatureImportance(feature=fname, importance=0.0, rank=rank))
            return result
        
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


# ---------------------------------------------------------------------------
# Data endpoints: descriptive stats & drift reports
# ---------------------------------------------------------------------------


@app.get("/data/describe", tags=["Data"])
async def data_describe(source: Optional[str] = None):
    """Retourne des statistiques descriptives pour le jeu de données.

    - `source`: optionnel, chemin local relatif ou URL à télécharger
    """
    # Déterminer la source des données
    data_path = None
    # Priorité: query param `source` -> env DATA_URL -> local data/application_train.csv
    env_url = os.environ.get("DATA_URL")
    try:
        if source:
            # si c'est une URL, tenter de télécharger
            if source.startswith("http"):
                df = pd.read_csv(source)
            else:
                data_path = Path(source)
        elif env_url:
            if env_url.startswith("http"):
                df = pd.read_csv(env_url)
            else:
                data_path = Path(env_url)
        else:
            candidate = DATA_DIR / "application_train.csv"
            if candidate.exists():
                data_path = candidate

        if data_path is not None:
            df = pd.read_csv(data_path)

        # Calculer stats
        desc = df.describe(include='all').to_dict()
        missing = df.isna().sum().to_dict()
        target_counts = df['TARGET'].value_counts().to_dict() if 'TARGET' in df.columns else {}

        return {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "describe": desc,
            "missing": missing,
            "target_distribution": target_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture des données: {e}")



@app.get("/data/drift", response_class=HTMLResponse, tags=["Data"])
async def data_drift_report():
    """Sert le rapport Evidently (HTML) si présent dans `reports/`.

    Si le fichier HTML n'existe pas, retourne 404.
    """
    # Chercher des fichiers report HTML connus
    candidates = [REPORTS_DIR / "evidently_full_report.html", REPORTS_DIR / "data_drift_report.html"]
    for c in candidates:
        if c.exists():
            return HTMLResponse(content=c.read_text(encoding='utf-8'), status_code=200)

    raise HTTPException(status_code=404, detail="Rapport de drift introuvable dans reports/")


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
