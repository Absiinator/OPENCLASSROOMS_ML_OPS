"""
Module d'inférence pour le scoring crédit.
==========================================

Ce module contient:
- Chargement du modèle et du préprocesseur
- Prédictions sur nouvelles données
- API de prédiction
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import joblib

# Chemins par défaut
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class CreditScoringModel:
    """
    Classe pour l'inférence du modèle de scoring crédit.
    
    Encapsule le modèle, le préprocesseur et le seuil optimal.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialise le modèle de scoring.
        
        Args:
            model_path: Chemin vers le modèle joblib
            preprocessor_path: Chemin vers le préprocesseur
            config_path: Chemin vers la configuration (seuil, etc.)
        """
        self.model_path = model_path or MODELS_DIR / "lgbm_model.joblib"
        self.preprocessor_path = preprocessor_path or MODELS_DIR / "preprocessor.joblib"
        self.config_path = config_path or MODELS_DIR / "model_config.json"
        
        self.model = None
        self.preprocessor = None
        self.config = None
        self.optimal_threshold = 0.5
        
        self._load()
    
    def _load(self):
        """Charge le modèle, le préprocesseur et la configuration."""
        # Charger le modèle
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"✅ Modèle chargé: {self.model_path}")
        else:
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        # Charger le préprocesseur
        if self.preprocessor_path.exists():
            self.preprocessor = joblib.load(self.preprocessor_path)
            print(f"✅ Préprocesseur chargé: {self.preprocessor_path}")
        else:
            raise FileNotFoundError(f"Préprocesseur non trouvé: {self.preprocessor_path}")
        
        # Charger la configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.optimal_threshold = self.config.get('optimal_threshold', 0.5)
            print(f"✅ Configuration chargée: seuil={self.optimal_threshold:.3f}")
        else:
            print(f"⚠️ Configuration non trouvée, utilisation du seuil par défaut: 0.5")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prédit les probabilités de défaut.
        
        Args:
            X: Features (DataFrame ou array)
            
        Returns:
            Probabilités de défaut (classe 1)
        """
        # Prétraitement si c'est un DataFrame
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Prédiction
        probas = self.model.predict_proba(X_processed)[:, 1]
        
        return probas
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Prédit les classes (0/1) selon le seuil.
        
        Args:
            X: Features
            threshold: Seuil de classification (défaut: optimal_threshold)
            
        Returns:
            Classes prédites (0 ou 1)
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def predict_with_details(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Prédiction avec détails supplémentaires.
        
        Returns:
            Dictionnaire avec probabilité, classe, seuil, risque
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        probas = self.predict_proba(X)
        classes = (probas >= threshold).astype(int)
        
        # Catégoriser le risque
        def get_risk_category(proba):
            if proba < 0.2:
                return "Très faible"
            elif proba < 0.4:
                return "Faible"
            elif proba < 0.6:
                return "Modéré"
            elif proba < 0.8:
                return "Élevé"
            else:
                return "Très élevé"
        
        results = []
        for i, (proba, pred) in enumerate(zip(probas, classes)):
            results.append({
                'index': i,
                'probability': float(proba),
                'prediction': int(pred),
                'decision': 'REFUSÉ' if pred == 1 else 'ACCEPTÉ',
                'risk_category': get_risk_category(proba),
                'threshold_used': threshold
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retourne l'importance des features.
        
        Args:
            top_n: Nombre de features à retourner
            
        Returns:
            DataFrame avec feature et importance
        """
        importance = pd.DataFrame({
            'feature': self.preprocessor.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)


def load_model(
    model_path: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None,
    config_path: Optional[Path] = None
) -> CreditScoringModel:
    """
    Fonction utilitaire pour charger le modèle.
    
    Returns:
        Instance de CreditScoringModel
    """
    return CreditScoringModel(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        config_path=config_path
    )


def predict_single(
    data: Dict[str, Any],
    model: Optional[CreditScoringModel] = None
) -> Dict[str, Any]:
    """
    Prédit pour un seul client.
    
    Args:
        data: Dictionnaire des features du client
        model: Modèle à utiliser (chargé si non fourni)
        
    Returns:
        Résultat de la prédiction
    """
    if model is None:
        model = load_model()
    
    # Convertir en DataFrame
    df = pd.DataFrame([data])
    
    # Prédiction
    results = model.predict_with_details(df)
    
    return results[0]


def predict_batch(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    model: Optional[CreditScoringModel] = None
) -> List[Dict[str, Any]]:
    """
    Prédit pour plusieurs clients.
    
    Args:
        data: DataFrame ou liste de dictionnaires
        model: Modèle à utiliser
        
    Returns:
        Liste des résultats de prédiction
    """
    if model is None:
        model = load_model()
    
    # Convertir en DataFrame si nécessaire
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    return model.predict_with_details(df)


if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module inference...")
    
    try:
        model = load_model()
        print(f"\n✅ Modèle chargé avec succès")
        print(f"   - Seuil optimal: {model.optimal_threshold:.3f}")
        print(f"   - Nombre de features: {len(model.preprocessor.feature_names)}")
        
        # Afficher top features
        top_features = model.get_feature_importance(10)
        print(f"\n📊 Top 10 features:")
        for _, row in top_features.iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")
            
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("   Exécutez d'abord src/train.py pour entraîner le modèle")
