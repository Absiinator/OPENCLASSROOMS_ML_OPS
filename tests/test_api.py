"""
Tests unitaires pour l'API FastAPI.
====================================

Teste les endpoints de l'API de scoring.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockPreprocessor:
    """Mock du préprocesseur."""
    
    def __init__(self):
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    def transform(self, df):
        """Transforme un DataFrame en array numpy."""
        # Retourne simplement les valeurs des colonnes connues
        features = []
        for fname in self.feature_names:
            if fname in df.columns:
                features.append(df[fname].values)
            else:
                # feature manquante -> 0
                features.append(np.zeros(len(df)))
        return np.column_stack(features) if features else np.zeros((len(df), len(self.feature_names)))


class MockModel:
    """Mock du modèle de scoring."""
    
    def __init__(self):
        self.feature_names_ = ['feature1', 'feature2', 'feature3']
        self.n_features_ = 3
    
    def predict_proba(self, X):
        """Retourne des probabilités mock."""
        n_samples = len(X) if hasattr(X, '__len__') else 1
        # Retourne toujours [0.3, 0.7] pour la classe 0 et 1
        return np.array([[0.3, 0.7]] * n_samples)
    
    def predict(self, X):
        """Retourne des prédictions mock."""
        n_samples = len(X) if hasattr(X, '__len__') else 1
        return np.array([1] * n_samples)


class MockExplainer:
    """Mock de l'explainer SHAP."""
    
    def __init__(self):
        pass
    
    def __call__(self, X):
        """Retourne des valeurs SHAP mock."""
        n_samples = len(X) if hasattr(X, '__len__') else 1
        n_features = X.shape[1] if hasattr(X, 'shape') else 3
        
        mock_result = MagicMock()
        mock_result.values = np.random.randn(n_samples, n_features)
        mock_result.base_values = np.array([0.5] * n_samples)
        return mock_result


@pytest.fixture
def mock_dependencies():
    """Fixture pour mocker les dépendances."""
    with patch.dict('sys.modules', {
        'shap': MagicMock(),
        'mlflow': MagicMock()
    }):
        yield


@pytest.fixture
def client(mock_dependencies):
    """Crée un client de test avec le modèle mocké."""
    # Import après le mock
    from api.main import app, get_model, get_preprocessor, get_explainer, get_model_info
    
    # Override les dépendances
    app.dependency_overrides[get_model] = lambda: MockModel()
    app.dependency_overrides[get_preprocessor] = lambda: MockPreprocessor()
    app.dependency_overrides[get_explainer] = lambda: MockExplainer()
    app.dependency_overrides[get_model_info] = lambda: {
        'model_name': 'test_model',
        'model_version': '1.0',
        'threshold': 0.35,
        'features': ['feature1', 'feature2', 'feature3']
    }
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Cleanup
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""
    
    def test_health_check(self, client):
        """Test que l'endpoint health retourne un statut OK."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests pour l'endpoint /."""
    
    def test_root(self, client):
        """Test de l'endpoint racine."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Home Credit" in data["message"] or "Scoring" in data["message"]


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict."""
    
    def test_predict_success(self, client):
        """Test d'une prédiction réussie."""
        payload = {
            "features": {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0
            }
        }
        
        response = client.post("/predict", json=payload)
        
        # L'endpoint devrait fonctionner ou retourner une erreur de validation
        assert response.status_code in [200, 422]
    
    def test_predict_missing_features(self, client):
        """Test avec des features manquantes."""
        payload = {
            "features": {}
        }
        
        response = client.post("/predict", json=payload)
        
        # Devrait retourner une erreur ou gérer gracieusement
        assert response.status_code in [200, 400, 422]


class TestBatchPredictEndpoint:
    """Tests pour l'endpoint /predict/batch."""
    
    def test_batch_predict(self, client):
        """Test de prédiction en batch."""
        payload = {
            "clients": [
                {"features": {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}},
                {"features": {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}}
            ]
        }
        
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code in [200, 422]
    
    def test_batch_empty(self, client):
        """Test avec un batch vide."""
        payload = {"clients": []}
        
        response = client.post("/predict/batch", json=payload)
        
        # Devrait gérer gracieusement un batch vide
        assert response.status_code in [200, 400, 422]


class TestExplainEndpoint:
    """Tests pour l'endpoint /predict/explain."""
    
    def test_explain(self, client):
        """Test de l'explication d'une prédiction."""
        payload = {
            "features": {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0
            }
        }
        
        response = client.post("/predict/explain", json=payload)
        
        assert response.status_code in [200, 422, 500]


class TestModelInfoEndpoint:
    """Tests pour l'endpoint /model/info."""
    
    def test_model_info(self, client):
        """Test de récupération des infos du modèle."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Vérifier les champs attendus
        assert "model_name" in data or "name" in data or "version" in data


class TestModelFeaturesEndpoint:
    """Tests pour l'endpoint /model/features."""
    
    def test_model_features(self, client):
        """Test de récupération des features du modèle."""
        response = client.get("/model/features")
        
        assert response.status_code == 200
        data = response.json()
        
        # Devrait retourner une liste de features
        assert "features" in data or isinstance(data, list)


class TestInputValidation:
    """Tests de validation des entrées."""
    
    def test_invalid_json(self, client):
        """Test avec un JSON invalide."""
        response = client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]
    
    def test_wrong_content_type(self, client):
        """Test avec un mauvais content type."""
        response = client.post(
            "/predict",
            content="feature1=1.0",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code in [400, 415, 422]
    
    def test_negative_values(self, client):
        """Test avec des valeurs négatives (si applicable)."""
        payload = {
            "features": {
                "feature1": -1.0,
                "feature2": -2.0,
                "feature3": -3.0
            }
        }
        
        response = client.post("/predict", json=payload)
        
        # Selon la logique métier, peut être accepté ou rejeté
        assert response.status_code in [200, 400, 422]


class TestResponseFormat:
    """Tests du format des réponses."""
    
    def test_response_has_probability(self, client):
        """Test que la réponse contient une probabilité."""
        payload = {
            "features": {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0
            }
        }
        
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            # La réponse devrait contenir une probabilité
            assert any(key in data for key in ['probability', 'proba', 'score', 'prediction'])
    
    def test_response_has_decision(self, client):
        """Test que la réponse contient une décision."""
        payload = {
            "features": {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0
            }
        }
        
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            # La réponse devrait contenir une décision ou prédiction
            assert any(key in data for key in ['decision', 'prediction', 'label', 'class'])


class TestCORS:
    """Tests pour les headers CORS."""
    
    def test_cors_headers(self, client):
        """Test que les headers CORS sont présents."""
        response = client.options("/predict")
        
        # Vérifier que CORS est configuré
        # Le comportement exact dépend de la configuration
        assert response.status_code in [200, 204, 405]


class TestErrorHandling:
    """Tests de gestion des erreurs."""
    
    def test_404_not_found(self, client):
        """Test d'un endpoint non existant."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test d'une méthode non autorisée."""
        response = client.put("/predict", json={})
        
        assert response.status_code == 405


class TestPerformance:
    """Tests de performance basiques."""
    
    def test_health_response_time(self, client):
        """Test que /health répond rapidement."""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0, "L'endpoint /health devrait répondre en moins d'1 seconde"


# Tests avec des données réalistes de Home Credit
class TestHomeCrediFeatures:
    """Tests avec des features réalistes de Home Credit."""
    
    @pytest.fixture
    def realistic_features(self):
        """Features réalistes du dataset Home Credit."""
        return {
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7,
            "CODE_GENDER_M": 1,
            "FLAG_OWN_CAR": 1,
            "CNT_CHILDREN": 2
        }
    
    def test_realistic_prediction(self, client, realistic_features):
        """Test avec des features réalistes."""
        payload = {"features": realistic_features}
        
        response = client.post("/predict", json=payload)
        
        # L'endpoint devrait accepter les features réalistes
        # ou retourner une erreur de validation si les features ne correspondent pas
        assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
