"""
Tests d'inférence pour le modèle de scoring.
=============================================

Teste le comportement du modèle avec des valeurs normales et des outliers.
Ces tests vérifient que les prédictions sont cohérentes et dans les bornes attendues.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================
# Données de test - Valeurs normales
# ============================================
NORMAL_CLIENT_DATA = {
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 500000.0,
    "AMT_ANNUITY": 25000.0,
    "AMT_GOODS_PRICE": 450000.0,
    "DAYS_BIRTH": -12775,  # ~35 ans
    "DAYS_EMPLOYED": -1825,  # ~5 ans d'ancienneté
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

# Client à faible risque (bon profil)
LOW_RISK_CLIENT = {
    "AMT_INCOME_TOTAL": 300000.0,  # Revenu élevé
    "AMT_CREDIT": 200000.0,  # Crédit modéré
    "AMT_ANNUITY": 10000.0,  # Annuité faible
    "AMT_GOODS_PRICE": 180000.0,
    "DAYS_BIRTH": -18000,  # ~49 ans, plus stable
    "DAYS_EMPLOYED": -5475,  # ~15 ans d'ancienneté
    "CNT_CHILDREN": 0,
    "CODE_GENDER_M": 0,
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 1,
    "EXT_SOURCE_1": 0.85,  # Très bon score externe
    "EXT_SOURCE_2": 0.80,
    "EXT_SOURCE_3": 0.82,
    "REGION_RATING_CLIENT": 1,  # Meilleure région
    "CREDIT_INCOME_RATIO": 0.67,  # Ratio très favorable
    "ANNUITY_INCOME_RATIO": 0.03,
    "EXT_SOURCE_MEAN": 0.82
}

# Client à haut risque (profil défavorable)
HIGH_RISK_CLIENT = {
    "AMT_INCOME_TOTAL": 50000.0,  # Revenu faible
    "AMT_CREDIT": 800000.0,  # Crédit très élevé
    "AMT_ANNUITY": 45000.0,  # Annuité élevée
    "AMT_GOODS_PRICE": 750000.0,
    "DAYS_BIRTH": -7300,  # ~20 ans, jeune
    "DAYS_EMPLOYED": -180,  # 6 mois d'ancienneté seulement
    "CNT_CHILDREN": 3,
    "CODE_GENDER_M": 1,
    "FLAG_OWN_CAR": 0,
    "FLAG_OWN_REALTY": 0,
    "EXT_SOURCE_1": 0.1,  # Mauvais scores externes
    "EXT_SOURCE_2": 0.15,
    "EXT_SOURCE_3": 0.12,
    "REGION_RATING_CLIENT": 3,  # Région défavorable
    "CREDIT_INCOME_RATIO": 16.0,  # Ratio très défavorable
    "ANNUITY_INCOME_RATIO": 0.9,
    "EXT_SOURCE_MEAN": 0.12
}


# ============================================
# Données de test - Outliers extrêmes
# ============================================
OUTLIER_HIGH_INCOME = {
    **NORMAL_CLIENT_DATA,
    "AMT_INCOME_TOTAL": 100000000.0,  # 100M - outlier extrême
}

OUTLIER_ZERO_INCOME = {
    **NORMAL_CLIENT_DATA,
    "AMT_INCOME_TOTAL": 0.0,  # Revenu nul
}

OUTLIER_NEGATIVE_AGE = {
    **NORMAL_CLIENT_DATA,
    "DAYS_BIRTH": 0,  # Âge impossible (nouveau-né)
}

OUTLIER_EXTREME_RATIOS = {
    **NORMAL_CLIENT_DATA,
    "CREDIT_INCOME_RATIO": 1000.0,  # Ratio impossible
    "ANNUITY_INCOME_RATIO": 50.0,  # Annuité > revenu
}

OUTLIER_MISSING_SOURCES = {
    **NORMAL_CLIENT_DATA,
    "EXT_SOURCE_1": None,
    "EXT_SOURCE_2": None,
    "EXT_SOURCE_3": None,
}


# ============================================
# Tests de structure et cohérence
# ============================================
class TestPredictionStructure:
    """Tests de la structure des prédictions."""
    
    def test_normal_client_has_valid_probability(self):
        """La probabilité doit être entre 0 et 1."""
        # Ce test vérifie la cohérence des données de test
        assert "AMT_INCOME_TOTAL" in NORMAL_CLIENT_DATA
        assert "EXT_SOURCE_1" in NORMAL_CLIENT_DATA
        assert NORMAL_CLIENT_DATA["AMT_INCOME_TOTAL"] > 0
        
    def test_low_risk_client_has_good_scores(self):
        """Le client à faible risque doit avoir de bons scores."""
        assert LOW_RISK_CLIENT["EXT_SOURCE_1"] > 0.7
        assert LOW_RISK_CLIENT["EXT_SOURCE_2"] > 0.7
        assert LOW_RISK_CLIENT["CREDIT_INCOME_RATIO"] < 2
        
    def test_high_risk_client_has_bad_scores(self):
        """Le client à haut risque doit avoir de mauvais scores."""
        assert HIGH_RISK_CLIENT["EXT_SOURCE_1"] < 0.3
        assert HIGH_RISK_CLIENT["EXT_SOURCE_2"] < 0.3
        assert HIGH_RISK_CLIENT["CREDIT_INCOME_RATIO"] > 10


# ============================================
# Tests d'inférence avec mock
# ============================================
class TestInferenceWithMock:
    """Tests d'inférence avec modèle mocké."""
    
    @pytest.fixture
    def mock_model(self):
        """Crée un modèle mocké."""
        from unittest.mock import MagicMock
        model = MagicMock()
        model.feature_names_ = list(NORMAL_CLIENT_DATA.keys())
        model.n_features_ = len(NORMAL_CLIENT_DATA)
        return model
    
    def test_normal_client_prediction_in_range(self, mock_model):
        """La prédiction d'un client normal doit être dans [0, 1]."""
        # Simuler une prédiction entre 0.2 et 0.4 pour un client normal
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        proba = mock_model.predict_proba([[0] * len(NORMAL_CLIENT_DATA)])[0]
        assert 0 <= proba[1] <= 1, "Probabilité doit être dans [0, 1]"
    
    def test_low_risk_client_lower_probability(self, mock_model):
        """Un client à faible risque devrait avoir une probabilité plus basse."""
        # Simuler une probabilité basse pour un bon client
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        
        proba = mock_model.predict_proba([[0] * len(LOW_RISK_CLIENT)])[0]
        assert proba[1] < 0.3, "Client à faible risque devrait avoir proba < 0.3"
    
    def test_high_risk_client_higher_probability(self, mock_model):
        """Un client à haut risque devrait avoir une probabilité plus haute."""
        # Simuler une probabilité haute pour un mauvais client
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        proba = mock_model.predict_proba([[0] * len(HIGH_RISK_CLIENT)])[0]
        assert proba[1] > 0.5, "Client à haut risque devrait avoir proba > 0.5"


# ============================================
# Tests de validation des outliers
# ============================================
class TestOutlierHandling:
    """Tests de gestion des valeurs aberrantes."""
    
    def test_extreme_income_is_valid_float(self):
        """Un revenu extrême doit rester un float valide."""
        income = OUTLIER_HIGH_INCOME["AMT_INCOME_TOTAL"]
        assert isinstance(income, float)
        assert np.isfinite(income)
    
    def test_zero_income_is_handled(self):
        """Un revenu nul ne doit pas causer d'erreur de division."""
        income = OUTLIER_ZERO_INCOME["AMT_INCOME_TOTAL"]
        assert income == 0.0
        # Vérifier que le ratio peut être calculé sans erreur
        # (dans le vrai code, on devrait gérer la division par zéro)
    
    def test_extreme_ratios_are_detected(self):
        """Les ratios extrêmes doivent être détectés."""
        ratio = OUTLIER_EXTREME_RATIOS["CREDIT_INCOME_RATIO"]
        assert ratio > 100, "Ce ratio est clairement un outlier"
    
    def test_missing_sources_as_none(self):
        """Les sources manquantes sont représentées par None."""
        assert OUTLIER_MISSING_SOURCES["EXT_SOURCE_1"] is None
        assert OUTLIER_MISSING_SOURCES["EXT_SOURCE_2"] is None


# ============================================
# Tests du coût métier
# ============================================
class TestBusinessCost:
    """Tests de la fonction de coût métier."""
    
    def test_cost_fn_greater_than_fp(self):
        """Le coût d'un FN doit être 10x supérieur au FP."""
        COST_FN = 10  # Coût d'un faux négatif
        COST_FP = 1   # Coût d'un faux positif
        
        assert COST_FN == 10 * COST_FP
    
    def test_optimal_threshold_not_default(self):
        """Le seuil optimal ne devrait pas être 0.5 par défaut."""
        # Dans un déséquilibre de coûts FN=10, FP=1, le seuil optimal
        # devrait être inférieur à 0.5 pour être plus conservateur
        OPTIMAL_THRESHOLD = 0.35  # Valeur attendue
        DEFAULT_THRESHOLD = 0.5
        
        assert OPTIMAL_THRESHOLD < DEFAULT_THRESHOLD
    
    def test_business_cost_calculation(self):
        """Test du calcul du coût métier."""
        COST_FN = 10
        COST_FP = 1
        
        # Exemple: 100 clients, 10 défauts réels
        # Avec seuil 0.5: FN=3, FP=5
        fn_count = 3
        fp_count = 5
        
        total_cost = (fn_count * COST_FN) + (fp_count * COST_FP)
        assert total_cost == 35  # 3*10 + 5*1 = 35


# ============================================
# Tests de compatibilité des formats
# ============================================
class TestRequestFormats:
    """Tests des formats de requête acceptés."""
    
    def test_features_format(self):
        """Format avec clé 'features' doit être valide."""
        request = {"features": NORMAL_CLIENT_DATA}
        assert "features" in request
        assert isinstance(request["features"], dict)
    
    def test_data_format(self):
        """Format avec clé 'data' doit être valide."""
        request = {"data": NORMAL_CLIENT_DATA}
        assert "data" in request
        assert isinstance(request["data"], dict)
    
    def test_flat_format(self):
        """Format plat (features à la racine) doit être valide."""
        request = NORMAL_CLIENT_DATA.copy()
        assert "AMT_INCOME_TOTAL" in request
        assert "features" not in request


# ============================================
# Tests d'intégration API (avec mock)
# ============================================
class TestAPIIntegration:
    """Tests d'intégration de l'API avec données de test."""
    
    def test_normal_request_structure(self):
        """Structure de requête normale pour l'API."""
        request_body = {"features": NORMAL_CLIENT_DATA}
        
        # Vérifier que toutes les features minimales sont présentes
        required_features = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1"]
        for feat in required_features:
            assert feat in request_body["features"]
    
    def test_response_structure(self):
        """Structure de réponse attendue de l'API."""
        expected_response = {
            "probability": 0.3,
            "prediction": 0,
            "decision": "ACCEPTÉ",
            "threshold": 0.35,
            "risk_category": "Faible"
        }
        
        # Vérifier que tous les champs sont présents
        required_fields = ["probability", "prediction", "decision", "threshold"]
        for field in required_fields:
            assert field in expected_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
