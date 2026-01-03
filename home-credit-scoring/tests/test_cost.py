"""
Tests unitaires pour le module de coût métier.
=============================================

Teste les fonctions de calcul du coût métier et d'optimisation du seuil.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    compute_business_cost,
    compute_normalized_cost,
    compute_cost_matrix,
    find_optimal_threshold,
    compute_all_metrics,
    COST_FN,
    COST_FP
)


class TestBusinessCost:
    """Tests pour le calcul du coût métier."""
    
    def test_cost_perfect_predictions(self):
        """Test avec des prédictions parfaites (coût = 0)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        
        cost = compute_business_cost(y_true, y_pred)
        assert cost == 0, "Le coût devrait être 0 pour des prédictions parfaites"
    
    def test_cost_all_false_negatives(self):
        """Test avec uniquement des faux négatifs."""
        y_true = np.array([1, 1, 1, 1])  # Tous en défaut
        y_pred = np.array([0, 0, 0, 0])  # Tous prédits non-défaut
        
        cost = compute_business_cost(y_true, y_pred)
        expected_cost = 4 * COST_FN  # 4 FN
        assert cost == expected_cost, f"Coût attendu: {expected_cost}, obtenu: {cost}"
    
    def test_cost_all_false_positives(self):
        """Test avec uniquement des faux positifs."""
        y_true = np.array([0, 0, 0, 0])  # Tous non-défaut
        y_pred = np.array([1, 1, 1, 1])  # Tous prédits défaut
        
        cost = compute_business_cost(y_true, y_pred)
        expected_cost = 4 * COST_FP  # 4 FP
        assert cost == expected_cost, f"Coût attendu: {expected_cost}, obtenu: {cost}"
    
    def test_cost_mixed(self):
        """Test avec un mélange de FN et FP."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])  # 1 FP, 1 FN
        
        cost = compute_business_cost(y_true, y_pred)
        expected_cost = 1 * COST_FN + 1 * COST_FP  # 10 + 1 = 11
        assert cost == expected_cost, f"Coût attendu: {expected_cost}, obtenu: {cost}"
    
    def test_cost_custom_weights(self):
        """Test avec des poids personnalisés."""
        y_true = np.array([0, 1])
        y_pred = np.array([1, 0])  # 1 FP, 1 FN
        
        cost = compute_business_cost(y_true, y_pred, cost_fn=20, cost_fp=5)
        expected_cost = 20 + 5  # 25
        assert cost == expected_cost
    
    def test_fn_more_costly_than_fp(self):
        """Vérifie que FN coûte plus cher que FP par défaut."""
        # 1 FN
        y_true_fn = np.array([1])
        y_pred_fn = np.array([0])
        cost_fn = compute_business_cost(y_true_fn, y_pred_fn)
        
        # 1 FP
        y_true_fp = np.array([0])
        y_pred_fp = np.array([1])
        cost_fp = compute_business_cost(y_true_fp, y_pred_fp)
        
        assert cost_fn > cost_fp, "Un FN devrait coûter plus qu'un FP"
        assert cost_fn == COST_FN
        assert cost_fp == COST_FP


class TestNormalizedCost:
    """Tests pour le coût normalisé."""
    
    def test_normalized_cost(self):
        """Test du coût normalisé."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])  # 1 FP, 1 FN
        
        normalized = compute_normalized_cost(y_true, y_pred)
        expected = (COST_FN + COST_FP) / 4  # (10 + 1) / 4 = 2.75
        assert normalized == expected


class TestCostMatrix:
    """Tests pour la matrice de coût."""
    
    def test_cost_matrix_values(self):
        """Test des valeurs de la matrice de coût."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1])  # TN=2, FP=1, FN=1, TP=1
        
        result = compute_cost_matrix(y_true, y_pred)
        
        assert result['TN'] == 2
        assert result['FP'] == 1
        assert result['FN'] == 1
        assert result['TP'] == 1
        assert result['cost_FP'] == 1 * COST_FP
        assert result['cost_FN'] == 1 * COST_FN
        assert result['total_cost'] == COST_FP + COST_FN


class TestOptimalThreshold:
    """Tests pour l'optimisation du seuil."""
    
    def test_find_optimal_threshold_basic(self):
        """Test basique de recherche du seuil optimal."""
        np.random.seed(42)
        
        # Données simulées
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])
        
        threshold, cost, details = find_optimal_threshold(y_true, y_proba)
        
        assert 0 < threshold < 1, "Le seuil doit être entre 0 et 1"
        assert cost >= 0, "Le coût doit être positif ou nul"
        assert 'all_results' in details
    
    def test_threshold_range(self):
        """Test que le seuil est dans une plage valide."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7])
        
        threshold, _, _ = find_optimal_threshold(y_true, y_proba)
        
        assert 0.01 <= threshold <= 0.99, "Le seuil doit être dans [0.01, 0.99]"
    
    def test_optimal_threshold_minimizes_cost(self):
        """Vérifie que le seuil optimal minimise le coût."""
        np.random.seed(42)
        n = 100
        
        y_true = np.random.binomial(1, 0.2, n)
        y_proba = y_true * 0.5 + np.random.uniform(0.1, 0.4, n)
        y_proba = np.clip(y_proba, 0, 1)
        
        optimal_threshold, optimal_cost, details = find_optimal_threshold(y_true, y_proba)
        
        # Calculer le coût pour d'autres seuils
        for result in details['all_results']:
            assert optimal_cost <= result['cost'], \
                f"Le seuil optimal ({optimal_threshold}) devrait avoir le coût minimal"


class TestAllMetrics:
    """Tests pour le calcul de toutes les métriques."""
    
    def test_metrics_structure(self):
        """Test de la structure des métriques retournées."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        
        metrics = compute_all_metrics(y_true, y_proba, threshold=0.5)
        
        required_keys = ['auc', 'accuracy', 'precision', 'recall', 
                        'f1_score', 'threshold', 'business_cost', 
                        'normalized_cost', 'TP', 'TN', 'FP', 'FN']
        
        for key in required_keys:
            assert key in metrics, f"Clé manquante: {key}"
    
    def test_metrics_ranges(self):
        """Test que les métriques sont dans des plages valides."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        
        metrics = compute_all_metrics(y_true, y_proba, threshold=0.5)
        
        assert 0 <= metrics['auc'] <= 1, "AUC doit être entre 0 et 1"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy doit être entre 0 et 1"
        assert 0 <= metrics['precision'] <= 1, "Precision doit être entre 0 et 1"
        assert 0 <= metrics['recall'] <= 1, "Recall doit être entre 0 et 1"
        assert 0 <= metrics['f1_score'] <= 1, "F1 doit être entre 0 et 1"
    
    def test_perfect_auc(self):
        """Test AUC parfait."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])  # Parfaitement séparé
        
        metrics = compute_all_metrics(y_true, y_proba, threshold=0.5)
        
        assert metrics['auc'] == 1.0, "AUC devrait être 1.0 pour une séparation parfaite"


class TestEdgeCases:
    """Tests des cas limites."""
    
    def test_empty_arrays(self):
        """Test avec des tableaux vides."""
        with pytest.raises(Exception):
            compute_business_cost(np.array([]), np.array([]))
    
    def test_single_element(self):
        """Test avec un seul élément."""
        y_true = np.array([1])
        y_pred = np.array([0])  # FN
        
        cost = compute_business_cost(y_true, y_pred)
        assert cost == COST_FN
    
    def test_all_same_class(self):
        """Test quand tous les échantillons sont de la même classe."""
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = compute_all_metrics(y_true, y_proba, threshold=0.5)
        
        assert metrics['TN'] == 4
        assert metrics['TP'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
