"""
Module de métriques métier pour le scoring crédit.
==================================================

Ce module contient:
- Calcul du coût métier (FN=10, FP=1)
- Optimisation du seuil selon le coût métier
- Métriques standards (AUC, accuracy, F1, etc.)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
from pathlib import Path

# Coûts métier définis
COST_FN = 10  # Coût d'un Faux Négatif (client en défaut non détecté)
COST_FP = 1   # Coût d'un Faux Positif (bon client refusé)


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP
) -> float:
    """
    Calcule le coût métier total.
    
    Logique métier:
    - FN (défaut non détecté) = perte financière majeure → coût=10
    - FP (bon client refusé) = perte de revenu potentiel → coût=1
    - TP (défaut détecté) = évitement de perte → coût=0
    - TN (bon client accepté) = revenu normal → coût=0
    
    Args:
        y_true: Labels réels (1=défaut, 0=non défaut)
        y_pred: Prédictions binaires
        cost_fn: Coût d'un faux négatif
        cost_fp: Coût d'un faux positif
        
    Returns:
        Coût métier total
    """
    # Use labels parameter to ensure 2x2 confusion matrix even for edge cases
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Handle edge case where confusion matrix might not be 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Only one class present - initialize zeros
        tn, fp, fn, tp = 0, 0, 0, 0
        unique = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique) == 1:
            if unique[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    return total_cost


def compute_normalized_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP
) -> float:
    """
    Calcule le coût métier normalisé par le nombre d'échantillons.
    
    Utile pour comparer des modèles sur des datasets de tailles différentes.
    """
    total_cost = compute_business_cost(y_true, y_pred, cost_fn, cost_fp)
    return total_cost / len(y_true)


def compute_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP
) -> Dict[str, float]:
    """
    Calcule le détail du coût métier avec la matrice de confusion.
    
    Returns:
        Dictionnaire avec TP, TN, FP, FN, coûts, et métriques
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    cost_from_fn = fn * cost_fn
    cost_from_fp = fp * cost_fp
    total_cost = cost_from_fn + cost_from_fp
    
    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'cost_FN': cost_from_fn,
        'cost_FP': cost_from_fp,
        'total_cost': total_cost,
        'normalized_cost': total_cost / len(y_true),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict]:
    """
    Trouve le seuil optimal minimisant le coût métier.
    
    Args:
        y_true: Labels réels
        y_proba: Probabilités prédites (pour classe 1)
        cost_fn: Coût d'un faux négatif
        cost_fp: Coût d'un faux positif
        thresholds: Seuils à évaluer (défaut: 0.01 à 0.99)
        
    Returns:
        Tuple (seuil_optimal, coût_minimal, détails)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    
    best_threshold = 0.5
    best_cost = float('inf')
    all_results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost = compute_business_cost(y_true, y_pred, cost_fn, cost_fp)
        
        all_results.append({
            'threshold': threshold,
            'cost': cost,
            'normalized_cost': cost / len(y_true)
        })
        
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    # Détails pour le seuil optimal
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    details = compute_cost_matrix(y_true, y_pred_optimal, cost_fn, cost_fp)
    details['all_results'] = all_results
    
    return best_threshold, best_cost, details


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP
) -> Dict[str, float]:
    """
    Calcule toutes les métriques pour le modèle.
    
    Returns:
        Dictionnaire avec AUC, accuracy, precision, recall, F1, coût métier
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold,
        'business_cost': compute_business_cost(y_true, y_pred, cost_fn, cost_fp),
        'normalized_cost': compute_normalized_cost(y_true, y_pred, cost_fn, cost_fp)
    }
    
    # Ajouter matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    })
    
    return metrics


def plot_threshold_optimization(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualise l'optimisation du seuil selon le coût métier.
    
    Génère un graphique montrant:
    - Coût métier en fonction du seuil
    - Seuil optimal
    - Comparaison avec seuil par défaut (0.5)
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    optimal_threshold, optimal_cost, details = find_optimal_threshold(
        y_true, y_proba, cost_fn, cost_fp, thresholds
    )
    
    # Calculer le coût pour chaque seuil
    costs = [r['cost'] for r in details['all_results']]
    
    # Coût au seuil 0.5
    y_pred_05 = (y_proba >= 0.5).astype(int)
    cost_05 = compute_business_cost(y_true, y_pred_05, cost_fn, cost_fp)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Coût vs Seuil
    ax1 = axes[0]
    ax1.plot(thresholds, costs, 'b-', linewidth=2, label='Coût métier')
    ax1.axvline(x=optimal_threshold, color='g', linestyle='--', linewidth=2, 
                label=f'Seuil optimal ({optimal_threshold:.2f})')
    ax1.axvline(x=0.5, color='r', linestyle=':', linewidth=2, 
                label=f'Seuil défaut (0.5)')
    ax1.scatter([optimal_threshold], [optimal_cost], color='g', s=100, zorder=5)
    ax1.scatter([0.5], [cost_05], color='r', s=100, zorder=5)
    
    ax1.set_xlabel('Seuil de classification', fontsize=12)
    ax1.set_ylabel('Coût métier total', fontsize=12)
    ax1.set_title('Optimisation du seuil métier\n(FN=10, FP=1)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Métriques vs Seuil
    ax2 = axes[1]
    
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))
    
    ax2.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax2.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax2.plot(thresholds, f1s, 'g-', label='F1-Score', linewidth=2)
    ax2.axvline(x=optimal_threshold, color='purple', linestyle='--', linewidth=2,
                label=f'Seuil optimal ({optimal_threshold:.2f})')
    
    ax2.set_xlabel('Seuil de classification', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Métriques en fonction du seuil', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualise la matrice de confusion avec les coûts métier.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Matrice de confusion
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Prédit Négatif\n(Pas de défaut)', 'Prédit Positif\n(Défaut)'])
    ax.set_yticklabels(['Réel Négatif\n(Pas de défaut)', 'Réel Positif\n(Défaut)'])
    
    # Annotations avec coûts
    texts = [
        [f'TN\n{tn}\nCoût: 0', f'FP\n{fp}\nCoût: {fp * cost_fp}'],
        [f'FN\n{fn}\nCoût: {fn * cost_fn}', f'TP\n{tp}\nCoût: 0']
    ]
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, texts[i][j], ha='center', va='center', 
                   color=color, fontsize=11, fontweight='bold')
    
    total_cost = fn * cost_fn + fp * cost_fp
    ax.set_title(f'Matrice de confusion avec coûts métier\n'
                 f'Coût total: {total_cost} (FN×{cost_fn} + FP×{cost_fp})', fontsize=12)
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Matrice de confusion sauvegardée: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: float = 0.5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualise la courbe ROC avec le point du seuil optimal.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Courbe ROC
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire')
    
    # Point du seuil optimal
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.scatter([fpr[idx]], [tpr[idx]], color='red', s=100, zorder=5,
               label=f'Seuil optimal ({optimal_threshold:.2f})')
    
    ax.set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
    ax.set_title('Courbe ROC - Home Credit Scoring', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Courbe ROC sauvegardée: {save_path}")
    
    return fig


def generate_metrics_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP
) -> str:
    """
    Génère un rapport texte des métriques.
    """
    # Seuil optimal
    optimal_threshold, optimal_cost, details = find_optimal_threshold(
        y_true, y_proba, cost_fn, cost_fp
    )
    
    # Métriques au seuil optimal
    metrics_optimal = compute_all_metrics(y_true, y_proba, optimal_threshold, cost_fn, cost_fp)
    
    # Métriques au seuil 0.5
    metrics_default = compute_all_metrics(y_true, y_proba, 0.5, cost_fn, cost_fp)
    
    report = f"""
================================================================================
                    RAPPORT DE MÉTRIQUES - HOME CREDIT SCORING
================================================================================

PARAMÈTRES MÉTIER:
  - Coût Faux Négatif (FN): {cost_fn} (client en défaut non détecté)
  - Coût Faux Positif (FP): {cost_fp} (bon client refusé)

--------------------------------------------------------------------------------
SEUIL PAR DÉFAUT (0.5):
--------------------------------------------------------------------------------
  AUC:              {metrics_default['auc']:.4f}
  Accuracy:         {metrics_default['accuracy']:.4f}
  Precision:        {metrics_default['precision']:.4f}
  Recall:           {metrics_default['recall']:.4f}
  F1-Score:         {metrics_default['f1_score']:.4f}
  
  Matrice de confusion:
    TP: {metrics_default['TP']:,}  |  FP: {metrics_default['FP']:,}
    FN: {metrics_default['FN']:,}  |  TN: {metrics_default['TN']:,}
  
  Coût métier:      {metrics_default['business_cost']:,.0f}

--------------------------------------------------------------------------------
SEUIL OPTIMAL ({optimal_threshold:.2f}):
--------------------------------------------------------------------------------
  AUC:              {metrics_optimal['auc']:.4f}
  Accuracy:         {metrics_optimal['accuracy']:.4f}
  Precision:        {metrics_optimal['precision']:.4f}
  Recall:           {metrics_optimal['recall']:.4f}
  F1-Score:         {metrics_optimal['f1_score']:.4f}
  
  Matrice de confusion:
    TP: {metrics_optimal['TP']:,}  |  FP: {metrics_optimal['FP']:,}
    FN: {metrics_optimal['FN']:,}  |  TN: {metrics_optimal['TN']:,}
  
  Coût métier:      {metrics_optimal['business_cost']:,.0f}

--------------------------------------------------------------------------------
GAIN DU SEUIL OPTIMAL:
--------------------------------------------------------------------------------
  Réduction du coût: {metrics_default['business_cost'] - metrics_optimal['business_cost']:,.0f}
  Réduction relative: {((metrics_default['business_cost'] - metrics_optimal['business_cost']) / metrics_default['business_cost'] * 100):.1f}%

================================================================================
"""
    
    return report


if __name__ == "__main__":
    # Test du module avec des données simulées
    print("🧪 Test du module metrics...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Simuler des données
    y_true = np.random.binomial(1, 0.1, n_samples)  # 10% de défauts
    y_proba = np.clip(y_true * 0.6 + np.random.normal(0.3, 0.15, n_samples), 0, 1)
    
    # Test optimisation seuil
    optimal_threshold, optimal_cost, details = find_optimal_threshold(y_true, y_proba)
    print(f"Seuil optimal: {optimal_threshold:.2f}")
    print(f"Coût optimal: {optimal_cost}")
    
    # Générer rapport
    report = generate_metrics_report(y_true, y_proba)
    print(report)
