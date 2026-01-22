"""
Module d'entraînement avec tracking MLflow.
==========================================

Ce module contient:
- Entraînement du modèle LightGBM
- Tracking MLflow (params, metrics, artifacts)
- Registration du modèle dans le registry MLflow
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import joblib
import matplotlib.pyplot as plt

from src.preprocessing import (
    prepare_train_test_data, 
    CreditScoringPreprocessor,
    DATA_DIR,
    MODELS_DIR
)
from src.metrics import (
    find_optimal_threshold,
    compute_all_metrics,
    plot_threshold_optimization,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_metrics_report,
    COST_FN,
    COST_FP
)

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "home-credit-scoring"
MODEL_NAME = "home_credit_model"

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME
) -> str:
    """
    Configure MLflow pour le tracking.
    
    Returns:
        ID de l'expérience
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Créer ou récupérer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow configuré:")
    print(f"   - Tracking URI: {tracking_uri}")
    print(f"   - Experiment: {experiment_name}")
    print(f"   - Experiment ID: {experiment_id}")
    
    return experiment_id


def get_default_lgb_params() -> Dict[str, Any]:
    """
    Retourne les hyperparamètres par défaut pour LightGBM.
    """
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 500,
        'early_stopping_rounds': 50,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    feature_names: Optional[list] = None
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Entraîne un modèle LightGBM.
    
    Returns:
        Tuple (modèle, historique d'entraînement)
    """
    if params is None:
        params = get_default_lgb_params()
    
    # Extraire les paramètres de callback
    early_stopping = params.pop('early_stopping_rounds', 50)
    
    # Créer et entraîner le modèle
    model = lgb.LGBMClassifier(**params)
    
    # Callbacks pour early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping),
        lgb.log_evaluation(period=100)
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=callbacks
    )
    
    # Historique
    history = {
        'best_iteration': model.best_iteration_,
        'best_score': model.best_score_
    }
    
    return model, history


def train_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocessor: CreditScoringPreprocessor,
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    run_name: Optional[str] = None,
    register_model: bool = True
) -> Tuple[lgb.LGBMClassifier, str, float]:
    """
    Entraîne un modèle avec tracking MLflow complet.
    
    Returns:
        Tuple (modèle, run_id, seuil_optimal)
    """
    if params is None:
        params = get_default_lgb_params()
    
    if run_name is None:
        run_name = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow
    setup_mlflow()
    
    # Split train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"\n📊 Données d'entraînement:")
    print(f"   - Train: {X_tr.shape[0]} échantillons")
    print(f"   - Validation: {X_val.shape[0]} échantillons")
    print(f"   - Features: {X_tr.shape[1]}")
    print(f"   - Distribution target train: {np.bincount(y_tr.astype(int))}")
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n🚀 MLflow Run démarré: {run_id}")
        
        # Log des paramètres
        mlflow.log_params(params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_features", X_tr.shape[1])
        mlflow.log_param("n_train_samples", X_tr.shape[0])
        mlflow.log_param("n_val_samples", X_val.shape[0])
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        
        # Entraînement
        print("\n🏋️ Entraînement du modèle...")
        model, history = train_lightgbm(
            X_tr, y_tr, X_val, y_val, 
            params.copy(),
            feature_names=preprocessor.feature_names
        )
        
        # Prédictions
        y_proba_train = model.predict_proba(X_tr)[:, 1]
        y_proba_val = model.predict_proba(X_val)[:, 1]
        
        # Optimisation du seuil
        optimal_threshold, optimal_cost, details = find_optimal_threshold(
            y_val, y_proba_val, COST_FN, COST_FP
        )
        print(f"\n🎯 Seuil optimal trouvé: {optimal_threshold:.3f}")
        
        # Métriques
        metrics_train = compute_all_metrics(y_tr, y_proba_train, optimal_threshold)
        metrics_val = compute_all_metrics(y_val, y_proba_val, optimal_threshold)
        
        # Log des métriques
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        mlflow.log_metric("train_auc", metrics_train['auc'])
        mlflow.log_metric("train_accuracy", metrics_train['accuracy'])
        mlflow.log_metric("train_business_cost", metrics_train['business_cost'])
        mlflow.log_metric("val_auc", metrics_val['auc'])
        mlflow.log_metric("val_accuracy", metrics_val['accuracy'])
        mlflow.log_metric("val_precision", metrics_val['precision'])
        mlflow.log_metric("val_recall", metrics_val['recall'])
        mlflow.log_metric("val_f1_score", metrics_val['f1_score'])
        mlflow.log_metric("val_business_cost", metrics_val['business_cost'])
        mlflow.log_metric("val_normalized_cost", metrics_val['normalized_cost'])
        mlflow.log_metric("best_iteration", history['best_iteration'])
        
        print(f"\n📈 Métriques de validation:")
        print(f"   - AUC: {metrics_val['auc']:.4f}")
        print(f"   - Accuracy: {metrics_val['accuracy']:.4f}")
        print(f"   - Precision: {metrics_val['precision']:.4f}")
        print(f"   - Recall: {metrics_val['recall']:.4f}")
        print(f"   - F1-Score: {metrics_val['f1_score']:.4f}")
        print(f"   - Coût métier: {metrics_val['business_cost']:,.0f}")
        
        # Générer et sauvegarder les artifacts
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Plot optimisation seuil
        fig_threshold = plot_threshold_optimization(
            y_val, y_proba_val, COST_FN, COST_FP,
            save_path=REPORTS_DIR / "threshold_optimization.png"
        )
        plt.close(fig_threshold)
        mlflow.log_artifact(str(REPORTS_DIR / "threshold_optimization.png"))
        
        # Plot confusion matrix
        y_pred_val = (y_proba_val >= optimal_threshold).astype(int)
        fig_cm = plot_confusion_matrix(
            y_val, y_pred_val, COST_FN, COST_FP,
            save_path=REPORTS_DIR / "confusion_matrix.png"
        )
        plt.close(fig_cm)
        mlflow.log_artifact(str(REPORTS_DIR / "confusion_matrix.png"))
        
        # Plot ROC curve
        fig_roc = plot_roc_curve(
            y_val, y_proba_val, optimal_threshold,
            save_path=REPORTS_DIR / "roc_curve.png"
        )
        plt.close(fig_roc)
        mlflow.log_artifact(str(REPORTS_DIR / "roc_curve.png"))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': preprocessor.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_fi, ax = plt.subplots(figsize=(10, 12))
        top_n = min(30, len(feature_importance))
        ax.barh(range(top_n), feature_importance['importance'].head(top_n))
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feature_importance['feature'].head(top_n))
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances (LightGBM)')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close(fig_fi)
        mlflow.log_artifact(str(REPORTS_DIR / "feature_importance.png"))
        
        # Sauvegarder feature importance CSV
        feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        mlflow.log_artifact(str(REPORTS_DIR / "feature_importance.csv"))
        
        # Rapport texte
        report = generate_metrics_report(y_val, y_proba_val, COST_FN, COST_FP)
        with open(REPORTS_DIR / "metrics_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact(str(REPORTS_DIR / "metrics_report.txt"))
        
        # Log du modèle avec signature
        signature = infer_signature(X_val, y_proba_val)
        
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME if register_model else None
        )
        
        # Sauvegarder aussi localement
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "lgbm_model.joblib")
        
        # Sauvegarder le seuil optimal (convertir numpy types en Python natifs)
        config = {
            "optimal_threshold": float(optimal_threshold),
            "cost_fn": int(COST_FN),
            "cost_fp": int(COST_FP),
            "model_name": MODEL_NAME,
            "run_id": run_id,
            "metrics": {
                "auc": float(metrics_val['auc']),
                "accuracy": float(metrics_val['accuracy']),
                "business_cost": float(metrics_val['business_cost'])
            }
        }
        with open(MODELS_DIR / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(str(MODELS_DIR / "model_config.json"))
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   - Run ID: {run_id}")
        print(f"   - Modèle sauvegardé: {MODELS_DIR / 'lgbm_model.joblib'}")
        
        if register_model:
            print(f"   - Modèle enregistré dans MLflow: {MODEL_NAME}")
        
        return model, run_id, optimal_threshold


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Effectue une validation croisée stratifiée.
    """
    if params is None:
        params = get_default_lgb_params()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        'auc': [],
        'accuracy': [],
        'business_cost': [],
        'optimal_threshold': []
    }
    
    print(f"\n🔄 Validation croisée ({n_splits} folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
        
        model, _ = train_lightgbm(X_tr, y_tr, X_val, y_val, params.copy())
        y_proba = model.predict_proba(X_val)[:, 1]
        
        threshold, _, _ = find_optimal_threshold(y_val, y_proba)
        metrics = compute_all_metrics(y_val, y_proba, threshold)
        
        cv_results['auc'].append(metrics['auc'])
        cv_results['accuracy'].append(metrics['accuracy'])
        cv_results['business_cost'].append(metrics['business_cost'])
        cv_results['optimal_threshold'].append(threshold)
        
        print(f"   Fold {fold}: AUC={metrics['auc']:.4f}, Cost={metrics['business_cost']:,.0f}")
    
    # Moyennes
    cv_summary = {
        'auc_mean': np.mean(cv_results['auc']),
        'auc_std': np.std(cv_results['auc']),
        'accuracy_mean': np.mean(cv_results['accuracy']),
        'accuracy_std': np.std(cv_results['accuracy']),
        'cost_mean': np.mean(cv_results['business_cost']),
        'cost_std': np.std(cv_results['business_cost']),
        'threshold_mean': np.mean(cv_results['optimal_threshold'])
    }
    
    print(f"\n📊 Résultats CV:")
    print(f"   - AUC: {cv_summary['auc_mean']:.4f} ± {cv_summary['auc_std']:.4f}")
    print(f"   - Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    print(f"   - Business Cost: {cv_summary['cost_mean']:,.0f} ± {cv_summary['cost_std']:,.0f}")
    
    return cv_summary


def main(
    sample_frac: Optional[float] = None,
    include_supplementary: bool = True
):
    """
    Pipeline d'entraînement complet.
    """
    print("=" * 80)
    print("        HOME CREDIT SCORING - ENTRAÎNEMENT MLflow")
    print("=" * 80)
    
    # Préparer les données
    X_train, X_test, y_train, test_ids, preprocessor = prepare_train_test_data(
        include_supplementary=include_supplementary,
        sample_frac=sample_frac
    )
    
    # Entraîner avec MLflow
    model, run_id, optimal_threshold = train_with_mlflow(
        X_train, y_train, preprocessor,
        register_model=True
    )
    
    # Prédictions sur test
    print("\n📤 Génération des prédictions sur le jeu de test...")
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= optimal_threshold).astype(int)
    
    # Sauvegarder les prédictions
    predictions = test_ids.copy()
    predictions['TARGET_PROBA'] = y_proba_test
    predictions['TARGET_PRED'] = y_pred_test
    predictions.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    
    # Soumission Kaggle format
    submission = test_ids.copy()
    submission['TARGET'] = y_proba_test
    submission.to_csv(MODELS_DIR / "submission.csv", index=False)
    
    print(f"✅ Prédictions sauvegardées: {MODELS_DIR / 'test_predictions.csv'}")
    print(f"✅ Soumission Kaggle: {MODELS_DIR / 'submission.csv'}")
    
    print("\n" + "=" * 80)
    print("        ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    
    return model, preprocessor, optimal_threshold


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement Home Credit Scoring")
    parser.add_argument("--sample", type=float, default=None, 
                       help="Fraction des données à utiliser (pour tests)")
    parser.add_argument("--no-supplementary", action="store_true",
                       help="Ne pas inclure les tables supplémentaires")
    
    args = parser.parse_args()
    
    main(
        sample_frac=args.sample,
        include_supplementary=not args.no_supplementary
    )
