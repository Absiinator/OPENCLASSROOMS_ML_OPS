"""
Module d'importance des features avec SHAP.
==========================================

Ce module contient:
- Importance globale des features (LightGBM native)
- Importance locale avec SHAP
- Visualisations SHAP
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP non installé. Certaines fonctionnalités seront limitées.")

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_global_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 30
) -> pd.DataFrame:
    """
    Retourne l'importance globale des features (méthode native LightGBM).
    
    Args:
        model: Modèle LightGBM entraîné
        feature_names: Noms des features
        top_n: Nombre de features à retourner
        
    Returns:
        DataFrame avec feature, importance, rank
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance = importance.sort_values('importance', ascending=False).reset_index(drop=True)
    importance['rank'] = range(1, len(importance) + 1)
    importance['importance_pct'] = importance['importance'] / importance['importance'].sum() * 100
    
    return importance.head(top_n)


def plot_global_importance(
    model,
    feature_names: List[str],
    top_n: int = 30,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualise l'importance globale des features.
    """
    importance = get_global_feature_importance(model, feature_names, top_n)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
    
    bars = ax.barh(range(len(importance)), importance['importance'], color=colors)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance['feature'])
    ax.invert_yaxis()
    
    # Ajouter les pourcentages
    for i, (bar, pct) in enumerate(zip(bars, importance['importance_pct'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    ax.set_xlabel('Importance (gain)', fontsize=12)
    ax.set_title(f'Top {len(importance)} Feature Importances - LightGBM', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
    
    return fig


class SHAPExplainer:
    """
    Classe pour les explications SHAP.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialise l'explainer SHAP.
        
        Args:
            model: Modèle LightGBM
            feature_names: Noms des features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP n'est pas installé. pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
    
    def compute_shap_values(
        self, 
        X: np.ndarray,
        max_samples: int = 1000
    ) -> np.ndarray:
        """
        Calcule les valeurs SHAP pour les données.
        
        Args:
            X: Features (array)
            max_samples: Nombre max d'échantillons (pour performance)
            
        Returns:
            Valeurs SHAP
        """
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # Pour classification binaire, prendre la classe positive
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    
    def get_local_explanation(
        self,
        X_single: np.ndarray,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Explique une prédiction individuelle.
        
        Args:
            X_single: Features d'un seul client (1D array)
            top_n: Nombre de features à expliquer
            
        Returns:
            Dictionnaire avec les contributions des features
        """
        X_single = X_single.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X_single)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values = shap_values.flatten()
        
        # Trier par valeur absolue
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
        
        explanation = {
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                               else self.explainer.expected_value),
            'contributions': []
        }
        
        for idx in sorted_indices:
            explanation['contributions'].append({
                'feature': self.feature_names[idx],
                'value': float(X_single[0, idx]),
                'shap_value': float(shap_values[idx]),
                'direction': 'augmente' if shap_values[idx] > 0 else 'diminue'
            })
        
        return explanation
    
    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Génère le summary plot SHAP.
        """
        shap_values = self.compute_shap_values(X)
        
        # Créer un DataFrame pour les noms de features
        X_df = pd.DataFrame(X[:len(shap_values)], columns=self.feature_names)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_df,
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Summary Plot - Impact des Features', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ SHAP Summary sauvegardé: {save_path}")
        
        return plt.gcf()
    
    def plot_force(
        self,
        X_single: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """
        Génère le force plot SHAP pour une prédiction.
        """
        X_single = X_single.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X_single)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        expected_value = (self.explainer.expected_value[1] 
                         if isinstance(self.explainer.expected_value, list) 
                         else self.explainer.expected_value)
        
        force_plot = shap.force_plot(
            expected_value,
            shap_values,
            X_single,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ SHAP Force Plot sauvegardé: {save_path}")
        
        return force_plot
    
    def plot_waterfall(
        self,
        X_single: np.ndarray,
        max_display: int = 15,
        save_path: Optional[Path] = None
    ):
        """
        Génère le waterfall plot SHAP pour une prédiction.
        """
        X_single = X_single.reshape(1, -1)
        
        shap_values = self.explainer(X_single)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(shap_values[0], max_display=max_display, show=False)
        
        plt.title('SHAP Waterfall - Contribution des Features', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ SHAP Waterfall sauvegardé: {save_path}")
        
        return plt.gcf()


def explain_prediction(
    model,
    preprocessor,
    client_data: Union[pd.DataFrame, Dict[str, Any]],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Explique une prédiction avec les contributions des features.
    
    Args:
        model: Modèle entraîné
        preprocessor: Préprocesseur
        client_data: Données du client
        top_n: Nombre de features à expliquer
        
    Returns:
        Dictionnaire d'explication
    """
    # Convertir en DataFrame si nécessaire
    if isinstance(client_data, dict):
        df = pd.DataFrame([client_data])
    else:
        df = client_data.copy()
    
    # Prétraiter
    X = preprocessor.transform(df)
    
    # Prédiction
    proba = model.predict_proba(X)[0, 1]
    
    # Explication SHAP si disponible
    if SHAP_AVAILABLE:
        explainer = SHAPExplainer(model, preprocessor.feature_names)
        local_explanation = explainer.get_local_explanation(X[0], top_n)
    else:
        # Fallback: utiliser l'importance globale
        importance = get_global_feature_importance(model, preprocessor.feature_names, top_n)
        local_explanation = {
            'base_value': 0.5,
            'contributions': [
                {'feature': row['feature'], 'value': float(X[0, preprocessor.feature_names.index(row['feature'])]),
                 'importance': row['importance'], 'direction': 'N/A'}
                for _, row in importance.iterrows()
            ],
            'note': 'SHAP non disponible, importance globale utilisée'
        }
    
    result = {
        'probability': float(proba),
        'prediction': int(proba >= 0.5),
        'explanation': local_explanation
    }
    
    return result


def generate_shap_report(
    model,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path = REPORTS_DIR
) -> Dict[str, Path]:
    """
    Génère un rapport complet SHAP.
    
    Returns:
        Dictionnaire des chemins vers les fichiers générés
    """
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP non disponible, génération du rapport impossible")
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    explainer = SHAPExplainer(model, feature_names)
    
    # Summary plot
    print("📊 Génération SHAP Summary Plot...")
    fig = explainer.plot_summary(X, save_path=output_dir / "shap_summary.png")
    plt.close(fig)
    generated_files['summary'] = output_dir / "shap_summary.png"
    
    # Importance globale (pour comparaison)
    print("📊 Génération Feature Importance Plot...")
    fig = plot_global_importance(model, feature_names, save_path=output_dir / "feature_importance_detailed.png")
    plt.close(fig)
    generated_files['importance'] = output_dir / "feature_importance_detailed.png"
    
    # Exemple de waterfall pour le premier échantillon
    print("📊 Génération SHAP Waterfall exemple...")
    fig = explainer.plot_waterfall(X[0], save_path=output_dir / "shap_waterfall_example.png")
    plt.close(fig)
    generated_files['waterfall'] = output_dir / "shap_waterfall_example.png"
    
    print(f"✅ Rapport SHAP généré dans {output_dir}")
    
    return generated_files


if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module feature_importance...")
    
    try:
        # Charger le modèle
        model = joblib.load(MODELS_DIR / "lgbm_model.joblib")
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        
        print(f"✅ Modèle et préprocesseur chargés")
        
        # Importance globale
        importance = get_global_feature_importance(model, preprocessor.feature_names, 10)
        print(f"\n📊 Top 10 features (importance globale):")
        for _, row in importance.iterrows():
            print(f"   {row['rank']}. {row['feature']}: {row['importance']:.2f} ({row['importance_pct']:.1f}%)")
        
        if SHAP_AVAILABLE:
            print("\n✅ SHAP disponible pour explications locales")
        else:
            print("\n⚠️ SHAP non disponible")
            
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("   Exécutez d'abord src/train.py pour entraîner le modèle")
