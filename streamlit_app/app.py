"""
Application Streamlit pour le scoring crédit Home Credit.
=========================================================

Interface utilisateur pour tester l'API de scoring.
Permet de :
- Saisir les caractéristiques d'un client
- Obtenir une prédiction de risque de défaut
- Visualiser les explications SHAP
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import json
import os

# ============================================
# Configuration de la page
# ============================================
st.set_page_config(
    page_title="Home Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Variables globales
# ============================================
# Endpoint API par défaut (hardcodé pour déploiement/CI stable)
API_URL = "http://localhost:8000"

# Chemins locaux pour fallback (prediction locale si l'API est inaccessible)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lgbm_model.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")

_LOCAL_MODEL_LOADED = False
_LOCAL_MODEL = None
_LOCAL_PREPROCESSOR = None

# ============================================
# Fonctions utilitaires
# ============================================

def check_api_health() -> bool:
    """Vérifie si l'API est accessible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info() -> Optional[Dict[str, Any]]:
    """Récupère les informations du modèle."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_model_features() -> Optional[list]:
    """Récupère la liste des features du modèle."""
    try:
        response = requests.get(f"{API_URL}/model/features", timeout=10)
        if response.status_code == 200:
            return response.json().get("features", [])
    except:
        pass
    return None


def predict(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Effectue une prédiction via l'API."""
    # Première tentative: appel à l'API
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        # Silent fallback to local prediction
        pass

    # Fallback local: charger modèle + préprocesseur et prédire
    try:
        return local_predict(features)
    except Exception as e:
        st.error(f"Erreur de prédiction locale: {e}")
        return None


def explain(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Obtient l'explication SHAP via l'API."""
    # Essayer via API
    try:
        response = requests.post(
            f"{API_URL}/predict/explain",
            json={"features": features},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    # Fallback local (SHAP si disponible)
    try:
        return local_explain(features)
    except Exception as e:
        st.error(f"Erreur d'explication locale: {e}")
        return None


def _load_local_model():
    """Charge et met en cache le préprocesseur et le modèle locaux."""
    global _LOCAL_MODEL_LOADED, _LOCAL_MODEL, _LOCAL_PREPROCESSOR
    if _LOCAL_MODEL_LOADED:
        return
    try:
        import joblib
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            _LOCAL_PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
            _LOCAL_MODEL = joblib.load(MODEL_PATH)
            _LOCAL_MODEL_LOADED = True
    except Exception:
        _LOCAL_MODEL_LOADED = False


def local_predict(features: Dict[str, float]) -> Dict[str, Any]:
    """Effectue une prédiction en local en cas d'indisponibilité de l'API."""
    _load_local_model()
    if not _LOCAL_MODEL_LOADED:
        raise RuntimeError("Modèle local indisponible (models/lgbm_model.joblib manquant)")

    # Préparer dataframe d'entrée
    df = pd.DataFrame([features])

    # Appliquer préprocessing si disponible
    X = df
    if _LOCAL_PREPROCESSOR is not None:
        try:
            X = _LOCAL_PREPROCESSOR.transform(df)
        except Exception:
            # Si transform échoue, tenter d'utiliser le dataframe tel quel
            X = df

    # Prédiction
    try:
        proba = _LOCAL_MODEL.predict_proba(X)[:, 1]
        prob = float(proba[0])
    except Exception:
        # Certains modèles retournent directement une prédiction continue
        pred = _LOCAL_MODEL.predict(X)
        prob = float(pred[0])

    # Seuil par défaut (fallback)
    threshold = 0.44
    decision = "approved" if prob < threshold else "rejected"

    return {"probability": prob, "prediction": int(prob >= threshold), "decision": decision, "threshold": threshold}


def local_explain(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Calcule une explication locale via SHAP si disponible."""
    _load_local_model()
    if not _LOCAL_MODEL_LOADED:
        raise RuntimeError("Modèle local indisponible pour explication")

    try:
        import shap
    except Exception:
        raise RuntimeError("SHAP non installé dans l'environnement; installez 'shap' pour obtenir des explications locales")

    df = pd.DataFrame([features])
    X = df
    if _LOCAL_PREPROCESSOR is not None:
        try:
            X = _LOCAL_PREPROCESSOR.transform(df)
        except Exception:
            X = df

    explainer = shap.Explainer(_LOCAL_MODEL)
    shap_values = explainer(X)

    # Renvoyer un mapping feature -> shap value (pour la première instance)
    try:
        shap_dict = {name: float(val) for name, val in zip(df.columns, shap_values.values[0][: len(df.columns)])}
    except Exception:
        # Fallback: utiliser summary_values si formes différentes
        shap_dict = {f: float(v) for f, v in zip(df.columns, shap_values.values[0])}

    base_value = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.5

    return {"shap_values": shap_dict, "base_value": base_value}


def create_gauge_chart(probability: float, threshold: float = 0.35) -> go.Figure:
    """Crée un graphique de jauge pour la probabilité."""
    
    # Couleur selon le risque
    if probability < threshold * 0.5:
        color = "green"
        risk = "Faible"
    elif probability < threshold:
        color = "orange"
        risk = "Modéré"
    else:
        color = "red"
        risk = "Élevé"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risque de défaut: {risk}", 'font': {'size': 20}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 50], 'color': 'lightgreen'},
                {'range': [threshold * 50, threshold * 100], 'color': 'lightyellow'},
                {'range': [threshold * 100, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_shap_waterfall(shap_values: Dict[str, float], base_value: float) -> go.Figure:
    """Crée un graphique waterfall pour les valeurs SHAP."""
    
    # Trier par valeur absolue
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ['red' if v > 0 else 'blue' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Impact des features sur la prédiction (SHAP)",
        xaxis_title="Impact sur la probabilité de défaut",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


# ============================================
# Interface principale
# ============================================

def main():
    # En-tête
    st.title("🏦 Home Credit Scoring")
    st.markdown("""
    **Outil de scoring de crédit** basé sur un modèle de Machine Learning.
    
    Cette application permet d'évaluer le risque de défaut de paiement d'un client
    en fonction de ses caractéristiques.
    """)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # URL de l'API (hardcodée pour le déploiement ; non modifiable depuis l'UI)
        st.markdown(f"**API URL:** {API_URL}")

        # NOTE: Les vérifications de santé doivent être réalisées côté backend et dans la CI.
        # L'application affiche les informations du modèle si l'API répond.

        # Informations du modèle
        st.header("📊 Informations modèle")
        model_info = get_model_info()
        if model_info:
            st.json(model_info)
        else:
            st.info("Informations non disponibles via l'API. L'application utilisera le modèle local si présent.")
    
    # Contenu principal
    tab1, tab2, tab3 = st.tabs(["📝 Saisie manuelle", "📁 Import fichier", "📖 Documentation"])
    
    # ============================================
    # Tab 1: Saisie manuelle
    # ============================================
    with tab1:
        st.header("Saisie des caractéristiques client")
        
        col1, col2, col3 = st.columns(3)
        
        features = {}
        
        with col1:
            st.subheader("💰 Informations financières")
            features["AMT_INCOME_TOTAL"] = st.number_input(
                "Revenu annuel total (€)",
                min_value=0.0,
                max_value=10000000.0,
                value=150000.0,
                step=10000.0
            )
            features["AMT_CREDIT"] = st.number_input(
                "Montant du crédit (€)",
                min_value=0.0,
                max_value=5000000.0,
                value=500000.0,
                step=50000.0
            )
            features["AMT_ANNUITY"] = st.number_input(
                "Annuité (€/an)",
                min_value=0.0,
                max_value=500000.0,
                value=25000.0,
                step=1000.0
            )
            features["AMT_GOODS_PRICE"] = st.number_input(
                "Prix du bien (€)",
                min_value=0.0,
                max_value=5000000.0,
                value=450000.0,
                step=50000.0
            )
        
        with col2:
            st.subheader("👤 Informations personnelles")
            age = st.number_input("Âge (années)", min_value=18, max_value=120, value=35, step=1)
            features["DAYS_BIRTH"] = -int(age) * 365
            
            years_employed = st.slider("Années d'emploi", 0, 50, 5)
            features["DAYS_EMPLOYED"] = -years_employed * 365
            
            features["CNT_CHILDREN"] = st.number_input(
                "Nombre d'enfants",
                min_value=0,
                max_value=20,
                value=1
            )
            
            gender = st.selectbox("Genre", ["Homme", "Femme"])
            features["CODE_GENDER_M"] = 1 if gender == "Homme" else 0
            
            own_car = st.checkbox("Propriétaire d'un véhicule", value=True)
            features["FLAG_OWN_CAR"] = 1 if own_car else 0
            
            own_realty = st.checkbox("Propriétaire immobilier", value=True)
            features["FLAG_OWN_REALTY"] = 1 if own_realty else 0
        
        with col3:
            st.subheader("📊 Scores externes")
            features["EXT_SOURCE_1"] = st.slider(
                "Score externe 1",
                0.0, 1.0, 0.5, 0.01
            )
            features["EXT_SOURCE_2"] = st.slider(
                "Score externe 2",
                0.0, 1.0, 0.6, 0.01
            )
            features["EXT_SOURCE_3"] = st.slider(
                "Score externe 3",
                0.0, 1.0, 0.55, 0.01
            )
            
            features["REGION_RATING_CLIENT"] = st.selectbox(
                "Rating région client",
                [1, 2, 3],
                index=1
            )
        
        # Calcul des ratios
        if features["AMT_INCOME_TOTAL"] > 0:
            features["CREDIT_INCOME_RATIO"] = features["AMT_CREDIT"] / features["AMT_INCOME_TOTAL"]
            features["ANNUITY_INCOME_RATIO"] = features["AMT_ANNUITY"] / features["AMT_INCOME_TOTAL"]
        else:
            features["CREDIT_INCOME_RATIO"] = 0
            features["ANNUITY_INCOME_RATIO"] = 0
        
        features["EXT_SOURCE_MEAN"] = np.mean([
            features["EXT_SOURCE_1"],
            features["EXT_SOURCE_2"],
            features["EXT_SOURCE_3"]
        ])
        
        st.markdown("---")
        
        # Boutons d'action
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            predict_btn = st.button("🎯 Prédire", type="primary", use_container_width=True)
        
        with col_btn2:
            explain_btn = st.button("🔍 Expliquer", use_container_width=True)
        
        # Affichage des résultats
        if predict_btn:
            with st.spinner("Calcul en cours..."):
                result = predict(features)
            
            if result:
                st.markdown("---")
                st.header("📊 Résultats")
                
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    probability = result.get("probability", result.get("proba", 0.5))
                    threshold = result.get("threshold", 0.35)
                    
                    # Jauge de risque
                    fig_gauge = create_gauge_chart(probability, threshold)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_res2:
                    decision = result.get("decision", result.get("prediction", ""))
                    
                    if probability < threshold:
                        st.success(f"""
                        ### ✅ Crédit accordé
                        
                        **Probabilité de défaut**: {probability*100:.1f}%  
                        **Seuil de décision**: {threshold*100:.1f}%
                        
                        Le client présente un risque acceptable.
                        """)
                    else:
                        st.error(f"""
                        ### ❌ Crédit refusé
                        
                        **Probabilité de défaut**: {probability*100:.1f}%  
                        **Seuil de décision**: {threshold*100:.1f}%
                        
                        Le client présente un risque trop élevé.
                        """)
                    
                    # Métriques supplémentaires
                    st.metric("Probabilité de défaut", f"{probability*100:.2f}%")
                    st.metric("Écart au seuil", f"{(probability - threshold)*100:+.2f}%")
        
        if explain_btn:
            with st.spinner("Calcul des explications..."):
                explanation = explain(features)
            
            if explanation and "shap_values" in explanation:
                st.markdown("---")
                st.header("🔍 Explication de la prédiction")
                
                shap_values = explanation["shap_values"]
                base_value = explanation.get("base_value", 0.5)
                
                # Graphique SHAP
                fig_shap = create_shap_waterfall(shap_values, base_value)
                st.plotly_chart(fig_shap, use_container_width=True)
                
                # Interprétation
                st.subheader("📝 Interprétation")
                
                # Top 3 features positives (augmentent le risque)
                positive_features = [(k, v) for k, v in shap_values.items() if v > 0]
                positive_features.sort(key=lambda x: x[1], reverse=True)
                
                if positive_features:
                    st.markdown("**Facteurs augmentant le risque:**")
                    for feat, val in positive_features[:3]:
                        st.markdown(f"- {feat}: +{val:.3f}")
                
                # Top 3 features négatives (diminuent le risque)
                negative_features = [(k, v) for k, v in shap_values.items() if v < 0]
                negative_features.sort(key=lambda x: x[1])
                
                if negative_features:
                    st.markdown("**Facteurs diminuant le risque:**")
                    for feat, val in negative_features[:3]:
                        st.markdown(f"- {feat}: {val:.3f}")
    
    # ============================================
    # Tab 2: Import fichier
    # ============================================
    with tab2:
        st.header("Import de fichier")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV",
            type=["csv"],
            help="Format attendu: une ligne par client, colonnes = features"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"**{len(df)} clients chargés**")
                st.dataframe(df.head())
                
                if st.button("🎯 Prédire pour tous les clients"):
                    with st.spinner("Calcul des prédictions..."):
                        results = []
                        progress = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            features = row.to_dict()
                            result = predict(features)
                            if result:
                                results.append({
                                    "index": idx,
                                    "probability": result.get("probability", result.get("proba")),
                                    "decision": result.get("decision", result.get("prediction"))
                                })
                            progress.progress((idx + 1) / len(df))
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            
                            st.success(f"✅ {len(results)} prédictions effectuées")
                            st.dataframe(results_df)
                            
                            # Téléchargement
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Télécharger les résultats",
                                csv,
                                "predictions.csv",
                                "text/csv"
                            )
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {e}")
    
    # ============================================
    # Tab 3: Documentation
    # ============================================
    with tab3:
        st.header("📖 Documentation")
        
        st.markdown("""
        ## À propos
        
        Cette application permet d'évaluer le risque de défaut de paiement 
        pour des demandes de crédit, en utilisant un modèle de Machine Learning
        entraîné sur les données Home Credit.
        
        ## Méthodologie
        
        ### Modèle
        - **Algorithme**: LightGBM (Gradient Boosting)
        - **Métrique d'optimisation**: Coût métier (FN=10, FP=1)
        - **Seuil de décision**: Optimisé pour minimiser le coût métier
        
        ### Features importantes
        - **Scores externes** (EXT_SOURCE_1, 2, 3): Scores de bureaux de crédit externes
        - **Ratios financiers**: Crédit/Revenu, Annuité/Revenu
        - **Ancienneté**: Emploi, Âge, Documents
        
        ### Interprétabilité
        L'explication des prédictions utilise **SHAP** (SHapley Additive exPlanations),
        qui permet de comprendre l'impact de chaque variable sur la décision.
        
        ## API Endpoints
        
        | Endpoint | Méthode | Description |
        |----------|---------|-------------|
        | `/health` | GET | Vérification de l'état de l'API |
        | `/predict` | POST | Prédiction pour un client |
        | `/predict/batch` | POST | Prédictions en batch |
        | `/predict/explain` | POST | Prédiction avec explications SHAP |
        | `/model/info` | GET | Informations sur le modèle |
        | `/model/features` | GET | Liste des features attendues |
        
        ## Coût métier
        
        Le modèle optimise un coût métier asymétrique:
        - **Faux Négatif (FN)**: Coût = 10 (client défaillant accepté)
        - **Faux Positif (FP)**: Coût = 1 (bon client refusé)
        
        Cette asymétrie reflète le fait qu'accepter un client qui fera défaut
        est 10 fois plus coûteux que refuser un bon client.
        """)


if __name__ == "__main__":
    main()
