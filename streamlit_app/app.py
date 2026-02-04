"""
Dashboard Streamlit - Home Credit Scoring
=========================================

Interface utilisateur pour le scoring de crédit.
Version refactorisée avec modules séparés.

Fonctionnalités (selon exigences) :
- Visualiser le score, probabilité et interprétation intelligible
- Informations descriptives du client
- Comparaison avec l'ensemble des clients via graphiques
- Accessibilité WCAG (contrastes, labels, navigation)
- Rapport de Data Drift (Evidently)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

# Import des modules locaux
import api_client
import constants

check_api_health = api_client.check_api_health
get_model_info = api_client.get_model_info
predict_client = api_client.predict_client
explain_prediction = api_client.explain_prediction
get_feature_importance = api_client.get_feature_importance
API_URL = api_client.API_URL

from constants import (
    REQUIRED_FEATURES,
    FEATURE_EXPLANATIONS,
    MODEL_CONFIG,
    get_default_features,
    calculate_ratios,
    get_risk_category,
    get_risk_color
)

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
# CSS pour l'accessibilité WCAG
# ============================================
st.markdown("""
<style>
    /* Contrastes élevés pour l'accessibilité WCAG AA */
    .stMetric label { color: #1a1a1a !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0a0a0a !important; font-size: 1.5rem !important; }
    
    /* Focus visible pour navigation clavier */
    button:focus, input:focus, select:focus, a:focus {
        outline: 3px solid #005fcc !important;
        outline-offset: 2px !important;
    }
    
    /* Indicateurs visuels clairs avec patterns */
    .risk-low { background-color: #d4edda !important; color: #155724 !important; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #28a745; }
    .risk-high { background-color: #f8d7da !important; color: #721c24 !important; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# ============================================
# Variables globales
# ============================================
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# Chemins pour données et rapports
if os.path.exists("/app/data"):
    PROJECT_ROOT = "/app"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DRIFT_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "evidently_full_report.html")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "application_train.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "reports", "feature_importance.csv")


# ============================================
# Fonctions utilitaires
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    """Charge les données de référence pour comparaison (cache 1h)."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, nrows=10000)
            return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_top_features_from_report(top_n: int = 15) -> List[str]:
    """Charge les features les plus importantes depuis le rapport (notebooks)."""
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        try:
            df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            if "feature" in df.columns and "importance" in df.columns:
                df = df.sort_values("importance", ascending=False)
                return df["feature"].head(top_n).tolist()
        except Exception:
            pass
    return []


def interpret_score(probability: float, threshold: float) -> dict:
    """Génère une interprétation textuelle pour non-experts."""
    if probability < threshold:
        decision = "ACCORDÉ"
        if probability < threshold * 0.5:
            confidence = "très élevée"
            explanation = "Profil très favorable. Risque de défaut minimal."
        else:
            confidence = "modérée"
            explanation = "Profil acceptable avec quelques points de vigilance."
    else:
        decision = "REFUSÉ"
        if probability > threshold * 1.5:
            confidence = "très élevée"
            explanation = "Profil à risque significatif. Défaut de paiement probable."
        else:
            confidence = "modérée"
            explanation = "Profil légèrement au-dessus du seuil de risque."
    
    return {
        "decision": decision,
        "confidence": confidence,
        "explanation": explanation,
        "probability_text": f"{probability*100:.1f}%",
        "threshold_text": f"{threshold*100:.1f}%"
    }


def get_feature_explanation(feature_name: str) -> str:
    """Retourne une explication en langage naturel d'une feature."""
    return FEATURE_EXPLANATIONS.get(feature_name, f"{feature_name}")


def check_mlflow_health(url: str) -> bool:
    """Vérifie si l'UI MLflow est accessible."""
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def create_gauge_chart(probability: float, threshold: float) -> go.Figure:
    """Crée une jauge visuelle du score de risque."""
    color = get_risk_color(probability)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 40}},
        delta={'reference': threshold * 100, 'position': "bottom", 'relative': False},
        title={'text': "Probabilité de défaut", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},
                {'range': [20, 40], 'color': '#fff3cd'},
                {'range': [40, 60], 'color': '#ffeaa7'},
                {'range': [60, 80], 'color': '#f8d7da'},
                {'range': [80, 100], 'color': '#f5c6cb'}
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
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "#1a1a1a"}
    )
    return fig


def create_comparison_chart(
    client_value: float,
    feature_name: str,
    reference_data: pd.DataFrame,
    group_filter: Optional[str] = None
) -> Optional[go.Figure]:
    """Crée un histogramme de comparaison client vs population."""
    if feature_name not in reference_data.columns:
        return None

    data = reference_data[feature_name].dropna()

    if group_filter and group_filter != "Tous les clients":
        if "TARGET" in reference_data.columns:
            if group_filter == "Clients sans défaut (TARGET=0)":
                data = reference_data[reference_data["TARGET"] == 0][feature_name].dropna()
            elif group_filter == "Clients en défaut (TARGET=1)":
                data = reference_data[reference_data["TARGET"] == 1][feature_name].dropna()

    if data.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        name="Distribution",
        opacity=0.7,
        marker_color="#4169E1"
    ))

    fig.add_vline(
        x=client_value,
        line_dash="dash",
        line_color="#C41E3A",
        line_width=3,
        annotation_text=f"Client: {client_value:.2f}",
        annotation_position="top"
    )

    percentile = (data < client_value).mean() * 100
    fig.update_layout(
        title=f"Distribution de {feature_name} (client au {percentile:.0f}e percentile)",
        xaxis_title=feature_name,
        yaxis_title="Nombre de clients",
        height=400,
        showlegend=False
    )

    return fig


def create_radar_comparison(
    client_features: Dict[str, float],
    reference_data: pd.DataFrame,
    selected_features: List[str]
) -> Optional[go.Figure]:
    """Crée un graphique radar pour comparer plusieurs features."""
    normalized_client = []
    normalized_mean = []

    for feat in selected_features:
        if feat in reference_data.columns and feat in client_features:
            ref_data = reference_data[feat].dropna()
            if ref_data.empty:
                continue
            min_val, max_val = ref_data.min(), ref_data.max()
            if max_val > min_val:
                client_norm = (client_features[feat] - min_val) / (max_val - min_val)
                mean_norm = (ref_data.mean() - min_val) / (max_val - min_val)
            else:
                client_norm, mean_norm = 0.5, 0.5
            normalized_client.append(client_norm)
            normalized_mean.append(mean_norm)

    if not normalized_client:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_mean + [normalized_mean[0]],
        theta=selected_features + [selected_features[0]],
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.3)',
        line_color='#4169E1',
        name='Moyenne population'
    ))
    fig.add_trace(go.Scatterpolar(
        r=normalized_client + [normalized_client[0]],
        theta=selected_features + [selected_features[0]],
        fill='toself',
        fillcolor='rgba(196, 30, 58, 0.3)',
        line_color='#C41E3A',
        name='Client actuel'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title={'text': "Comparaison multi-critères", 'font': {'size': 16}},
        height=500,
        paper_bgcolor='white'
    )

    return fig


# ============================================
# Interface utilisateur principale
# ============================================

def render_sidebar(reference_data: pd.DataFrame):
    """Affiche la sidebar avec navigation, docs et statuts."""
    with st.sidebar:
        st.title("🏦 Home Credit")
        st.divider()

        # Navigation
        st.header("📍 Navigation")
        nav_options = [
            ("🎯 Scoring", "scoring"),
            ("📈 Data Drift", "drift"),
            ("📖 Documentation", "docs")
        ]
        if "current_page" not in st.session_state:
            st.session_state.current_page = "scoring"
        for label, page_key in nav_options:
            btn_type = "primary" if st.session_state.current_page == page_key else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state.current_page = page_key
                st.rerun()

        st.divider()

        # Documentation
        st.header("🔗 Documentation")
        with st.expander("🔍 URLs configurées", expanded=False):
            st.code(f"API_URL={API_URL}")
            st.code(f"MLFLOW_URL={MLFLOW_URL}")
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("Swagger", f"{API_URL}/docs", use_container_width=True)
        with col2:
            st.link_button("ReDoc", f"{API_URL}/redoc", use_container_width=True)
        st.link_button("MLflow UI", MLFLOW_URL, use_container_width=True)

        st.divider()

        # État des services
        st.header("🏥 État")
        api_ok = check_api_health()
        mlflow_ok = check_mlflow_health(MLFLOW_URL)
        drift_exists = os.path.exists(DRIFT_REPORT_PATH)
        st.write(f"{'✅' if api_ok else '⚠️'} API: {'OK' if api_ok else 'Hors ligne'}")
        st.write(f"{'✅' if mlflow_ok else '⚠️'} MLflow: {'OK' if mlflow_ok else 'Hors ligne'}")
        st.write(f"{'✅' if drift_exists else '⚠️'} Drift: {'OK' if drift_exists else 'Absent'}")

        st.divider()

        # Modèle
        st.header("🤖 Modèle")
        model_info = get_model_info()
        if model_info:
            st.metric("Seuil", f"{model_info.get('optimal_threshold', 0.44):.2%}")
        else:
            st.caption("Infos indisponibles")

        st.divider()

        # Dataset
        st.header("📊 Dataset")
        if reference_data is not None and not reference_data.empty:
            st.metric("Clients", f"{len(reference_data):,}")
            if "TARGET" in reference_data.columns:
                st.metric("Taux défaut", f"{reference_data['TARGET'].mean():.1%}")
            with st.expander("💰 Finances"):
                if "AMT_INCOME_TOTAL" in reference_data.columns:
                    st.write(f"Revenu: {reference_data['AMT_INCOME_TOTAL'].median():,.0f}€")
                if "AMT_CREDIT" in reference_data.columns:
                    st.write(f"Crédit: {reference_data['AMT_CREDIT'].median():,.0f}€")
            with st.expander("📊 Scores"):
                for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                    if col in reference_data.columns:
                        st.write(f"{col}: {reference_data[col].median():.3f}")
        else:
            st.warning("📂 Données manquantes")

        st.divider()
        st.caption("v1.0.0 • Home Credit Scoring")


def render_prediction_tab():
    """Onglet principal: Scoring et prédiction."""
    st.header("🎯 Scoring de crédit")
    
    # Vérifier l'API (ne bloque pas la comparaison)
    api_ok = check_api_health()
    if not api_ok:
        st.warning("⚠️ API indisponible. La comparaison reste possible, la prédiction est désactivée.")
        st.info(f"API_URL: {API_URL}")

    ref_data = load_reference_data()
    
    st.markdown("### Saisie des informations client")
    
    # Initialiser les features avec les valeurs par défaut
    if 'client_features' not in st.session_state:
        st.session_state.client_features = get_default_features()
    
    # Formulaire de saisie en colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Finances")
        st.session_state.client_features["AMT_INCOME_TOTAL"] = st.number_input(
            "Revenu annuel (€)",
            min_value=0.0, max_value=10000000.0,
            value=float(st.session_state.client_features.get("AMT_INCOME_TOTAL", 150000)),
            step=10000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_INCOME_TOTAL", "")
        )
        st.session_state.client_features["AMT_CREDIT"] = st.number_input(
            "Montant crédit (€)",
            min_value=0.0, max_value=5000000.0,
            value=float(st.session_state.client_features.get("AMT_CREDIT", 500000)),
            step=10000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_CREDIT", "")
        )
        st.session_state.client_features["AMT_ANNUITY"] = st.number_input(
            "Annuité (€/an)",
            min_value=0.0, max_value=500000.0,
            value=float(st.session_state.client_features.get("AMT_ANNUITY", 25000)),
            step=1000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_ANNUITY", "")
        )
        st.session_state.client_features["AMT_GOODS_PRICE"] = st.number_input(
            "Prix du bien (€)",
            min_value=0.0, max_value=5000000.0,
            value=float(st.session_state.client_features.get("AMT_GOODS_PRICE", 450000)),
            step=10000.0,
            help=FEATURE_EXPLANATIONS.get("AMT_GOODS_PRICE", "")
        )
    
    with col2:
        st.markdown("#### 👤 Personnel")
        age_years = st.slider(
            "Âge (années)",
            min_value=18, max_value=80,
            value=35,
            help="Converti en jours pour le modèle"
        )
        st.session_state.client_features["DAYS_BIRTH"] = -age_years * 365
        
        exp_years = st.slider(
            "Ancienneté emploi (années)",
            min_value=0, max_value=50,
            value=5,
            help="Converti en jours pour le modèle"
        )
        st.session_state.client_features["DAYS_EMPLOYED"] = -exp_years * 365
        
        st.session_state.client_features["CNT_CHILDREN"] = st.number_input(
            "Nombre d'enfants",
            min_value=0, max_value=20,
            value=int(st.session_state.client_features.get("CNT_CHILDREN", 1))
        )
        
        gender = st.selectbox("Genre", ["Homme", "Femme"])
        st.session_state.client_features["CODE_GENDER_M"] = 1 if gender == "Homme" else 0
        
        st.session_state.client_features["FLAG_OWN_CAR"] = 1 if st.checkbox("Possède une voiture", value=True) else 0
        st.session_state.client_features["FLAG_OWN_REALTY"] = 1 if st.checkbox("Propriétaire immobilier", value=True) else 0
    
    with col3:
        st.markdown("#### 📊 Scores externes")
        st.session_state.client_features["EXT_SOURCE_1"] = st.slider(
            "Score externe 1",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.client_features.get("EXT_SOURCE_1", 0.5)),
            step=0.01,
            help=FEATURE_EXPLANATIONS.get("EXT_SOURCE_1", "")
        )
        st.session_state.client_features["EXT_SOURCE_2"] = st.slider(
            "Score externe 2",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.client_features.get("EXT_SOURCE_2", 0.6)),
            step=0.01,
            help=FEATURE_EXPLANATIONS.get("EXT_SOURCE_2", "")
        )
        st.session_state.client_features["EXT_SOURCE_3"] = st.slider(
            "Score externe 3",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.client_features.get("EXT_SOURCE_3", 0.55)),
            step=0.01,
            help=FEATURE_EXPLANATIONS.get("EXT_SOURCE_3", "")
        )
        st.session_state.client_features["REGION_RATING_CLIENT"] = st.selectbox(
            "Note région",
            options=[1, 2, 3],
            index=1,
            help="1=faible risque, 3=risque élevé"
        )
    
    # Calculer les ratios automatiquement
    features = calculate_ratios(st.session_state.client_features)
    st.session_state.current_features = features
    
    st.markdown("---")

    # Comparaison population (sans prédiction)
    with st.expander("📊 Comparaison avec la population (sans prédiction)", expanded=False):
        render_comparison_section(features, ref_data, show_header=False)

    st.markdown("---")

    # Bouton de prédiction
    if st.button("🔮 Calculer le score", type="primary", use_container_width=True, disabled=not api_ok):
        with st.spinner("Calcul en cours..."):
            result = predict_client(features)
        
        if result and "error" not in result:
            # Afficher les résultats
            probability = result.get("probability", 0)
            threshold = result.get("threshold", 0.44)
            decision = result.get("decision", "")
            
            # Interprétation
            interpretation = interpret_score(probability, threshold)
            
            # Layout résultats
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                # Jauge de risque
                fig_gauge = create_gauge_chart(probability, threshold)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with res_col2:
                # Décision
                if decision == "ACCEPTÉ":
                    st.markdown(f'<div class="risk-low"><h2>✅ {decision}</h2></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-high"><h2>❌ {decision}</h2></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                ### Interprétation
                
                - **Probabilité de défaut**: {interpretation['probability_text']}
                - **Seuil de décision**: {interpretation['threshold_text']}
                - **Confiance**: {interpretation['confidence']}
                
                {interpretation['explanation']}
                """)
            
            # Stocker pour comparaison
            st.session_state.last_prediction = result
            st.session_state.last_features = features

            # Résumé descriptif du client
            st.subheader("👤 Résumé du profil client")
            profile_col1, profile_col2, profile_col3 = st.columns(3)
            with profile_col1:
                st.markdown("**Situation financière**")
                st.write(f"- Revenu: {features['AMT_INCOME_TOTAL']:,.0f} €")
                st.write(f"- Crédit demandé: {features['AMT_CREDIT']:,.0f} €")
                st.write(f"- Ratio crédit/revenu: {features['CREDIT_INCOME_RATIO']:.2f}")
            with profile_col2:
                st.markdown("**Situation personnelle**")
                age_years = abs(int(features['DAYS_BIRTH'])) // 365
                employed_years = abs(int(features['DAYS_EMPLOYED'])) // 365
                st.write(f"- Âge: {age_years} ans")
                st.write(f"- Ancienneté emploi: {employed_years} ans")
                st.write(f"- Enfants: {features['CNT_CHILDREN']}")
            with profile_col3:
                st.markdown("**Scores de crédit**")
                st.write(f"- Score moyen: {features['EXT_SOURCE_MEAN']:.2f}")
                st.write(f"- Propriétaire: {'Oui' if features['FLAG_OWN_REALTY'] else 'Non'}")
                st.write(f"- Véhicule: {'Oui' if features['FLAG_OWN_CAR'] else 'Non'}")
            
        else:
            status_code = result.get("status_code") if result else None
            detail = result.get("detail", "Erreur inconnue") if result else "Pas de réponse"
            if isinstance(detail, dict):
                detail = detail.get("detail", detail)
            detail = str(detail)
            if status_code == 404:
                st.error("❌ Endpoint API introuvable (404).")
                st.info(f"API_URL: {API_URL}")
                st.info(f"Endpoint attendu: {API_URL}/predict")
            else:
                st.error(f"❌ Erreur API ({status_code}): {detail}")


def render_comparison_section(
    features: Dict[str, float],
    ref_data: pd.DataFrame,
    show_header: bool = True
):
    """Section comparaison avec la population."""
    if show_header:
        st.header("📊 Comparaison avec la population")
    if ref_data.empty:
        st.warning("⚠️ Données de référence non disponibles.")
        return

    st.markdown("""
    **Comparez les caractéristiques du client avec l'ensemble de la population ou un groupe de clients similaires.**
    """)

    group_filter = st.selectbox(
        "🎯 Groupe de comparaison",
        ["Tous les clients", "Clients sans défaut (TARGET=0)", "Clients en défaut (TARGET=1)"],
        help="Sélectionnez le groupe avec lequel comparer le client"
    )

    numeric_cols = ref_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["SK_ID_CURR", "TARGET", "index"]
    available_features = [f for f in numeric_cols if f not in exclude_cols and f in features]

    if not available_features:
        st.warning("Aucune feature comparable disponible.")
        return

    explained_features = list(FEATURE_EXPLANATIONS.keys())
    priority_features = [f for f in explained_features if f in available_features]
    other_features = [f for f in available_features if f not in priority_features]
    available_features = priority_features + sorted(other_features)

    tab1, tab2, tab3 = st.tabs([
        "🎯 Vue Radar",
        "📈 Comparaison détaillée",
        "📋 Statistiques complètes"
    ])

    with tab1:
        st.subheader("🎯 Comparaison multi-critères")
        # Utiliser les features les plus importantes issues du notebook (reports/feature_importance.csv)
        top_from_report = load_top_features_from_report(15)
        default_radar = [f for f in top_from_report if f in available_features][:6]
        if len(default_radar) < 3:
            default_radar = [f for f in [
                "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1",
                "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY",
                "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO"
            ] if f in available_features][:6]

        radar_features = st.multiselect(
            "Caractéristiques à comparer (3-8 recommandé)",
            available_features,
            default=default_radar,
            help="Choisissez jusqu'à 8 caractéristiques pour le radar",
            format_func=lambda x: f"{x} - {get_feature_explanation(x)[:40]}..."
        )

        if radar_features and len(radar_features) >= 3:
            fig_radar = create_radar_comparison(features, ref_data, radar_features)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("Impossible de créer le radar.")
        else:
            st.info("Sélectionnez au moins 3 caractéristiques.")

    with tab2:
        st.subheader("📈 Comparaison détaillée par caractéristique")
        selected_feature = st.selectbox(
            "Sélectionnez une caractéristique",
            available_features,
            help="Voir la distribution et la position du client",
            format_func=lambda x: f"{x}"
        )

        st.info(f"**{selected_feature}**: {get_feature_explanation(selected_feature)}")
        client_value = features.get(selected_feature)
        if client_value is None:
            st.warning("Valeur client indisponible pour cette caractéristique.")
        else:
            fig = create_comparison_chart(client_value, selected_feature, ref_data, group_filter)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                ref_col = ref_data[selected_feature].dropna()
                percentile = (ref_col < client_value).mean() * 100 if len(ref_col) > 0 else 0
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Valeur client", f"{client_value:,.2f}")
                with stat_col2:
                    st.metric("Moyenne population", f"{ref_col.mean():,.2f}")
                with stat_col3:
                    st.metric("Percentile", f"{percentile:.0f}%")

                if percentile < 25:
                    st.warning("⚠️ Client dans les 25% les plus bas.")
                elif percentile > 75:
                    st.success("✅ Client dans les 25% les plus hauts.")
                else:
                    st.info("ℹ️ Client dans la moyenne.")
            else:
                st.warning("Impossible de créer le graphique.")

    with tab3:
        st.subheader("📋 Statistiques complètes du client")
        comparison_data = []
        for feat in available_features:
            if feat in ref_data.columns:
                client_val = features.get(feat)
                if client_val is None:
                    continue
                ref_col = ref_data[feat].dropna()
                if len(ref_col) == 0:
                    continue
                percentile = (ref_col < client_val).mean() * 100
                comparison_data.append({
                    "Caractéristique": feat,
                    "Valeur client": f"{client_val:,.2f}",
                    "Moyenne pop.": f"{ref_col.mean():,.2f}",
                    "Médiane pop.": f"{ref_col.median():,.2f}",
                    "Percentile": f"{percentile:.0f}%",
                })

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune statistique disponible.")


def render_drift_tab():
    """Onglet rapport de Data Drift (Evidently)."""
    st.header("📈 Analyse du Data Drift")

    st.markdown("""
    ## Rapport Evidently
    
    Le rapport de data drift permet de détecter les dérives entre:
    - **Données d'entraînement** (référence)
    - **Données de production** (nouvelles données)
    
    **Métriques surveillées**:
    - Distribution des features
    - Valeurs manquantes
    - Corrélations
    - Tests statistiques (Kolmogorov-Smirnov, Chi²)
    """)

    if os.path.exists(DRIFT_REPORT_PATH):
        try:
            with open(DRIFT_REPORT_PATH, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.markdown("### Rapport complet Evidently")
            st.components.v1.html(html_content, height=1200, scrolling=True)
        except Exception as e:
            st.error(f"Erreur lors du chargement du rapport: {e}")
    else:
        st.warning("📋 Rapport Evidently non disponible.")
        st.info("Le rapport est généré par le notebook `notebooks/04_Drift_Evidently.ipynb`.")


def render_documentation_tab():
    """Onglet documentation."""
    st.header("📖 Documentation")
    
    st.markdown("""
    ## Guide d'utilisation
    
    ### 🎯 Onglet Scoring
    Saisissez les caractéristiques du client pour obtenir:
    - La probabilité de défaut de paiement
    - La décision (Accordé/Refusé)
    - Une interprétation pour non-experts
    
    ### 📊 Comparaison (dans l'onglet Scoring)
    Comparez le profil du client avec l'ensemble de la population:
    - Visualisation de la distribution
    - Position du client (percentile)
    
    ### 📈 Onglet Data Drift
    Rapport Evidently analysant la stabilité des données.
    
    ---
    
    ## Features importantes
    
    | Feature | Description | Impact |
    |---------|-------------|--------|
    | EXT_SOURCE_1/2/3 | Scores de crédit externes | Plus élevé = meilleur |
    | CREDIT_INCOME_RATIO | Ratio crédit/revenu | Plus bas = meilleur |
    | DAYS_BIRTH | Âge (jours négatifs) | Plus âgé = plus stable |
    | AMT_CREDIT | Montant du crédit | - |
    
    ---
    
    ## Seuil de décision
    
    **Seuil optimal: 0.44** (issu de l'optimisation du coût métier)
    
    - Coût Faux Négatif (FN): 10 (mauvais client accepté)
    - Coût Faux Positif (FP): 1 (bon client refusé)
    
    Le modèle minimise: `10 × FN + 1 × FP`
    """)

    st.markdown("### Liens utiles")
    st.markdown(f"- API Swagger: {API_URL}/docs")
    st.markdown(f"- API ReDoc: {API_URL}/redoc")
    st.markdown(f"- MLflow UI: {MLFLOW_URL}")


# ============================================
# Point d'entrée principal
# ============================================

def main():
    """Fonction principale de l'application."""
    st.title("🏦 Home Credit - Outil de Scoring")
    st.markdown("""
    **Outil d'aide à la décision pour l'octroi de crédit**
    
    Cette application évalue le risque de défaut de paiement et fournit une interprétation
    claire du score pour chaque demande de crédit.
    """)

    reference_data = load_reference_data()
    render_sidebar(reference_data)

    current_page = st.session_state.get("current_page", "scoring")

    if current_page == "scoring":
        render_prediction_tab()
    elif current_page == "drift":
        render_drift_tab()
    elif current_page == "docs":
        render_documentation_tab()
    else:
        st.session_state.current_page = "scoring"
        render_prediction_tab()


if __name__ == "__main__":
    main()
