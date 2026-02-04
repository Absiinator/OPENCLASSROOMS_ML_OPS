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


def create_comparison_chart(client_value: float, ref_data: pd.Series, feature_name: str) -> go.Figure:
    """Crée un histogramme de comparaison client vs population."""
    # Filtrer uniquement les valeurs numériques valides
    ref_numeric = pd.to_numeric(ref_data, errors='coerce').dropna()
    
    if len(ref_numeric) == 0:
        return None
    
    fig = go.Figure()
    
    # Histogramme de la population
    fig.add_trace(go.Histogram(
        x=ref_numeric,
        name="Population",
        opacity=0.7,
        marker_color='#3498db'
    ))
    
    # Ligne verticale pour le client
    fig.add_vline(
        x=client_value,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Client: {client_value:.2f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"Position du client - {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Nombre de clients",
        height=350,
        showlegend=True
    )
    
    return fig


# ============================================
# Interface utilisateur principale
# ============================================

def render_sidebar():
    """Affiche la sidebar avec statut API et navigation."""
    with st.sidebar:
        st.title("🏦 Home Credit")
        st.markdown("---")
        
        # Statut de l'API
        api_ok = check_api_health()
        if api_ok:
            st.success("✅ API connectée")
            model_info = get_model_info()
            if model_info:
                st.info(f"📊 Seuil optimal: {model_info.get('optimal_threshold', 0.44)}")
        else:
            st.error("❌ API non disponible")
            st.info(f"URL: {API_URL}")

        # Statut MLflow
        mlflow_ok = check_mlflow_health(MLFLOW_URL)
        if mlflow_ok:
            st.success("✅ MLflow accessible")
        else:
            st.warning("⚠️ MLflow indisponible")
        if MLFLOW_URL:
            st.markdown(f"[Ouvrir MLflow UI]({MLFLOW_URL})")
        
        st.markdown("---")
        st.markdown("### ℹ️ À propos")
        st.markdown("""
        Cette application permet d'évaluer le risque de défaut 
        de paiement pour les demandes de crédit.
        
        **Seuil de décision**: 0.44 (optimisé coût métier)
        - Coût Faux Négatif: 10
        - Coût Faux Positif: 1
        """)


def render_prediction_tab():
    """Onglet principal: Scoring et prédiction."""
    st.header("🎯 Scoring de crédit")
    
    # Vérifier l'API
    if not check_api_health():
        st.error("⚠️ L'API n'est pas disponible. Veuillez réessayer plus tard.")
        return
    
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
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("🔮 Calculer le score", type="primary", use_container_width=True):
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
            
        else:
            error_msg = result.get("detail", "Erreur inconnue") if result else "Pas de réponse"
            st.error(f"❌ Erreur: {error_msg}")


def render_comparison_tab():
    """Onglet comparaison avec la population."""
    st.header("📊 Comparaison avec la population")
    
    # Charger données de référence
    ref_data = load_reference_data()
    
    if ref_data.empty:
        st.warning("Données de référence non disponibles.")
        return
    
    # Vérifier qu'on a des features à comparer
    if 'last_features' not in st.session_state:
        st.info("💡 Effectuez d'abord une prédiction dans l'onglet Scoring pour comparer le client.")
        return
    
    features = st.session_state.last_features
    
    # Sélectionner les features à comparer (uniquement numériques)
    numeric_cols = ref_data.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [f for f in features.keys() if f in numeric_cols]
    
    if not available_features:
        st.warning("Aucune feature comparable disponible.")
        return
    
    selected_feature = st.selectbox(
        "Sélectionnez une caractéristique à comparer",
        available_features,
        format_func=lambda x: f"{x} - {FEATURE_EXPLANATIONS.get(x, '')[:50]}..."
    )
    
    if selected_feature and selected_feature in ref_data.columns:
        client_val = features.get(selected_feature)
        ref_col = ref_data[selected_feature]
        
        # Créer le graphique de comparaison
        fig = create_comparison_chart(client_val, ref_col, selected_feature)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            ref_numeric = pd.to_numeric(ref_col, errors='coerce').dropna()
            if len(ref_numeric) > 0:
                percentile = (ref_numeric < client_val).mean() * 100
                
                st.markdown(f"""
                ### Statistiques
                - **Valeur client**: {client_val:.2f}
                - **Médiane population**: {ref_numeric.median():.2f}
                - **Percentile du client**: {percentile:.1f}%
                
                📌 *Le client se situe au {percentile:.0f}ème percentile, c'est-à-dire que {percentile:.0f}% de la population a une valeur inférieure.*
                """)
        else:
            st.warning("Impossible de créer le graphique pour cette caractéristique.")
    
    # Afficher l'explication de la feature
    if selected_feature:
        with st.expander(f"ℹ️ Signification de {selected_feature}"):
            st.markdown(FEATURE_EXPLANATIONS.get(selected_feature, "Pas de description disponible."))


def render_drift_tab():
    """Onglet rapport de Data Drift (Evidently)."""
    st.header("📈 Analyse du Data Drift")
    
    st.markdown("""
    Le Data Drift analyse la différence de distribution entre les données d'entraînement 
    et les nouvelles données en production. Un drift significatif peut indiquer que le 
    modèle doit être réentraîné.
    """)
    
    if os.path.exists(DRIFT_REPORT_PATH):
        try:
            with open(DRIFT_REPORT_PATH, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Erreur lors du chargement du rapport: {e}")
    else:
        st.warning("📋 Rapport Evidently non disponible.")
        st.info(f"Chemin attendu: {DRIFT_REPORT_PATH}")


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
    
    ### 📊 Onglet Comparaison
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


# ============================================
# Point d'entrée principal
# ============================================

def main():
    """Fonction principale de l'application."""
    
    # Sidebar
    render_sidebar()
    
    # Onglets principaux
    tabs = st.tabs([
        "🎯 Scoring",
        "📊 Comparaison",
        "📈 Data Drift",
        "📖 Documentation"
    ])
    
    with tabs[0]:
        render_prediction_tab()
    
    with tabs[1]:
        render_comparison_tab()
    
    with tabs[2]:
        render_drift_tab()
    
    with tabs[3]:
        render_documentation_tab()


if __name__ == "__main__":
    main()
