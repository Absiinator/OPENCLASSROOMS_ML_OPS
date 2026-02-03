"""
Application Streamlit pour le scoring crédit Home Credit.
=========================================================

Interface utilisateur complète pour le scoring de crédit.
Fonctionnalités :
- Visualiser le score et la probabilité avec interprétation intelligible
- Informations descriptives du client
- Comparaison avec l'ensemble des clients ou groupes similaires
- Accessibilité WCAG (contrastes, labels, navigation clavier)
- Modification en temps réel des caractéristiques
- Rapport de Data Drift
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
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
# CSS pour l'accessibilité WCAG
# ============================================
st.markdown("""
<style>
    /* Contrastes élevés pour l'accessibilité WCAG AA */
    .stMetric label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0a0a0a !important;
        font-size: 1.5rem !important;
    }
    
    /* Focus visible pour navigation clavier */
    button:focus, input:focus, select:focus, a:focus {
        outline: 3px solid #005fcc !important;
        outline-offset: 2px !important;
    }
    
    /* Indicateurs visuels clairs avec patterns */
    .risk-low {
        background-color: #d4edda !important;
        color: #155724 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .risk-high {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Variables globales
# ============================================
# Endpoints configurés via variables d'environnement (déploiement Render)
# Ces variables DOIVENT être configurées dans Render → Environment
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# === LOGS DEBUG AU DÉMARRAGE ===
print("=" * 60)
print("[STREAMLIT] CONFIGURATION DES VARIABLES D'ENVIRONNEMENT")
print("=" * 60)
print(f"[STREAMLIT] API_URL = {API_URL}")
print(f"[STREAMLIT] MLFLOW_URL = {MLFLOW_URL}")
print(f"[STREAMLIT] STREAMLIT_SERVER_ADDRESS = {os.getenv('STREAMLIT_SERVER_ADDRESS', 'non défini')}")
print(f"[STREAMLIT] STREAMLIT_SERVER_PORT = {os.getenv('STREAMLIT_SERVER_PORT', 'non défini')}")
print(f"[STREAMLIT] PORT = {os.getenv('PORT', 'non défini')}")
print("=" * 60)

# Chemins pour données et rapports (téléchargés dans Docker)
if os.path.exists("/app/data"):
    PROJECT_ROOT = "/app"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DRIFT_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "evidently_full_report.html")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "application_train.csv")

# ============================================
# Fonctions utilitaires
# ============================================

@st.cache_data(ttl=30, show_spinner=False)
def check_api_health() -> bool:
    """Vérifie si l'API est accessible (cache 30s)."""
    try:
        # Timeout augmenté pour cold start Render (services gratuits)
        response = requests.get(f"{API_URL}/health", timeout=15)
        return response.status_code == 200
    except:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def get_model_info() -> Optional[Dict[str, Any]]:
    """Récupère les informations du modèle (cache 5 min)."""
    try:
        # Timeout augmenté pour cold start Render
        response = requests.get(f"{API_URL}/model/info", timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_model_features() -> Optional[list]:
    """Récupère la liste des noms des features du modèle."""
    try:
        # Timeout augmenté pour cold start Render
        response = requests.get(f"{API_URL}/model/feature-names", timeout=30)
        if response.status_code == 200:
            return response.json().get("features", [])
    except:
        pass
    return None


def predict(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Effectue une prédiction via l'API (pas de fallback local)."""
    try:
        # Timeout augmenté pour Render free tier (cold start ~30-60s)
        print(f"[STREAMLIT] Envoi requête POST {API_URL}/predict")
        print(f"[STREAMLIT] Payload: features avec {len(features)} champs")
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=90,  # 90s pour cold start Render
            headers={"Content-Type": "application/json"}
        )
        print(f"[STREAMLIT] Réponse: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            st.error(f"🔴 Erreur API ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🔴 API non accessible. Vérifiez que l'API est déployée et accessible.")
        st.info(f"💡 URL configurée: {API_URL}")
        return None
    except requests.exceptions.Timeout:
        st.error("🔴 Timeout de l'API. Le serveur met trop de temps à répondre.")
        return None
    except Exception as e:
        st.error(f"🔴 Erreur de connexion à l'API: {str(e)}")
        return None


def explain(features: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Obtient l'explication SHAP via l'API (pas de fallback local)."""
    try:
        # Timeout long pour SHAP (calcul intensif + cold start)
        print(f"[STREAMLIT] Envoi requête POST {API_URL}/predict/explain")
        response = requests.post(
            f"{API_URL}/predict/explain",
            json={"features": features},
            timeout=120,  # 120s pour SHAP + cold start Render
            headers={"Content-Type": "application/json"}
        )
        print(f"[STREAMLIT] Réponse explain: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            st.error(f"🔴 Erreur API explications ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🔴 API non accessible pour les explications.")
        return None
    except requests.exceptions.Timeout:
        st.error("🔴 Timeout lors du calcul des explications SHAP.")
        return None
    except Exception as e:
        st.error(f"🔴 Erreur d'explication: {str(e)}")
        return None


# ============================================
# Fonctions pour données de référence et comparaison
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_reference_data() -> Optional[pd.DataFrame]:
    """Charge les données de référence pour les comparaisons (cache 1h)."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, nrows=10000)  # Limiter pour performance
            return df
        except Exception:
            pass
    return None


def interpret_score(probability: float, threshold: float) -> Dict[str, str]:
    """Génère une interprétation textuelle du score pour non-experts."""
    distance_to_threshold = abs(probability - threshold)
    
    if probability < threshold:
        decision = "ACCORDÉ"
        if probability < threshold * 0.3:
            confidence = "très élevée"
            explanation = "Le profil du client présente des caractéristiques très favorables. Le risque de défaut est minimal."
        elif probability < threshold * 0.6:
            confidence = "élevée"
            explanation = "Le profil du client est globalement positif. Le risque de défaut est faible."
        else:
            confidence = "modérée"
            explanation = "Le profil du client est acceptable mais présente quelques points de vigilance."
    else:
        decision = "REFUSÉ"
        if probability > threshold * 1.5:
            confidence = "très élevée"
            explanation = "Le profil du client présente des risques significatifs. Le défaut de paiement est probable."
        elif probability > threshold * 1.2:
            confidence = "élevée"
            explanation = "Le profil du client présente plusieurs facteurs de risque importants."
        else:
            confidence = "modérée"
            explanation = "Le profil du client est légèrement au-dessus du seuil de risque acceptable."
    
    return {
        "decision": decision,
        "confidence": confidence,
        "explanation": explanation,
        "probability_text": f"{probability*100:.1f}%",
        "threshold_text": f"{threshold*100:.1f}%",
        "distance_text": f"{distance_to_threshold*100:.1f} points"
    }


def get_feature_explanation(feature_name: str) -> str:
    """Retourne une explication en langage naturel d'une feature."""
    explanations = {
        "AMT_INCOME_TOTAL": "Revenu annuel total du client en euros",
        "AMT_CREDIT": "Montant total du crédit demandé",
        "AMT_ANNUITY": "Montant de l'annuité (paiement périodique)",
        "EXT_SOURCE_1": "Score de crédit externe (source 1) - Plus élevé = meilleur profil",
        "EXT_SOURCE_2": "Score de crédit externe (source 2) - Plus élevé = meilleur profil",
        "EXT_SOURCE_3": "Score de crédit externe (source 3) - Plus élevé = meilleur profil",
        "CREDIT_INCOME_RATIO": "Ratio crédit/revenu - Plus bas = meilleure capacité",
        "DAYS_BIRTH": "Âge du client (en jours depuis la naissance)",
        "DAYS_EMPLOYED": "Ancienneté dans l'emploi actuel",
    }
    return explanations.get(feature_name, f"Caractéristique: {feature_name}")


def create_comparison_chart(
    client_value: float,
    feature_name: str,
    reference_data: pd.DataFrame,
    group_filter: Optional[str] = None
) -> Optional[go.Figure]:
    """Crée un graphique de comparaison accessible."""
    if feature_name not in reference_data.columns:
        return None
    
    data = reference_data[feature_name].dropna()
    
    # Appliquer le filtre de groupe
    if group_filter and group_filter != "Tous les clients":
        if "TARGET" in reference_data.columns:
            if group_filter == "Clients sans défaut (TARGET=0)":
                data = reference_data[reference_data["TARGET"] == 0][feature_name].dropna()
            elif group_filter == "Clients en défaut (TARGET=1)":
                data = reference_data[reference_data["TARGET"] == 1][feature_name].dropna()
    
    fig = go.Figure()
    
    # Histogramme avec couleur accessible
    fig.add_trace(go.Histogram(
        x=data,
        name="Distribution",
        marker_color='#4169E1',  # Bleu royal - bon contraste
        opacity=0.7,
        hovertemplate="Valeur: %{x}<br>Nombre: %{y}<extra></extra>"
    ))
    
    # Ligne verticale pour le client
    fig.add_vline(
        x=client_value,
        line_width=4,
        line_dash="dash",
        line_color="#C41E3A",  # Rouge cardinal
        annotation_text=f"Client: {client_value:.2f}",
        annotation_position="top",
        annotation_font_size=14,
        annotation_font_color="#C41E3A"
    )
    
    # Position du client (percentile)
    percentile = (data < client_value).mean() * 100
    
    fig.update_layout(
        title={
            'text': f"<b>Distribution de {feature_name}</b><br><sup>Client au {percentile:.0f}e percentile</sup>",
            'font': {'size': 16, 'color': '#1a1a1a'}
        },
        xaxis_title=feature_name,
        yaxis_title="Nombre de clients",
        height=400,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Arial, sans-serif', 'color': '#1a1a1a'}
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
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
    
    # Population moyenne
    fig.add_trace(go.Scatterpolar(
        r=normalized_mean + [normalized_mean[0]],
        theta=selected_features + [selected_features[0]],
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.3)',
        line_color='#4169E1',
        name='Moyenne population'
    ))
    
    # Client
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
        title={'text': "<b>Comparaison multi-critères</b>", 'font': {'size': 16}},
        height=500,
        paper_bgcolor='white'
    )
    
    return fig


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
    st.title("🏦 Home Credit - Outil de Scoring")
    st.markdown("""
    **Outil d'aide à la décision pour l'octroi de crédit**
    
    Cette application évalue le risque de défaut de paiement et fournit une interprétation 
    claire du score pour chaque demande de crédit.
    """)
    
    # Charger les données de référence pour comparaison
    reference_data = load_reference_data()
    
    # Sidebar - Navigation et Configuration
    with st.sidebar:
        st.title("🏦 Home Credit")
        
        st.divider()
        
        # Section Navigation principale
        st.header("📍 Navigation")
        
        # Boutons de navigation
        nav_options = [
            ("🎯 Scoring Client", "scoring"),
            ("📊 Comparaison", "comparison"),
            ("📁 Import / Simulation", "simulation"),
            ("📈 Data Drift", "drift"),
            ("📖 Documentation", "docs")
        ]
        
        # Initialiser la page dans session_state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "scoring"
        
        for label, page_key in nav_options:
            btn_type = "primary" if st.session_state.current_page == page_key else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.divider()
        
        # Liens services externes
        st.header("🔗 Services")
        
        # Debug : afficher les URLs configurées
        with st.expander("🔍 URLs configurées", expanded=False):
            st.code(f"API_URL={API_URL}")
            st.code(f"MLFLOW_URL={MLFLOW_URL}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("📊 MLflow", MLFLOW_URL, use_container_width=True)
        with col2:
            st.link_button("🌐 API", f"{API_URL}/docs", use_container_width=True)
        
        st.divider()
        
        # Section État des Services (compact)
        st.header("🏥 État")
        
        api_healthy = check_api_health()
        drift_exists = os.path.exists(DRIFT_REPORT_PATH)
        
        st.write(f"{'✅' if api_healthy else '⚠️'} API: {'OK' if api_healthy else 'Hors ligne'}")
        st.write(f"{'✅' if drift_exists else '⚠️'} Drift: {'OK' if drift_exists else 'Absent'}")
        
        st.divider()
        
        # Section Modèle ML (compact)
        st.header("🤖 Modèle")
        model_info = get_model_info()
        if model_info:
            st.metric("Seuil", f"{model_info.get('threshold', 0.5):.2%}")
        else:
            st.caption("Infos indisponibles")
        
        st.divider()
        
        # Section Statistiques du Dataset (compact)
        st.header("📊 Dataset")
        
        if reference_data is not None and not reference_data.empty:
            st.metric("Clients", f"{len(reference_data):,}")
            
            if 'TARGET' in reference_data.columns:
                st.metric("Taux défaut", f"{reference_data['TARGET'].mean():.1%}")
            
            with st.expander("💰 Finances"):
                if 'AMT_INCOME_TOTAL' in reference_data.columns:
                    st.write(f"Revenu: {reference_data['AMT_INCOME_TOTAL'].median():,.0f}€")
                if 'AMT_CREDIT' in reference_data.columns:
                    st.write(f"Crédit: {reference_data['AMT_CREDIT'].median():,.0f}€")
            
            with st.expander("📊 Scores"):
                for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
                    if col in reference_data.columns:
                        st.write(f"{col}: {reference_data[col].median():.3f}")
        else:
            st.warning("📂 Données manquantes")
            st.caption("Copiez application_train.csv dans data/")
        
        st.divider()
        st.caption("v1.0.0 • Home Credit Scoring")
    
    # Initialiser features dans session_state pour modification en temps réel
    if 'features' not in st.session_state:
        st.session_state.features = {}
    
    # ============================================
    # Contenu principal basé sur la navigation
    # ============================================
    
    current_page = st.session_state.get('current_page', 'scoring')
    
    # ============================================
    # Page: Scoring Client
    # ============================================
    if current_page == "scoring":
        st.header("🎯 Évaluation du risque client")
        st.markdown("Saisissez les caractéristiques pour obtenir le score avec une interprétation détaillée.")
        
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
        
        # Stocker features dans session_state pour comparaison
        st.session_state.features = features.copy()
        
        st.markdown("---")
        
        # Boutons d'action accessibles
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            predict_btn = st.button("🎯 Calculer le score", type="primary", use_container_width=True, help="Calculer la probabilité de défaut")
        
        with col_btn2:
            explain_btn = st.button("🔍 Expliquer le score", use_container_width=True, help="Voir les facteurs influençant le score")
        
        # Affichage des résultats
        if predict_btn:
            with st.spinner("Calcul en cours..."):
                result = predict(features)
            
            if result:
                probability = result.get("probability", result.get("proba", 0.5))
                threshold = result.get("threshold", 0.44)
                
                # Interprétation intelligible
                interpretation = interpret_score(probability, threshold)
                
                st.markdown("---")
                st.header("📊 Résultats de l'évaluation")
                
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    # Jauge de risque
                    fig_gauge = create_gauge_chart(probability, threshold)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_res2:
                    # Interprétation textuelle accessible
                    if interpretation["decision"] == "ACCORDÉ":
                        st.markdown(f"""
                        <div class="risk-low" role="alert">
                        <h3>✅ Crédit {interpretation['decision']}</h3>
                        <p><strong>Probabilité de défaut:</strong> {interpretation['probability_text']}</p>
                        <p><strong>Seuil de décision:</strong> {interpretation['threshold_text']}</p>
                        <p><strong>Écart au seuil:</strong> -{interpretation['distance_text']}</p>
                        <p><strong>Confiance:</strong> {interpretation['confidence']}</p>
                        <hr>
                        <p>{interpretation['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-high" role="alert">
                        <h3>❌ Crédit {interpretation['decision']}</h3>
                        <p><strong>Probabilité de défaut:</strong> {interpretation['probability_text']}</p>
                        <p><strong>Seuil de décision:</strong> {interpretation['threshold_text']}</p>
                        <p><strong>Écart au seuil:</strong> +{interpretation['distance_text']}</p>
                        <p><strong>Confiance:</strong> {interpretation['confidence']}</p>
                        <hr>
                        <p>{interpretation['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Métriques clés
                st.subheader("📈 Métriques clés")
                met1, met2, met3, met4 = st.columns(4)
                met1.metric("Probabilité", f"{probability*100:.1f}%")
                met2.metric("Seuil", f"{threshold*100:.1f}%")
                met3.metric("Écart", f"{(probability-threshold)*100:+.1f}%")
                met4.metric("Décision", interpretation['decision'])
                
                # Informations descriptives du client
                st.subheader("👤 Résumé du profil client")
                
                profile_col1, profile_col2, profile_col3 = st.columns(3)
                
                with profile_col1:
                    st.markdown("**Situation financière**")
                    st.write(f"- Revenu: {features['AMT_INCOME_TOTAL']:,.0f} €")
                    st.write(f"- Crédit demandé: {features['AMT_CREDIT']:,.0f} €")
                    st.write(f"- Ratio crédit/revenu: {features['CREDIT_INCOME_RATIO']:.2f}")
                
                with profile_col2:
                    st.markdown("**Situation personnelle**")
                    age_years = abs(features['DAYS_BIRTH']) // 365
                    employed_years = abs(features['DAYS_EMPLOYED']) // 365
                    st.write(f"- Âge: {age_years} ans")
                    st.write(f"- Ancienneté emploi: {employed_years} ans")
                    st.write(f"- Enfants: {features['CNT_CHILDREN']}")
                
                with profile_col3:
                    st.markdown("**Scores de crédit**")
                    st.write(f"- Score moyen: {features['EXT_SOURCE_MEAN']:.2f}")
                    st.write(f"- Propriétaire: {'Oui' if features['FLAG_OWN_REALTY'] else 'Non'}")
                    st.write(f"- Véhicule: {'Oui' if features['FLAG_OWN_CAR'] else 'Non'}")
        
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
                        expl = get_feature_explanation(feat)
                        st.markdown(f"- **{feat}** ({val:.3f}): {expl}")
    
    # ============================================
    # Page: Comparaison avec la population
    # ============================================
    elif current_page == "comparison":
        st.header("📊 Comparaison avec la population")
        
        if reference_data is None:
            st.warning("⚠️ Données de référence non disponibles pour la comparaison.")
            st.info("Placez le fichier `application_train.csv` dans le dossier `data/`")
        else:
            st.markdown("Comparez les caractéristiques du client avec l'ensemble de la population ou un groupe de clients similaires.")
            
            # Sélection du groupe de comparaison
            group_filter = st.selectbox(
                "Groupe de comparaison",
                ["Tous les clients", "Clients sans défaut (TARGET=0)", "Clients en défaut (TARGET=1)"],
                help="Sélectionnez le groupe avec lequel comparer le client"
            )
            
            # Features disponibles pour comparaison
            numeric_features = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                               "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            available_features = [f for f in numeric_features if f in reference_data.columns]
            
            if st.session_state.features:
                features = st.session_state.features
                
                # Graphique radar multi-critères
                st.subheader("🎯 Vue d'ensemble - Comparaison multi-critères")
                
                radar_features = st.multiselect(
                    "Caractéristiques à comparer",
                    available_features,
                    default=available_features[:5] if len(available_features) >= 5 else available_features,
                    help="Choisissez jusqu'à 8 caractéristiques"
                )
                
                if radar_features and len(radar_features) >= 3:
                    fig_radar = create_radar_comparison(features, reference_data, radar_features)
                    if fig_radar:
                        st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Sélectionnez au moins 3 caractéristiques pour le graphique radar")
                
                st.divider()
                
                # Comparaison individuelle par feature
                st.subheader("📈 Comparaison détaillée par caractéristique")
                
                selected_feature = st.selectbox(
                    "Sélectionnez une caractéristique",
                    available_features,
                    help="Voir la distribution et la position du client"
                )
                
                if selected_feature and selected_feature in features:
                    client_value = features[selected_feature]
                    
                    fig_comparison = create_comparison_chart(
                        client_value, selected_feature, reference_data, group_filter
                    )
                    
                    if fig_comparison:
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Statistiques textuelles
                        ref_data = reference_data[selected_feature].dropna()
                        percentile = (ref_data < client_value).mean() * 100
                        
                        st.markdown(f"""
                        **Statistiques pour {selected_feature}:**
                        - Valeur du client: **{client_value:,.2f}**
                        - Moyenne population: {ref_data.mean():,.2f}
                        - Médiane population: {ref_data.median():,.2f}
                        - Position du client: **{percentile:.0f}e percentile**
                        """)
            else:
                st.info("👆 Veuillez d'abord saisir les caractéristiques d'un client dans 'Scoring Client'")
    
    # ============================================
    # Page: Import fichier / Simulation temps réel
    # ============================================
    elif current_page == "simulation":
        st.header("📁 Import de fichier et simulation")
        
        # Section Import
        st.subheader("📤 Import de fichier CSV")
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
        
        st.divider()
        
        # Section Simulation temps réel
        st.subheader("🔄 Simulation interactive")
        st.markdown("Modifiez les valeurs ci-dessous pour voir l'impact sur le score en temps réel.")
        
        if st.session_state.features:
            sim_col1, sim_col2 = st.columns(2)
            
            with sim_col1:
                sim_income = st.number_input(
                    "Revenu simulé (€)",
                    min_value=0.0,
                    max_value=10000000.0,
                    value=st.session_state.features.get("AMT_INCOME_TOTAL", 150000.0),
                    step=10000.0,
                    key="sim_income"
                )
                
                sim_credit = st.number_input(
                    "Crédit simulé (€)",
                    min_value=0.0,
                    max_value=5000000.0,
                    value=st.session_state.features.get("AMT_CREDIT", 500000.0),
                    step=50000.0,
                    key="sim_credit"
                )
            
            with sim_col2:
                sim_ext1 = st.slider(
                    "Score externe 1 simulé",
                    0.0, 1.0,
                    st.session_state.features.get("EXT_SOURCE_1", 0.5),
                    0.01,
                    key="sim_ext1"
                )
                
                sim_ext2 = st.slider(
                    "Score externe 2 simulé",
                    0.0, 1.0,
                    st.session_state.features.get("EXT_SOURCE_2", 0.6),
                    0.01,
                    key="sim_ext2"
                )
            
            if st.button("🔄 Recalculer le score", type="primary"):
                # Construire les features simulées
                sim_features = st.session_state.features.copy()
                sim_features["AMT_INCOME_TOTAL"] = sim_income
                sim_features["AMT_CREDIT"] = sim_credit
                sim_features["EXT_SOURCE_1"] = sim_ext1
                sim_features["EXT_SOURCE_2"] = sim_ext2
                
                if sim_income > 0:
                    sim_features["CREDIT_INCOME_RATIO"] = sim_credit / sim_income
                
                sim_features["EXT_SOURCE_MEAN"] = np.mean([
                    sim_ext1, sim_ext2, sim_features.get("EXT_SOURCE_3", 0.55)
                ])
                
                with st.spinner("Calcul..."):
                    sim_result = predict(sim_features)
                    orig_result = predict(st.session_state.features)
                
                if sim_result and orig_result:
                    sim_prob = sim_result.get("probability", 0.5)
                    orig_prob = orig_result.get("probability", 0.5)
                    delta = sim_prob - orig_prob
                    
                    st.markdown("### Résultat de la simulation")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Score original", f"{orig_prob*100:.1f}%")
                    res_col2.metric("Score simulé", f"{sim_prob*100:.1f}%", f"{delta*100:+.1f}%")
                    res_col3.metric("Décision", sim_result.get("decision", "N/A").upper())
                    
                    if delta < 0:
                        st.success("✅ Les modifications améliorent le profil de risque")
                    elif delta > 0:
                        st.warning("⚠️ Les modifications augmentent le risque")
                    else:
                        st.info("ℹ️ Pas de changement significatif")
        else:
            st.info("👆 Saisissez d'abord un client dans 'Scoring Client'")
    
    # ============================================
    # Page: Data Drift
    # ============================================
    elif current_page == "drift":
        st.header("📈 Surveillance du Data Drift")
        
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
        
        # Afficher le rapport HTML si disponible
        if os.path.exists(DRIFT_REPORT_PATH):
            with open(DRIFT_REPORT_PATH, 'r', encoding='utf-8') as f:
                report_html = f.read()
            
            st.markdown("### Rapport complet Evidently")
            st.components.v1.html(report_html, height=1200, scrolling=True)
        else:
            st.warning(f"⚠️ Rapport de drift non trouvé: {DRIFT_REPORT_PATH}")
            st.info("""
            **Pour générer le rapport**:
            1. Exécutez le notebook `notebooks/04_Drift_Evidently.ipynb`
            2. Le rapport sera généré dans `reports/evidently_full_report.html`
            """)
    
    # ============================================
    # Page: Documentation
    # ============================================
    elif current_page == "docs":
        st.header("📖 Documentation")
        
        st.markdown(f"""
        ## À propos
        
        Cette application permet d'évaluer le risque de défaut de paiement 
        pour des demandes de crédit, en utilisant un modèle de Machine Learning
        entraîné sur les données Home Credit.
        
        ## Fonctionnalités
        
        - **Scoring Client**: Évaluation du risque avec interprétation intelligible
        - **Comparaison**: Position du client par rapport à la population
        - **Simulation**: Modification en temps réel des caractéristiques
        - **Data Drift**: Surveillance de la qualité des données
        
        ## Accessibilité WCAG
        
        Cette application respecte les critères d'accessibilité **WCAG 2.1 niveau AA**:
        - Contrastes de couleurs suffisants (ratio 4.5:1 minimum)
        - Navigation au clavier possible
        - Labels descriptifs pour tous les éléments interactifs
        - Messages d'état accessibles
        
        ## Méthodologie
        
        ### Modèle
        - **Algorithme**: LightGBM (Gradient Boosting)
        - **Métrique d'optimisation**: Coût métier (FN=10, FP=1)
        - **Seuil de décision**: Optimisé pour minimiser le coût métier
        
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
        
        ## Liens utiles
        
        - [Documentation API (Swagger)]({API_URL}/docs)
        - [MLflow UI]({MLFLOW_URL})
        - [Guide de déploiement Render](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/RENDER_SETUP.md)
        
        ## Coût métier
        
        Le modèle optimise un coût métier asymétrique:
        - **Faux Négatif (FN)**: Coût = 10 (client défaillant accepté)
        - **Faux Positif (FP)**: Coût = 1 (bon client refusé)
        """)


if __name__ == "__main__":
    main()
