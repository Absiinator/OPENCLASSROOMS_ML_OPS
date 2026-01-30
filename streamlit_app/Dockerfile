# =============================================================================
# Dockerfile - Home Credit Scoring Dashboard (Streamlit)
# =============================================================================
# Image pour le déploiement du dashboard Streamlit sur Render
# Déploiement gratuit en Web Service

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.0.0"
LABEL description="Dashboard Streamlit Home Credit Scoring"

# Variables d'environnement (noms conformes aux services Render)
# API_URL et MLFLOW_URL seront configurées sur Render
# STREAMLIT_SERVER_ADDRESS et STREAMLIT_SERVER_PORT pour la configuration Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    API_URL=http://localhost:8000 \
    MLFLOW_URL=http://localhost:5000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Installer les dépendances système minimales (curl et unzip pour télécharger les données)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copier requirements et installer
COPY streamlit_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY streamlit_app/app.py ./app.py

# Note: Pas de fallback local - toutes les prédictions passent par l'API
# On copie src/ uniquement pour d'éventuels utilitaires

# Copier les sources (utilitaires si nécessaire)
COPY src/ ./src/

# Créer le répertoire data et télécharger les données Home Credit
# URL: OpenClassrooms S3 bucket
RUN mkdir -p /app/data \
    && curl -L -o /tmp/data.zip \
    "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip" \
    && unzip -o /tmp/data.zip -d /app/data/ \
    && rm /tmp/data.zip \
    && echo "✅ Données téléchargées et décompressées dans /app/data/"

# Copier les rapports Evidently pour l'onglet Data Drift
COPY reports/ ./reports/

# Configuration Streamlit
RUN mkdir -p ~/.streamlit
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > ~/.streamlit/config.toml

# Permissions
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande de démarrage avec logs des variables d'environnement
CMD ["sh", "-c", "echo '[DASHBOARD] API_URL=${API_URL}' && echo '[DASHBOARD] MLFLOW_URL=${MLFLOW_URL}' && echo '[DASHBOARD] PORT=${PORT}' && streamlit run app.py --server.port=${PORT} --server.address=${STREAMLIT_SERVER_ADDRESS}"]
