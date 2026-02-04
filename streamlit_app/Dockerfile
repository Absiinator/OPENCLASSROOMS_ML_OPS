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
# Build args et variables d'environnement (injectables via --build-arg)
# API_URL et MLFLOW_URL peuvent être passées au build ou configurées sur Render
ARG API_URL=http://localhost:8000
ARG MLFLOW_URL=http://localhost:5000
ARG PORT=8501
ARG STREAMLIT_SERVER_ADDRESS=0.0.0.0
ARG STREAMLIT_SERVER_PORT=8501
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=${PORT} \
    API_URL=${API_URL} \
    MLFLOW_URL=${MLFLOW_URL} \
    STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS} \
    STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT}

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

# Copier l'application et les modules locaux
COPY streamlit_app/app.py ./app.py
COPY streamlit_app/api_client.py ./api_client.py
COPY streamlit_app/constants.py ./constants.py

# Note: Pas de fallback local - toutes les prédictions passent par l'API
# On copie src/ uniquement pour d'éventuels utilitaires

# Copier les sources (utilitaires si nécessaire)
COPY src/ ./src/

# Créer le répertoire data et télécharger les données Home Credit
# URL: OpenClassrooms S3 bucket
RUN mkdir -p /app/data \
    && curl -L -o /tmp/data.zip \
    "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip" \
    && mkdir -p /tmp/data_extract \
    && unzip -o /tmp/data.zip -d /tmp/data_extract \
    && find /tmp/data_extract -type f -name "*.csv" -exec mv {} /app/data/ \; \
    && rm -rf /tmp/data.zip /tmp/data_extract \
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
