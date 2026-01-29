# =============================================================================
# Dockerfile - MLflow UI Server
# =============================================================================
# Image optimisée pour le tracking MLflow sur Render
# Configuration légère pour tier gratuit (512MB RAM)

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.0.0"
LABEL description="MLflow UI Server Home Credit Scoring"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=/app/mlruns \
    PORT=5000

WORKDIR /app

# Installer MLflow (version légère pour tier gratuit)
RUN pip install --no-cache-dir mlflow==2.9.2

# Créer le répertoire pour les artefacts MLflow
RUN mkdir -p /app/mlruns

# Copier les runs MLflow existants depuis la racine du projet
COPY mlruns/ /app/mlruns/

# Exposer le port
EXPOSE ${PORT}

# Commande de démarrage du serveur MLflow
# Configuration optimisée pour tier gratuit Render (512MB RAM):
# - 1 worker au lieu de 4 par défaut
# - Timeout élevé pour éviter les WORKER TIMEOUT
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri /app/mlruns \
    --default-artifact-root /app/mlruns \
    --workers 1 \
    --gunicorn-opts "--timeout 120 --keep-alive 5 --log-level warning"
