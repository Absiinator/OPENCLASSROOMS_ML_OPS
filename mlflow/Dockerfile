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
RUN mkdir -p /app/mlruns /app/notebooks_mlruns /app/mlruns_root

# Copier les runs MLflow depuis les 2 sources
# 1. mlruns/ à la racine du projet (peut contenir des runs)
COPY mlruns/. /app/mlruns_root/

# 2. notebooks/mlruns/ qui contient les vrais runs d'expériences
COPY notebooks/mlruns/. /app/notebooks_mlruns/

# Fusionner les deux sources dans /app/mlruns (destination finale)
RUN echo "📦 Copie des MLflow runs..." && \
    echo "=== Contenu mlruns_root ===" && \
    ls -la /app/mlruns_root/ && \
    echo "=== Contenu notebooks_mlruns ===" && \
    ls -la /app/notebooks_mlruns/ && \
    echo "=== Fusion dans mlruns ===" && \
    cp -rv /app/mlruns_root/* /app/mlruns/ 2>/dev/null || true && \
    cp -rv /app/notebooks_mlruns/* /app/mlruns/ 2>/dev/null || true && \
    rm -rf /app/notebooks_mlruns /app/mlruns_root && \
    echo "✅ Fusion terminée" && \
    echo "=== Contenu final mlruns ===" && \
    ls -laR /app/mlruns/ | head -50 && \
    echo "=== Recherche meta.yaml ===" && \
    find /app/mlruns -name "meta.yaml" -type f 2>/dev/null | head -20 || echo "⚠️ Aucun meta.yaml trouvé"

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
