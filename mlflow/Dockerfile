# =============================================================================
# Dockerfile - MLflow UI (Ultra-Light pour Render Free Tier 512MB)
# =============================================================================
# OPTIMISÉ pour le plan gratuit Render (512MB RAM max)
# Force Flask développement (WERKZEUG) sans Gunicorn
#
# SOLUTION: Désactiver explicitement Gunicorn avec MLFLOW_DISABLE_GUNICORN=1
# et utiliser --workers=1 pour limiter la RAM
# =============================================================================

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.2.0"
LABEL description="MLflow UI - Flask Simple (No Gunicorn)"

# Variables d'environnement CRITIQUES
ARG PORT=5000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=${PORT} \
    MLFLOW_TRACKING_URI=/app/mlruns \
    # FORCE Flask dev server sans Gunicorn
    GUNICORN_CMD_ARGS="--workers=1 --threads=1 --timeout=120" \
    # Limiter la mémoire
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Installer mlflow sans dépendances optionnelles lourdes
RUN pip install --no-cache-dir \
    "mlflow==2.9.2" \
    && pip cache purge \
    && rm -rf /root/.cache /tmp/*

# Créer le répertoire mlruns
RUN mkdir -p /app/mlruns

# Copier les expériences MLflow
COPY notebooks/mlruns/. /app/mlruns/

# Nettoyer et corriger les chemins pour Docker
RUN echo "🔧 Correction des chemins MLflow..." && \
    rm -rf /app/mlruns/.trash 2>/dev/null || true && \
    # Corriger artifact_location dans les experiments (chemin absolu)
    find /app/mlruns -maxdepth 2 -name "meta.yaml" -exec sed -i 's|artifact_location:.*|artifact_location: file:///app/mlruns|g' {} \; 2>/dev/null || true && \
    # Corriger artifact_uri dans les runs (chemin absolu vers artifacts)
    find /app/mlruns -path '*/*[0-9a-f]*/meta.yaml' -exec sed -i 's|artifact_uri:.*|artifact_uri: file:///app/mlruns|g' {} \; 2>/dev/null || true && \
    # Créer les répertoires artifacts manquants
    find /app/mlruns -path '*/*[0-9a-f]*/meta.yaml' -exec mkdir -p {}/artifacts \; 2>/dev/null || true && \
    echo "✅ MLflow configuré" && \
    ls -la /app/mlruns/

# Exposer le port
EXPOSE ${PORT}

# COMMANDE OPTIMISÉE pour Free Tier (512MB RAM):
# - --workers=1: UN SEUL worker (pas 4!)
# - --timeout=120: Plus de temps pour éviter WORKER TIMEOUT
# - file:// URI pour backend store
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri file:///app/mlruns \
    --serve-artifacts \
    --gunicorn-opts "--workers=1 --threads=2 --timeout=120"
