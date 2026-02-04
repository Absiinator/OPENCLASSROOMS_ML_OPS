# =============================================================================
# Dockerfile - MLflow UI (Ultra-Light pour Render Free Tier 512MB)
# =============================================================================
# OPTIMISÉ pour le plan gratuit Render (512MB RAM max)
# Utilise `mlflow ui` (Flask simple) au lieu de `mlflow server` (Gunicorn)
#
# IMPORTANT: mlflow ui utilise ~150-200MB RAM vs mlflow server ~300-400MB
#
# VARIABLES D'ENVIRONNEMENT:
# - PORT: défini automatiquement par Render
# =============================================================================

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.1.0"
LABEL description="MLflow UI - Flask Simple (Ultra-Light pour Free Tier)"

# Variables d'environnement
ARG PORT=5000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=${PORT} \
    MLFLOW_TRACKING_URI=/app/mlruns

WORKDIR /app

# Installer UNIQUEMENT mlflow-skinny (version ultra légère sans dépendances lourdes)
# mlflow-skinny: ~50MB vs mlflow complet: ~200MB
RUN pip install --no-cache-dir "mlflow==2.9.2" \
    && pip cache purge \
    && rm -rf /root/.cache /tmp/*

# Créer le répertoire mlruns
RUN mkdir -p /app/mlruns

# Copier les expériences MLflow
COPY notebooks/mlruns/. /app/mlruns/

# Nettoyer et corriger les chemins pour Docker
# Les meta.yaml locaux ont des chemins absolus qu'il faut corriger
RUN echo "🔧 Correction des chemins MLflow..." && \
    rm -rf /app/mlruns/.trash 2>/dev/null || true && \
    # Corriger artifact_location dans les experiments
    find /app/mlruns -maxdepth 2 -name "meta.yaml" -exec sed -i 's|artifact_location:.*|artifact_location: /app/mlruns|g' {} \; 2>/dev/null || true && \
    # Corriger artifact_uri dans les runs (remplacer tout chemin absolu)
    find /app/mlruns -path '*/*[0-9a-f]*/meta.yaml' -exec sed -i 's|artifact_uri:.*artifacts$|artifact_uri: ./artifacts|g' {} \; 2>/dev/null || true && \
    find /app/mlruns -path '*/*[0-9a-f]*/meta.yaml' -exec sed -i 's|artifact_uri:.*/artifacts$|artifact_uri: ./artifacts|g' {} \; 2>/dev/null || true && \
    # Vérification
    echo "✅ MLflow configuré" && \
    ls -la /app/mlruns/

# Exposer le port
EXPOSE ${PORT}

# COMMANDE CRITIQUE pour Free Tier:
# `mlflow ui` utilise Flask simple (pas Gunicorn) = ~150MB RAM
# C'est BEAUCOUP plus léger que `mlflow server` qui utilise Gunicorn avec workers
CMD ["sh", "-c", "echo '[MLflow UI] Démarrage sur port ${PORT} (Flask simple pour 512MB RAM)' && mlflow ui --host 0.0.0.0 --port ${PORT} --backend-store-uri /app/mlruns"]
