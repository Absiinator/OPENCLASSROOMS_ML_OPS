# =============================================================================
# Dockerfile - MLflow UI Server (Ultra-Light pour Render Free Tier)
# =============================================================================
# Image minimaliste pour le tracking MLflow sur Render (512MB RAM)
# IMPORTANT: Utilise Flask directement avec --workers 1 pour éviter OOM
#
# VARIABLES D'ENVIRONNEMENT:
# - PORT: défini par Render automatiquement (défaut: 5000)
# =============================================================================

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.0.1"
LABEL description="MLflow UI Server - Ultra Light (Single Worker)"

# Variables d'environnement (PORT sera écrasé par Render)
ARG PORT=5000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=${PORT} \
    MLFLOW_TRACKING_URI=/app/mlruns \
    # IMPORTANT: Limiter gunicorn à 1 worker pour free tier
    GUNICORN_CMD_ARGS="--workers=1 --threads=2 --timeout=120"

WORKDIR /app

# Installer MLflow version légère (sans extras lourds)
RUN pip install --no-cache-dir "mlflow==2.9.2" \
    && pip cache purge \
    && rm -rf /root/.cache

# Créer le répertoire mlruns
RUN mkdir -p /app/mlruns

# Copier les expériences (notebooks/mlruns)
COPY notebooks/mlruns/. /app/mlruns/

# Nettoyer et corriger les chemins absolus → relatifs pour Docker
# Les meta.yaml peuvent contenir des chemins locaux qu'il faut corriger
RUN echo "🔧 Correction des chemins MLflow..." && \
    rm -rf /app/mlruns/.trash 2>/dev/null || true && \
    # Corriger artifact_location (expériences) - remplacer tout chemin absolu par le répertoire courant
    find /app/mlruns -name "meta.yaml" -exec sed -i 's|artifact_location:.*mlruns/[0-9]*|artifact_location: .|g' {} \; 2>/dev/null || true && \
    # Corriger artifact_uri (runs) - remplacer par chemin relatif vers artifacts
    find /app/mlruns -path '*/*[0-9a-f]*/meta.yaml' -exec sed -i 's|artifact_uri:.*artifacts$|artifact_uri: ./artifacts|g' {} \; 2>/dev/null || true && \
    find /app/mlruns -path '*/*[0-9a-f]*/meta.yaml' -exec sed -i 's|artifact_uri:.*/artifacts$|artifact_uri: ./artifacts|g' {} \; 2>/dev/null || true && \
    # Vérification
    echo "=== Structure MLruns ===" && \
    ls -la /app/mlruns/ && \
    echo "=== Exemples de meta.yaml ===" && \
    head -3 /app/mlruns/*/meta.yaml 2>/dev/null || echo "Pas de meta.yaml trouvé" && \
    echo "✅ MLflow prêt"

# Exposer le port
EXPOSE ${PORT}

# PAS de healthcheck avec curl (nécessite installation supplémentaire)
# Render gère son propre healthcheck

# Commande de démarrage - Flask simple avec 1 seul worker pour éviter OOM
# IMPORTANT: --workers 1 est CRITIQUE pour le free tier 512MB
CMD ["sh", "-c", "echo '[MLflow] Démarrage sur port ${PORT} (1 worker pour free tier)' && mlflow server --host 0.0.0.0 --port ${PORT} --backend-store-uri /app/mlruns --serve-artifacts --workers 1"]
