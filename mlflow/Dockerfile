# =============================================================================
# Dockerfile - MLflow UI Server (Ultra-Light pour Render Free Tier)
# =============================================================================
# Image minimaliste pour le tracking MLflow sur Render (512MB RAM)
# Utilise mlflow ui (Flask) au lieu de mlflow server (gunicorn) = moins de RAM
#
# VARIABLES D'ENVIRONNEMENT:
# - PORT: défini par Render automatiquement (défaut: 5000)
# =============================================================================

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.0.0"
LABEL description="MLflow UI Server - Ultra Light"

# Variables d'environnement (PORT sera écrasé par Render)
ARG PORT=5000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=${PORT} \
    MLFLOW_TRACKING_URI=/app/mlruns

WORKDIR /app

# Installer MLflow version légère (sans extras lourds)
RUN pip install --no-cache-dir "mlflow==2.9.2" \
    && pip cache purge \
    && rm -rf /root/.cache

# Créer le répertoire mlruns
RUN mkdir -p /app/mlruns

# Copier les expériences (notebooks/mlruns)
COPY notebooks/mlruns/. /app/mlruns/

# Nettoyer et corriger les chemins pour Docker
# IMPORTANT: Les meta.yaml contiennent des chemins locaux absolus qu'il faut remplacer
RUN echo "🔧 Correction des chemins MLflow pour Docker..." && \
    # Supprimer les fichiers temporaires
    rm -rf /app/mlruns/.trash 2>/dev/null || true && \
    # Corriger TOUS les chemins artifact_location (expériences)
    find /app/mlruns -name "meta.yaml" -type f -exec sed -i 's|artifact_location:.*|artifact_location: /app/mlruns|g' {} \; 2>/dev/null || true && \
    # Corriger TOUS les chemins artifact_uri (runs) - pattern générique
    find /app/mlruns -name "meta.yaml" -type f -exec sed -i 's|artifact_uri: /.*mlruns/|artifact_uri: /app/mlruns/|g' {} \; 2>/dev/null || true && \
    # Supprimer les modèles volumineux (on garde les métriques et params)
    find /app/mlruns -name "*.pkl" -delete 2>/dev/null || true && \
    find /app/mlruns -name "*.joblib" -delete 2>/dev/null || true && \
    # Afficher la structure finale
    echo "=== Structure MLruns ===" && \
    find /app/mlruns -type d -name "[0-9]*" | head -5 && \
    du -sh /app/mlruns && \
    echo "✅ MLflow prêt pour déploiement"

# Exposer le port
EXPOSE ${PORT}

# Commande de démarrage
CMD ["sh", "-c", "echo '[MLflow] Démarrage UI sur port ${PORT}' && mlflow ui --host 0.0.0.0 --port ${PORT} --backend-store-uri /app/mlruns"]
