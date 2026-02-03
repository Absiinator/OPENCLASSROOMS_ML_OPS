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

# Copier les expériences (notebooks/mlruns avec chemins normalisés)
COPY notebooks/mlruns/. /app/mlruns/

# Vérifier la structure - pas besoin de sed complexe car les chemins sont relatifs!
RUN echo "📁 Structure MLflow (chemins relatifs = reproductibles):" && \
    ls -la /app/mlruns/ && \
    echo "✅ MLflow prêt pour déploiement"

# Exposer le port
EXPOSE ${PORT}

# Health check (MLflow UI répond sur /)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Commande de démarrage avec logs détaillés
CMD ["sh", "-c", "echo '[MLflow] === DÉMARRAGE ===' && echo '[MLflow] PORT=${PORT}' && echo '[MLflow] Contenu mlruns:' && ls -la /app/mlruns/ && echo '[MLflow] Expériences:' && ls /app/mlruns/*/meta.yaml 2>/dev/null | head -5 && echo '[MLflow] Démarrage UI...' && mlflow ui --host 0.0.0.0 --port ${PORT} --backend-store-uri /app/mlruns"]
