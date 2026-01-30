# =============================================================================
# Dockerfile - MLflow UI Server
# =============================================================================
# Image optimisée pour le tracking MLflow sur Render
# Configuration ultra-légère pour tier gratuit (512MB RAM)

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

# Installer MLflow sans dépendances lourdes (version minimale pour UI)
RUN pip install --no-cache-dir mlflow==2.9.2

# Créer le répertoire mlruns
RUN mkdir -p /app/mlruns

# IMPORTANT: On copie SEULEMENT notebooks/mlruns qui contient les vraies expériences
# Le dossier mlruns/ à la racine contient un doublon qu'on ignore
COPY notebooks/mlruns/. /app/mlruns/

# Nettoyer les fichiers .trash et corriger les chemins absolus pour Docker
RUN echo "🔧 Nettoyage et configuration MLflow..." && \
    rm -rf /app/mlruns/.trash 2>/dev/null || true && \
    # Corriger artifact_location dans meta.yaml des expériences
    find /app/mlruns -maxdepth 2 -name "meta.yaml" -type f -exec sed -i 's|artifact_location:.*|artifact_location: /app/mlruns|g' {} \; 2>/dev/null || true && \
    # Corriger artifact_uri dans meta.yaml des runs (chemins locaux -> Docker)
    find /app/mlruns -name "meta.yaml" -type f -exec sed -i 's|artifact_uri:.*home-credit-scoring/notebooks/mlruns/|artifact_uri: /app/mlruns/|g' {} \; 2>/dev/null || true && \
    find /app/mlruns -name "meta.yaml" -type f -exec sed -i 's|artifact_uri:.*Lepage_Jeffrey.*mlruns/|artifact_uri: /app/mlruns/|g' {} \; 2>/dev/null || true && \
    # Afficher le contenu pour debug
    echo "=== Experiments trouvés ===" && \
    find /app/mlruns -maxdepth 2 -name "meta.yaml" -exec dirname {} \; 2>/dev/null && \
    echo "=== Runs trouvés ===" && \
    find /app/mlruns -maxdepth 3 -type d -name "[0-9a-f]*" 2>/dev/null | head -10 && \
    echo "✅ MLflow configuré"

# Exposer le port
EXPOSE ${PORT}

# Commande de démarrage SANS gunicorn (utiliser le serveur intégré plus léger)
# Le serveur MLflow intégré est suffisant pour le tier gratuit
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port ${PORT} --backend-store-uri /app/mlruns --default-artifact-root /app/mlruns"]
