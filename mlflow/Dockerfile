# Dockerfile pour MLflow UI Server
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=/app/mlruns \
    PORT=5000

WORKDIR /app

# Installer MLflow et dépendances
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    boto3 \
    psycopg2-binary

# Créer le répertoire pour les artefacts MLflow
# Note: En production (Render), mlruns/ sera vide au démarrage car non persistant.
# Les expériences doivent être loggées depuis les scripts d'entraînement vers ce serveur.
RUN mkdir -p /app/mlruns

# Exposer le port
EXPOSE ${PORT}

# Commande de démarrage du serveur MLflow
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri /app/mlruns \
    --default-artifact-root /app/mlruns
