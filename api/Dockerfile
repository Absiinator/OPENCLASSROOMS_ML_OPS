# =============================================================================
# Dockerfile - Home Credit Scoring API
# =============================================================================
# Image optimisée pour le déploiement de l'API de scoring crédit
# Compatible avec Render, Railway, et autres cloud providers

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.0.0"
LABEL description="API de scoring crédit Home Credit"

# Build args et variables d'environnement (injectables via --build-arg)
# Permet à GitHub Actions de transmettre des valeurs au moment du build si nécessaire
ARG HOST=0.0.0.0
ARG PORT=8000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=${HOST} \
    PORT=${PORT}

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système (curl et unzip pour télécharger les données)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copier les fichiers de requirements
COPY api/requirements.txt ./api/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r api/requirements.txt

# Copier le code source
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY reports/ ./reports/

# Créer les répertoires nécessaires
RUN mkdir -p /app/data /app/reports

# Télécharger et décompresser les données Home Credit
# URL: OpenClassrooms S3 bucket
# Extraction robuste: replace les CSV à la racine de /app/data
RUN curl -L -o /tmp/data.zip \
    "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip" \
    && mkdir -p /tmp/data_extract \
    && unzip -o /tmp/data.zip -d /tmp/data_extract \
    && find /tmp/data_extract -type f -name "*.csv" -exec mv {} /app/data/ \; \
    && rm -rf /tmp/data.zip /tmp/data_extract \
    && echo "✅ Données téléchargées et décompressées dans /app/data/"

# Changer la propriété des fichiers
RUN chown -R appuser:appuser /app

# Utiliser l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Commande de démarrage utilisant HOST et PORT
CMD ["sh", "-c", "echo '[API] Démarrage avec HOST=${HOST} PORT=${PORT}' && uvicorn api.main:app --host ${HOST} --port ${PORT}"]
