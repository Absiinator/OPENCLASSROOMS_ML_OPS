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

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
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

# Créer les répertoires nécessaires
RUN mkdir -p /app/data /app/reports

# Changer la propriété des fichiers
RUN chown -R appuser:appuser /app

# Utiliser l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Commande de démarrage
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
