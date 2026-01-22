# =============================================================================
# Dockerfile - Home Credit Scoring Dashboard (Streamlit)
# =============================================================================
# Image pour le déploiement du dashboard Streamlit sur Render
# Déploiement gratuit en Web Service

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.0.0"
LABEL description="Dashboard Streamlit Home Credit Scoring"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    API_URL=http://localhost:8000

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copier requirements et installer
COPY streamlit_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY streamlit_app/app.py ./app.py

# Copier les modèles pour le fallback local (optionnel)
COPY models/ ./models/

# Copier les sources nécessaires pour le fallback
COPY src/ ./src/

# Configuration Streamlit
RUN mkdir -p ~/.streamlit
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > ~/.streamlit/config.toml

# Permissions
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande de démarrage
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
