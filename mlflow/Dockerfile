# =============================================================================
# Dockerfile - MLflow UI (Render Free Tier 512MB)
# =============================================================================
# Optimisé pour le plan gratuit Render (512MB RAM max)
# Utilise mlflow server avec 1 worker Gunicorn pour limiter la RAM
# =============================================================================

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Home Credit Scoring Team"
LABEL version="1.2.0"
LABEL description="MLflow UI - Gunicorn 1 worker (Render Free Tier)"

# Variables d'environnement CRITIQUES
ARG PORT=5000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=${PORT} \
    MLFLOW_TRACKING_URI=/app/mlruns \
    # Limiter le nombre de workers Gunicorn
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
    # Assurer un meta.yaml dans les dossiers "models" d'expérience
    # (évite l'erreur MissingConfig quand MLflow scanne les runs)
    for d in /app/mlruns/*/models; do \
        if [ -d "$d" ] && [ ! -f "$d/meta.yaml" ]; then \
            exp_id=$(basename "$(dirname "$d")"); \
            mkdir -p "${d}/artifacts" 2>/dev/null || true; \
            printf "artifact_uri: file:///app/mlruns/${exp_id}/models/artifacts\nend_time: 0\nentry_point_name: ''\nexperiment_id: '${exp_id}'\nlifecycle_stage: deleted\nrun_id: models\nrun_uuid: models\nrun_name: models\nsource_name: ''\nsource_type: 4\nsource_version: ''\nstart_time: 0\nstatus: 4\ntags: []\nuser_id: mlflow\n" > "$d/meta.yaml"; \
        fi; \
    done && \
    # Corriger TOUS les meta.yaml d'experiments
    for f in /app/mlruns/*/meta.yaml; do \
        if [ -f "$f" ]; then \
            experiment_dir=$(dirname "$f"); \
            exp_id=$(basename "$experiment_dir"); \
            sed -i "s|artifact_location:.*|artifact_location: file:///app/mlruns/${exp_id}|g" "$f" 2>/dev/null || true; \
            echo "✅ Fixed: $f"; \
        fi; \
    done && \
    # Corriger TOUS les meta.yaml des runs
    for f in /app/mlruns/*/*/meta.yaml; do \
        if [ -f "$f" ]; then \
            run_dir=$(dirname "$f"); \
            run_id=$(basename "$run_dir"); \
            exp_dir=$(dirname "$run_dir"); \
            exp_id=$(basename "$exp_dir"); \
            sed -i "s|artifact_uri:.*|artifact_uri: file:///app/mlruns/${exp_id}/${run_id}/artifacts|g" "$f" 2>/dev/null || true; \
            mkdir -p "${run_dir}/artifacts" 2>/dev/null || true; \
            if ! grep -q '^run_uuid:' "$f"; then \
                echo "run_uuid: ${run_id}" >> "$f"; \
            fi; \
            echo "✅ Fixed run: $f"; \
        fi; \
    done && \
    echo "✅ MLflow configuré" && \
    find /app/mlruns -name "meta.yaml" -exec cat {} \; 2>/dev/null | head -20

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
