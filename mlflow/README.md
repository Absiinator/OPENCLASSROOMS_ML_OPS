# MLflow UI Server

Service de suivi des expérimentations Machine Learning avec MLflow.

## 📋 Description

Ce conteneur Docker déploie une interface MLflow UI pour visualiser et comparer les expérimentations ML du projet Home Credit Scoring.

## 🚀 Déploiement

### Local

```bash
# Depuis la racine du projet
docker build -f mlflow/Dockerfile -t home-credit-mlflow .
docker run -p 5000:5000 home-credit-mlflow
```

Accéder à : http://localhost:5000

### Production (Render)

Le déploiement est automatique via GitHub Actions (`.github/workflows/deploy.yml`).

L'image est construite et poussée vers GHCR, puis déployée sur Render.

## 📊 Contenu

### Runs MLflow

Le conteneur inclut les runs MLflow du projet :
- `mlruns/` : Runs du projet principal
- Métriques : AUC, F1, Precision, Recall, Business Cost
- Artefacts : Modèles, rapports, graphiques

### Configuration

Variables d'environnement :
- `PORT` : Port d'écoute (défaut: 5000)
- `MLFLOW_TRACKING_URI` : Backend store (`/app/mlruns`)
- `MLFLOW_BACKEND_STORE_URI` : Alias du tracking URI

**Configuration Render (tier gratuit)** :
- 1 worker (au lieu de 4) pour économiser la RAM (512MB disponibles)
- Timeout augmenté à 120s pour éviter les WORKER TIMEOUT
- Dépendances minimales (pas de boto3/psycopg2)

## 🔍 Fonctionnalités

L'interface MLflow UI permet de :
- 📊 Visualiser les métriques d'entraînement
- 🔍 Comparer les différents runs
- 📈 Tracer les courbes d'apprentissage
- 📦 Gérer les versions de modèles
- 📥 Télécharger les artefacts

## 🛠️ Dépendances

Voir [requirements.txt](requirements.txt) :
- `mlflow==2.9.2` : Framework MLflow (version légère, sans boto3/psycopg2 pour économiser la RAM)

## ⚡ Optimisations pour le Tier Gratuit Render (512MB RAM)

### Stratégie d'optimisation

Le Dockerfile utilise **`mlflow server --gunicorn-opts "--workers=1"`** pour forcer un seul worker :

| Configuration | Consommation RAM | Détail |
|---------------|-----------------|--------|
| **mlflow server --workers=1** (actuel) | ~200-250 MB | 1 seul worker Gunicorn |
| mlflow server (défaut 4 workers) | ~400-500 MB | **CRASH - dépassement RAM** |
| mlflow ui | Variable | Peut encore utiliser Gunicorn en interne |

**Clé du succès** : `--gunicorn-opts "--workers=1 --threads=2 --timeout=120"`

### Configuration appliquée

```dockerfile
# Dockerfile: forcer 1 seul worker pour économiser la RAM
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri /app/mlruns \
    --serve-artifacts \
    --gunicorn-opts "--workers=1 --threads=2 --timeout=120"
```

**Paramètres critiques** :
- `--workers=1` : UN seul processus worker (vs 4 par défaut)
- `--threads=2` : 2 threads par worker pour gérer les requêtes
- `--timeout=120` : 2 minutes pour éviter WORKER TIMEOUT

## 📝 Notes

- Les runs MLflow du dossier `mlruns/` local sont copiés dans l'image Docker lors du build
- **Tier gratuit Render** : 512MB RAM, service arrêté après 15 min d'inactivité
- **Optimisations appliquées** :
  - ✅ 1 seul worker Gunicorn (économise ~200-300MB)
  - ✅ Timeout augmenté à 120s (évite WORKER TIMEOUT)
  - ✅ Dépendances minimales (mlflow v2.9.2)
  - ✅ Variables d'environnement `MALLOC_ARENA_MAX=2` pour limiter la mémoire
- Les runs sont accessibles en **lecture seule** - les nouvelles expériences ne seront pas persistées (tier gratuit)

## 🔧 Dépannage

### "Out of Memory" ou "SIGKILL"

**Si vous voyez ces erreurs en production** :
1. Vérifiez que le Dockerfile utilise `--workers=1` (pas le défaut de 4)
2. Vérifiez la RAM allouée (512MB = limite du tier gratuit)
3. Attendez 1-2 min au démarrage (premier chargement est lent)

**Solution** : Upgrade vers un plan payant si vous avez vraiment besoin de multiple workers.
