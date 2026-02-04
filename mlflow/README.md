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

Le Dockerfile utilise **`mlflow ui`** (Flask simple) au lieu de **`mlflow server`** (Gunicorn avec multiple workers):

| Configuration | Consommation RAM | Détail |
|---------------|-----------------|--------|
| **mlflow ui** (actuel) | ~150-200 MB | Flask simple, pas de Gunicorn |
| mlflow server --workers 1 | ~250-300 MB | Gunicorn + 1 worker = encore trop lourd |
| mlflow server (défaut) | ~400-500 MB | Gunicorn + 4 workers = **dépassement RAM** |

**Bénéfice** : mlflow ui tient facilement dans les 512MB du tier gratuit sans crashes.

### Configuration appliquée

```dockerfile
# Dockerfile: utilisation de mlflow ui (ultra-léger)
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "${PORT}", "--backend-store-uri", "/app/mlruns"]
```

Aucune configuration Gunicorn nécessaire (mlflow ui utilise Flask directement).

## 📝 Notes

- Les runs MLflow du dossier `mlruns/` local sont copiés dans l'image Docker lors du build
- **Tier gratuit Render** : 512MB RAM, service arrêté après 15 min d'inactivité
- **Optimisations appliquées** :
  - ✅ `mlflow ui` au lieu de `mlflow server` (économise ~100-150MB)
  - ✅ Dépendances minimales (mlflow v2.9.2 sans extras)
  - ✅ Pas de workers multiples ou timeouts problématiques
- Les runs sont accessibles en **lecture seule** - les nouvelles expériences ne seront pas persistées (tier gratuit)

## 🔧 Dépannage

### "Out of Memory" ou "SIGKILL"

**Si vous voyez ces erreurs en production** :
1. Vérifiez que le Dockerfile utilise `mlflow ui` (pas `mlflow server --workers N`)
2. Vérifiez la RAM allouée (512MB = limite du tier gratuit)
3. Attendez 1-2 min au démarrage (premier chargement est lent)

**Solution** : Upgrade vers un plan payant si vous avez vraiment besoin de multiple workers.
