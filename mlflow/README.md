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

## 📝 Notes

- Les runs MLflow du dossier `mlruns/` local sont copiés dans l'image Docker lors du build
- **Tier gratuit Render** : 512MB RAM, service arrêté après 15 min d'inactivité
- **Optimisations appliquées** : 1 worker, timeout 120s, dépendances minimales
- Les runs sont accessibles en **lecture seule** - les nouvelles expériences ne seront pas persistées (tier gratuit)
