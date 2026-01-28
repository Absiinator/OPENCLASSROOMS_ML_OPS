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

## 🔍 Fonctionnalités

L'interface MLflow UI permet de :
- 📊 Visualiser les métriques d'entraînement
- 🔍 Comparer les différents runs
- 📈 Tracer les courbes d'apprentissage
- 📦 Gérer les versions de modèles
- 📥 Télécharger les artefacts

## 🛠️ Dépendances

Voir [requirements.txt](requirements.txt) :
- `mlflow==2.9.2` : Framework MLflow

## 📝 Notes

- Les données sont persistées dans le conteneur (`/app/mlruns`)
- Le plan gratuit Render arrête les services après 15 min d'inactivité
