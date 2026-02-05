# Guide de Configuration Render pour Déploiement

Ce guide explique le déploiement des 3 services sur Render.

## 🏗️ Architecture CI/CD

**Render (recommandé, plan gratuit)** :
- `render.yaml` décrit les 3 services et build **directement depuis les Dockerfiles** du repo
- Déploiement automatique possible via `autoDeploy: true`

**GitHub Actions (optionnel)** :
- Lint + tests + build d’images Docker
- Push vers GHCR si vous souhaitez un registry externe

## 📋 Prérequis

1. **Compte Render** : Créez un compte gratuit sur [render.com](https://render.com)
2. **Compte GitHub** : Votre repo doit être sur GitHub (déjà fait ✅)
3. **Dockerfiles** : Les images sont construites par Render à partir du repo

## 🏗️ Architecture Docker

Le projet utilise 3 Dockerfiles distincts pour les 3 services :

### API (`api/Dockerfile`)

- **Base** : `python:3.10-slim`
- **Port** : 8000
- **Contenu** :
  - Code source (`src/`, `api/`)
  - ✅ **Modèles pré-entraînés inclus** (`models/lgbm_model.joblib`, `preprocessor.joblib`, `model_config.json`)
  - ✅ **Données téléchargées automatiquement** depuis S3 OpenClassrooms lors du build
  - Dépendances Python pour FastAPI, LightGBM, SHAP
- **Téléchargement des données** : Le Dockerfile télécharge et décompresse automatiquement les données depuis :
  ```
  https://s3-eu-west-1.amazonaws.com/static.oc-static.com/.../home-credit-default-risk.zip
  ```
- **Health check** : `/health` (vérifie que les modèles sont chargés)
- **Commande** : `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Dashboard (`streamlit_app/Dockerfile`)

- **Base** : `python:3.10-slim`
- **Port** : 8501
- **Contenu** :
  - App Streamlit (`app.py`) : Scoring avec comparaison intégrée, Data Drift, Documentation
  - Sources (`src/`)
  - ✅ **Données téléchargées automatiquement** depuis S3 OpenClassrooms lors du build
- **Health check** : `/_stcore/health`
- **Commande** : `streamlit run app.py --server.port=$PORT`

### MLflow (`mlflow/Dockerfile`)

- **Base** : `python:3.10-slim`
- **Port** : 5000
- **Contenu** : Répertoire `notebooks/mlruns/` copié dans l'image avec correction automatique des chemins
- **Commande** : `mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri file:///app/mlruns`

⚠️ **Notes importantes** :

- Les **données sont téléchargées automatiquement** lors du build Docker (~500MB)
- **MLflow** : Les runs/experiments sont copiés en lecture seule depuis `notebooks/mlruns/`

## 🔧 Variables d'environnement

### Injectées par Render (via render.yaml)

| Service       | Variable     | Valeur injectée par Render                        |
| ------------- | ------------ | ------------------------------------------------- |
| **API**       | `PORT`       | Automatique (Render)                              |
| **Dashboard** | `PORT`       | Automatique (Render)                              |
| **Dashboard** | `API_URL`    | `https://home-credit-scoring-api.onrender.com`    |
| **Dashboard** | `MLFLOW_URL` | `https://home-credit-scoring-mlflow.onrender.com` |
| **Dashboard** | `GITHUB_REPO_URL` | `https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS` |
| **MLflow**    | `PORT`       | Automatique (Render)                              |

> 💡 `API_URL`, `MLFLOW_URL` et `PORT` sont définies dans `render.yaml` et écrasent les valeurs par défaut.  
> `GITHUB_REPO_URL` est **optionnelle** : ajoutez-la si vous souhaitez afficher un autre repo que le défaut.

---

## 🚀 Déploiement avec Blueprint (render.yaml)

1. Créer un **Blueprint** dans Render et sélectionner le repo
2. Render lit `render.yaml` et crée **3 services** en plan gratuit
3. Vérifier que les variables `API_URL` et `MLFLOW_URL` correspondent aux **Live URLs** Render

### Lien du repo (où le renseigner ?)

- **Si vous utilisez le Blueprint** : vous liez le **repo une seule fois** au moment de créer le Blueprint.  
  Les 3 services héritent automatiquement du même repo.
- **Si vous créez les services manuellement** : vous devez lier **le même repo** à **chaque service** (API, Dashboard, MLflow).

### Variables d’environnement à vérifier

- **Dashboard** : `API_URL`, `MLFLOW_URL` (doivent viser les URLs Render)
- **Dashboard** : `GITHUB_REPO_URL` (optionnelle, pour afficher le lien GitHub dans l’interface)
- **API / MLflow** : `PORT` est déjà défini dans `render.yaml`

### Notes plan gratuit

- Démarrage à froid après inactivité (~30-60s)
- 512MB de RAM par service
- MLflow configuré avec 1 worker Gunicorn (voir `mlflow/Dockerfile`)

## ✅ Vérifications après déploiement

- **API** : `/health` doit retourner `healthy` et `model_loaded=true`
- **Dashboard** : l’interface doit s’ouvrir et appeler l’API sans erreur
- **MLflow** : l’UI doit afficher les runs existants

## 🎯 URLs Finales

Une fois déployé, notez vos URLs :

```bash
# API
https://home-credit-api.onrender.com

# Dashboard
https://home-credit-dashboard.onrender.com

# MLflow UI
https://home-credit-mlflow.onrender.com

# Documentation API
https://home-credit-api.onrender.com/docs
```

## 🔗 Récapitulatif des Variables d'Environnement

### Variables à configurer sur Render

| Service             | Variable       | Valeur                                | Obligatoire ?                    |
| ------------------- | -------------- | ------------------------------------- | -------------------------------- |
| **API**       | `PORT`       | Défini automatiquement par Render    | ❌ Non                           |
| **API**       | `HOST`       | `0.0.0.0`                           | ❌ Non (défini dans Dockerfile) |
| **Dashboard** | `PORT`       | Défini automatiquement par Render    | ❌ Non                           |
| **Dashboard** | `API_URL`    | `https://votre-api.onrender.com`    | ✅**OUI**                  |
| **Dashboard** | `MLFLOW_URL` | `https://votre-mlflow.onrender.com` | ✅**OUI**                  |
| **Dashboard** | `GITHUB_REPO_URL` | `https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS` | ❌ Non |
| **MLflow**    | `PORT`       | Défini automatiquement par Render    | ❌ Non                           |

## 📝 Notes Importantes

### ⚠️ Versions Critiques (à respecter)

| Dépendance | Version | Raison |
|-----------|---------|--------|
| **Pydantic** | >=2.5.0,<3.0.0 | Compatibilité Optional fields + Pydantic v2 ConfigDict |
| **FastAPI** | >=0.104.0,<0.116.0 | Compatibilité avec Pydantic v2.5+ |
| **MLflow** | 2.9.2 | Léger (~50MB) vs versions récentes (~200MB+) |

⚠️ **Si vous updatez ces versions, testez localement d'abord !**

- Les changements Pydantic v3 pourraient casser la validation API (erreur 422)
- Les versions FastAPI incompatibles pourraient casser la sérialisation JSON
- Les versions MLflow plus récentes consomment plus de RAM

Consultez [README.md - Versions Critiques](README.md#--versions-critiques---pydantic-v2) pour plus de détails.

### ⚠️ Limitations du Plan Gratuit

- **Sleep après 15 min d'inactivité** : Premier appel prend ~30-60s
- **750h/mois** par service gratuit
- **Pas de custom domain** sur le plan gratuit

### 🔄 Workflow de déploiement (render.yaml)

- À chaque push, Render rebuild et redéploie les services si `autoDeploy: true`
- En plan gratuit, le premier démarrage peut être lent (cold start)

### 🐛 Dépannage

**Déploiement échoué**
- Vérifier les logs Render (build + runtime)
- Confirmer que le téléchargement des données réussit pendant le build

**Dashboard ne joint pas l’API**
- Vérifier `API_URL` dans le service Dashboard
- Vérifier que l’API répond sur `/health`

**"Model not loaded"**
- Vérifier que `models/` est présent dans le repo
- Vérifier que l’API charge bien `/app/models/`

**MLflow instable**
- Attendre 1-2 minutes au premier démarrage
- Vérifier que le Dockerfile utilise 1 worker Gunicorn

---

## ✅ Checklist Finale

- [ ] Blueprint Render créé depuis le repo
- [ ] Services actifs : API, Dashboard, MLflow
- [ ] API `/health` retourne `healthy`
- [ ] Dashboard affiche un score via l’API
- [ ] MLflow UI accessible
