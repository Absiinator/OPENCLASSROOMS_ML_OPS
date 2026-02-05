# 🏦 Home Credit Scoring - Projet MLOps Complet

[![CI/CD - Tests & Build](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/actions/workflows/ci-cd.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Projet complet de **scoring de crédit** basé sur le dataset [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) de Kaggle. Ce projet met en œuvre les meilleures pratiques **MLOps** pour construire, déployer et monitorer un modèle de Machine Learning en production.

### 🎯 Objectif métier

Prédire la **probabilité de défaut de paiement** d'un client demandant un crédit, en optimisant le coût métier avec :
- **Coût d'un Faux Négatif (FN)** : 10 (accepter un client qui fera défaut)
- **Coût d'un Faux Positif (FP)** : 1 (refuser un bon client)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Ce fichier - Vue d'ensemble du projet |
| [RENDER_SETUP.md](RENDER_SETUP.md) | Guide complet de déploiement sur Render (API, Dashboard, MLflow) |
| [api/README.md](api/README.md) | README spécifique de l’API (objectif, endpoints, artefacts) |
| [streamlit_app/README.md](streamlit_app/README.md) | README spécifique du dashboard Streamlit |
| [mlflow/README.md](mlflow/README.md) | README spécifique du service MLflow |
| [tests/README.md](tests/README.md) | Documentation des tests unitaires et d'intégration |
| [presentation_outline.txt](presentation_outline.txt) | Plan de présentation - Phase 1 (MLOps) |
| [presentation_outline_phase2.txt](presentation_outline_phase2.txt) | Plan de présentation - Phase 2 (Dashboard + Veille) |

### 📓 Notebooks (GitHub)

- [01_EDA.ipynb](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/01_EDA.ipynb)
- [02_Preprocessing_Features.ipynb](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/02_Preprocessing_Features.ipynb)
- [03_Model_Training_MLflow.ipynb](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb)
- [04_Drift_Evidently.ipynb](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/04_Drift_Evidently.ipynb)

## ✅ Conformité CE (preuves dans le repo)

### 1) Stratégie de modélisation
- **CE1** Encodage des variables catégorielles : [Notebook 02](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/02_Preprocessing_Features.ipynb) · [src/preprocessing.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/preprocessing.py)
- **CE2** Feature engineering : [Notebook 02](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/02_Preprocessing_Features.ipynb) · [reports/new_features_correlations.png](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/new_features_correlations.png)
- **CE3** Transformations mathématiques : [Notebook 02](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/02_Preprocessing_Features.ipynb)
- **CE4** Normalisation : [Notebook 02](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/02_Preprocessing_Features.ipynb) · [src/preprocessing.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/preprocessing.py)
- **CE5** Stratégie alignée au besoin métier : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb) · [presentation_outline.txt](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/presentation_outline.txt)
- **CE6** Variable cible pertinente : [Notebook 01](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/01_EDA.ipynb)
- **CE7** Vérification data leakage : [Notebook 01](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/01_EDA.ipynb) · [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb)
- **CE8** Tests d’algorithmes (linéaire & non‑linéaire) : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb)

### 2) Évaluation des performances
- **CE1** Score métier FN/FP + métriques : [src/metrics.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/metrics.py) · [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb) · [reports/metrics_report.txt](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/metrics_report.txt)
- **CE2** Indicateurs complémentaires : [ROC](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/roc_curve.png) · [Confusion](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/confusion_matrix.png)
- **CE3** Split train/test : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb) · [src/train.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/train.py)
- **CE4** Modèle de référence : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb)
- **CE5** Déséquilibre des classes : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb) · [src/train.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/train.py)
- **CE6** Optimisation hyper‑paramètres : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb)
- **CE7** Validation croisée : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb)
- **CE8** Résultats du simple au complexe + choix final : [Notebook 03](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb) · [presentation_outline.txt](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/presentation_outline.txt)
- **CE9** Feature importance globale & locale : [reports/feature_importance.csv](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/feature_importance.csv) · [streamlit_app/app.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/streamlit_app/app.py)

### 3) Pipeline d’entraînement & registry
- **CE1** Pipeline reproductible : [src/train.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/train.py) · [src/preprocessing.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/src/preprocessing.py)
- **CE2** Sérialisation + stockage modèles : [models/model_config.json](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/models/model_config.json) · [notebooks/mlruns](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/tree/main/notebooks/mlruns)
- **CE3** Mesures formalisées : [reports/metrics_report.txt](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/metrics_report.txt) · [notebooks/mlruns](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/tree/main/notebooks/mlruns)

### 4) Versioning du code
- **CE1** Repo Git + GitHub : [repo](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS)
- **CE2** Historique des commits : [commits](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/commits/main)
- **CE3** Packages & versions : [environment.yml](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/environment.yml) · [pyproject.toml](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/pyproject.toml) · [api/requirements.txt](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/api/requirements.txt)
- **CE4** Fichier introductif & structure : [README.md](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/README.md) · [api/README.md](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/api/README.md) · [streamlit_app/README.md](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/streamlit_app/README.md)
- **CE5** Scripts commentés : [src/](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/tree/main/src) · [api/main.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/api/main.py)

### 5) Déploiement continu de l’API
- **CE1** Pipeline CI/CD : [.github/workflows/ci-cd.yml](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/.github/workflows/ci-cd.yml) · [render.yaml](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/render.yaml)
- **CE2** API de prédiction opérationnelle : [api/main.py](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/api/main.py) · [api/README.md](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/api/README.md)
- **CE3** Déploiement cloud : [render.yaml](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/render.yaml) · [RENDER_SETUP.md](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/RENDER_SETUP.md)
- **CE4** Tests unitaires automatisés : [tests/](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/tree/main/tests) · [.github/workflows/ci-cd.yml](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/.github/workflows/ci-cd.yml)
- **CE5** API indépendante du dashboard : [api/](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/tree/main/api) · [streamlit_app/](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/tree/main/streamlit_app)

### 6) Suivi de performance & drift
- **CE1** Stratégie de suivi (data drift) : [Notebook 04](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/04_Drift_Evidently.ipynb)
- **CE2** Simulation + rapport Evidently : [reports/evidently_full_report.html](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/evidently_full_report.html) · [Notebook 04](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/04_Drift_Evidently.ipynb)
- **CE3** Analyse stabilité + actions : [Notebook 04](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/04_Drift_Evidently.ipynb) · [reports/evidently_full_report.html](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/reports/evidently_full_report.html)

## 🏗️ Architecture du projet

```
home-credit-scoring/
├── 📁 api/                     # API FastAPI de scoring (Dockerfile inclus)
├── 📁 mlflow/                  # Service MLflow UI (Dockerfile + README)
├── 📁 streamlit_app/           # Dashboard Streamlit (Dockerfile inclus)
├── 📁 models/                  # Modèles entraînés (trackés, sans Git LFS)
├── 📁 data/                    # Fichiers CSV locaux (optionnels en déploiement)
├── 📁 notebooks/               # Notebooks + tracking MLflow (notebooks/mlruns)
├── 📁 reports/                 # Rapports Evidently + figures
├── 📁 src/                     # Code source (prétraitement, entraînement, metrics)
├── 📁 tests/                   # Tests unitaires
├── render.yaml                 # Blueprint Render (3 services)
├── run.py                      # Lancement local (API/Dashboard/MLflow)
└── README.md                   # Ce fichier
```

## 🚀 Démarrage rapide

### Prérequis

- Python 3.10+
- Conda ou pip
- Docker (optionnel, pour le déploiement)
- Compte Kaggle (pour les données)

### Installation locale

- Python 3.10+ requis
- Dépendances décrites dans `environment.yml`, `pyproject.toml` et `api/requirements.txt`
- Le script `run.py` orchestre les services en local (API, Dashboard, MLflow)

### Données (local vs déploiement)

- En local, les CSV sont attendus dans `data/`
- En déploiement (Docker/Render), les images téléchargent et extraient automatiquement le dataset dans `/app/data`, sans Git LFS

### Entraînement et tracking

- Le notebook `03_Model_Training_MLflow.ipynb` (lien GitHub ci-dessus) logge les runs MLflow dans `notebooks/mlruns/`
- Les modèles exportés sont versionnés dans `models/` et utilisés par l’API pour l’inférence

### Lancement local

`run.py` expose les commandes `train`, `api`, `dashboard`, `mlflow`, `all` (ports par défaut ci‑dessous).

### Ports par défaut

| Service | Port | URL |
|---------|------|-----|
| API FastAPI | 8000 | http://localhost:8000 |
| Dashboard Streamlit | 8501 | http://localhost:8501 |
| MLflow UI | 5002 | http://localhost:5002 |

*En déploiement Docker/Render, MLflow écoute sur le port 5000 (voir `render.yaml`).*

### Lancer avec Docker

Le projet fournit **3 Dockerfiles** (API, Dashboard, MLflow). Chaque image est prête pour le déploiement sur Render (plan gratuit).

#### 1. API (api/Dockerfile)
- **Port** : 8000
- **Contenu** :
  - Modèle LightGBM, préprocesseur et configuration **trackés dans `models/`**
  - Code API FastAPI + modules `src/`
  - Rapports Evidently (`reports/`) pour l’endpoint `/data/drift`
  - **Données téléchargées automatiquement** pendant le build (extraction vers `/app/data`)

#### 2. Dashboard (streamlit_app/Dockerfile)
- **Port** : 8501
- **Variables obligatoires** : `API_URL`, `MLFLOW_URL`
- **Contenu** :
  - Application Streamlit (Scoring + Comparaison intégrée, Data Drift, Documentation)
  - Rapports Evidently dans `reports/`
  - **Données téléchargées automatiquement** pendant le build (extraction vers `/app/data`)

#### 3. MLflow (mlflow/Dockerfile)
- **Port** : 5000
- **Contenu** :
- MLflow UI avec runs copiés depuis `notebooks/mlruns/` (voir dossier sur GitHub)
  - Registry MLflow disponible (lecture seule en production)

> 📝 **Notes** :
> - Les **données sont téléchargées au build** des images API et Dashboard (pas de Git LFS).
> - **MLflow** est configuré pour le plan gratuit (1 worker Gunicorn, mémoire limitée).

## 📊 Résultats du modèle

| Métrique | Valeur |
|----------|--------|
| AUC-ROC | ~0.76 |
| Seuil optimal | ~0.35 |
| Accuracy | ~0.70 |
| Coût métier normalisé | Optimisé |

## 🔧 Fonctionnalités principales

### 1. 📈 Prétraitement avancé

- Agrégation des tables auxiliaires (bureau, previous_application, etc.)
- Feature engineering (ratios financiers, agrégats temporels)
- Gestion des valeurs manquantes
- Encodage des variables catégorielles

### 2. 🧠 Modélisation

- **LightGBM** avec class_weight='balanced'
- Optimisation du seuil de décision via le coût métier
- Cross-validation stratifiée
- Logging complet avec MLflow

### 3. 🔍 Explicabilité

- **SHAP** pour les explications locales et globales
- Feature importance intégrée
- Visualisations interactives

### 4. 📉 Monitoring du drift

- Rapports **Evidently** automatisés
- Détection du data drift et prediction drift
- Alertes sur la dérive des features

### 5. 🌐 API REST (Pydantic v2 compatible)

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/health` | GET | Health check (vérifie que les modèles sont chargés) |
| `/predict` | POST | Prédiction unique - **Supporte 3 formats JSON** |
| `/predict/batch` | POST | Prédictions en batch |
| `/predict/explain` | POST | Prédiction + SHAP |
| `/model/info` | GET | Infos du modèle (seuil, version, features) |
| `/model/features` | GET | Liste des features |

**Format supporté pour `/predict`** :

```json
{
  "features": {
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "DAYS_BIRTH": -18000
  }
}
```

**Notes** :
- L'API charge automatiquement les modèles au démarrage depuis `/app/models/` dans Docker.
- Seul le champ `features` est traité.

### 🏷️ Libellés des variables (Dashboard)

- Le dashboard affiche des **libellés explicites** pour toutes les colonnes (y compris features agrégées/engineered).
- Le mapping UI est défini dans `streamlit_app/app.py` via :
  - `FEATURE_LABELS` + `FEATURE_LABEL_OVERRIDES`
  - `PREFIX_LABELS`, `TOKEN_LABELS`, `STAT_SUFFIXES`
- **Important** : l’API attend toujours les **noms de colonnes d’origine**.  
  Les libellés n’affectent pas le payload envoyé à `/predict`.

### 6. 🔄 CI/CD & Déploiement Render (plan gratuit)

- `render.yaml` décrit les 3 services (API, Dashboard, MLflow)
- Render **construit les images depuis les Dockerfiles** du repo
- `autoDeploy: true` active le déploiement automatique à chaque push
- Le workflow GitHub Actions (présent dans `.github/workflows/ci-cd.yml`) reste **optionnel** : il build/push des images GHCR, mais Render n’en dépend pas

Pour le guide complet de déploiement, consultez [RENDER_SETUP.md](RENDER_SETUP.md).

## 📁 Données

### 📦 Fichiers suivis dans le repo (sans Git LFS)

- `models/` : artefacts nécessaires à l’inférence (API)
- `notebooks/mlruns/` : runs MLflow + registry (lecture seule en prod)
- `reports/` : rapports Evidently utilisés par le dashboard
- `data/` : utile en local, **non requis** en déploiement (download au build)

Le projet utilise les données du challenge Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) :

| Fichier | Description |
|---------|-------------|
| `application_train.csv` | Demandes de crédit (entraînement) avec TARGET |
| `application_test.csv` | Demandes de crédit (test) |
| `bureau.csv` | Crédits antérieurs chez d'autres institutions |
| `bureau_balance.csv` | Historique mensuel des crédits bureau |
| `previous_application.csv` | Demandes antérieures chez Home Credit |
| `POS_CASH_balance.csv` | Historique des prêts point de vente |
| `credit_card_balance.csv` | Historique des cartes de crédit |
| `installments_payments.csv` | Historique des paiements |

### Pipeline de Traitement des Données

Le modèle a été entraîné sur **245+ features engineered**, mais l'API accepte des requêtes avec seulement **17 features brutes**. La transformation est **automatique** :

```
Dashboard/API (17 features)
    ↓
create_application_features()  [ratios, moyennes, conversions]
    ↓
CreditScoringPreprocessor.transform() [imputation, encoding]
    ↓
LightGBM Model (245 features)
```

**17 Features requises :**
- **Finances** : `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`
- **Temporel** : `DAYS_BIRTH`, `DAYS_EMPLOYED`
- **Personnel** : `CNT_CHILDREN`, `CODE_GENDER_M`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`
- **Scores** : `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `REGION_RATING_CLIENT`
- **Ratios** : `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_MEAN`

**Gestion automatique :**
- ✅ Features engineered ajoutées dynamiquement
- ✅ ~200 colonnes d'agrégation imputées avec la médiane apprises lors de l'entraînement
- ✅ Encodage des catégorielles
- ✅ Aucune configuration manuelle nécessaire

## 🧪 Tests

La structure et les conventions de tests sont décrites dans `tests/README.md`.
Les tests sont conçus pour être rapides en CI/CD et couvrent le coût métier,
le prétraitement et l’API (pas de tests de déploiement Render).

## � Versions Critiques - Pydantic v2

### Compatibilité Pydantic v2

L'API utilise **Pydantic v2.5+** avec un format unique pour les requêtes :

```python
# api/models.py
from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features du client")
```

**Pourquoi cette approche ?**
- ✅ Schéma OpenAPI simple et explicite
- ✅ Évite les erreurs de format côté client
- ✅ Compatible avec le dashboard Streamlit

### Table de versions

| Dépendance | Version | Raison |
|-----------|---------|--------|
| **Pydantic** | >=2.5.0,<3.0.0 | Compatibilité ConfigDict + Optional fields |
| **FastAPI** | >=0.104.0,<0.116.0 | Compatibilité Pydantic v2.5+ |
| **MLflow** | 2.9.2 | Léger (~50MB) vs versions récentes (~200MB+) |
| **Python** | 3.10+ | tomli conditionnel pour pyproject.toml (Python < 3.11) |

⚠️ **Si vous updatez ces versions, testez localement d'abord !** Les changements Pydantic v3 pourraient casser la validation.

## 🔁 CI/CD et Déploiement

### CI/CD (optionnel)

Le workflow GitHub Actions (`.github/workflows/ci-cd.yml`) exécute :
- **Lint** (black, isort, flake8)
- **Tests unitaires** (pytest)
- **Build d’images Docker** (API, Dashboard, MLflow) et push vers GHCR

⚠️ Render n’a pas besoin de GHCR si vous utilisez `render.yaml` : il build directement depuis le repo.

### Configuration Render (render.yaml)

Le fichier `render.yaml` décrit **3 services Docker** en plan gratuit :

| Service | Nom par défaut | Port | Health Check | Variables clés |
|---------|----------------|------|--------------|----------------|
| **API** | `home-scoring-api` | 8000 | `/health` | `PORT`, `PYTHONPATH`, `HOST` |
| **Dashboard** | `home-scoring-dashboard` | 8501 | `/_stcore/health` | `PORT`, `API_URL`, `MLFLOW_URL` |
| **MLflow** | `home-scoring-mlflow` | 5000 | `/` | `PORT` |

**Point clé** : `API_URL` et `MLFLOW_URL` doivent correspondre aux URLs réelles des services Render.  
Si vous renommez les services, adaptez ces variables dans `render.yaml`.

### Variables d'environnement (référence)

| Variable | Service | Description | Valeur par défaut |
|----------|---------|-------------|-------------------|
| `HOST` | API | Host d’écoute | `0.0.0.0` |
| `PORT` | API/Dashboard/MLflow | Port d’écoute | 8000 / 8501 / 5000 |
| `PYTHONPATH` | API | Chemin Python | `/app` |
| `API_URL` | Dashboard | URL de l’API | URL Render de l’API |
| `MLFLOW_URL` | Dashboard | URL MLflow UI | URL Render MLflow |
| `STREAMLIT_SERVER_ADDRESS` | Dashboard | Adresse Streamlit | `0.0.0.0` |
| `STREAMLIT_SERVER_PORT` | Dashboard | Port Streamlit | `8501` |
| `MLFLOW_TRACKING_URI` | MLflow | Backend store | `/app/mlruns` |

## 📖 Documentation API

La documentation interactive est disponible via :
- **Swagger UI** : `http://localhost:8000/docs` - Tests des endpoints directement
- **Swagger** : `http://localhost:8000/docs` - Documentation interactive

### Exemple de payload (17 features minimal)

```json
{
  "features": {
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 500000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -5000,
    "CNT_CHILDREN": 1,
    "CODE_GENDER_M": 1,
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 1,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.55,
    "REGION_RATING_CLIENT": 2,
    "CREDIT_INCOME_RATIO": 3.33,
    "ANNUITY_INCOME_RATIO": 0.167,
    "EXT_SOURCE_MEAN": 0.55
  }
}
```

### Exemple de réponse

```json
{
  "client_id": null,
  "probability": 0.23,
  "prediction": 0,
  "decision": "ACCEPTED",
  "risk_category": "low",
  "threshold": 0.44
}
```

### Notes importantes

- ✅ **L'API accepte 17+ features** - Toutes les colonnes supplémentaires sont ignorées (mode `extra="allow"`)
- ✅ **Colonnes manquantes comblées automatiquement** - Les ~200 colonnes d'agrégation sont imputées avec la médiane
- ✅ **Feature engineering automatique** - Ratios, moyennes et conversions créés automatiquement
- ✅ **Format JSON unique** - Accepte uniquement `{"features": {...}}`
- ⚠️ **Seuil par défaut : 0.44** - Optimisé pour minimiser le coût métier (FN=10, FP=1)

## 🐛 Problèmes Courants

### Erreur 422 "Field required" sur `/predict`

**Cause** : Incompatibilité Pydantic v2 avec les champs optionnels mal configurés

**Solution** : Vérifiez que vous utilisez `Pydantic>=2.5.0` et envoyez le JSON avec le format correct :

```json
{"features": {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000, ...}}
```

Consulter [Versions Critiques - Pydantic v2](#-versions-critiques---pydantic-v2) pour les détails.

### MLflow crashing avec "Out of Memory" ou "SIGKILL" sur Render

**Cause** : trop de workers Gunicorn ou dépendances lourdes sur un plan 512MB.

**Solution** : le Dockerfile utilise **`mlflow server` avec 1 worker** + dépendances minimales.

**Vérification** : voir [mlflow/Dockerfile](mlflow/Dockerfile) et [mlflow/README.md](mlflow/README.md)

| Configuration | RAM | Status |
|---------------|-----|--------|
| **mlflow server --workers=1** (actuel) | ~200-250 MB | ✅ Fonctionne |
| mlflow server (défaut 4 workers) | ~400-500 MB | ❌ CRASH |

### Dashboard ne peut pas se connecter à l'API

**Cause** : Variables d'environnement `API_URL` ou `MLFLOW_URL` non configurées

**Solution (Render)** :
1. Allez sur le service **home-scoring-dashboard**
2. **Environment** → Ajouter/modifier :
   - `API_URL=https://home-scoring-api.onrender.com`
   - `MLFLOW_URL=https://home-scoring-mlflow.onrender.com`
3. Redémarrer le service (Deploy → Select Commit → Deploy)

**Solution (Local)** :
```bash
export API_URL=http://localhost:8000
export MLFLOW_URL=http://localhost:5000
streamlit run streamlit_app/app.py
```

## 🤝 Contribution

Les contributions sont les bienvenues :

1. Forker le repository
2. Créer une branche dédiée
3. Proposer les changements via Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

**Réalisé dans le cadre du projet OpenClassrooms "Réalisez un dashboard et assurez une veille technique"**
