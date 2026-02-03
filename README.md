# 🏦 Home Credit Scoring - Projet MLOps Complet

[![CI - Tests & Linting](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/actions/workflows/ci.yml/badge.svg)](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/actions/workflows/ci.yml)
[![CD - Deploy](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/actions/workflows/deploy.yml/badge.svg)](https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/actions/workflows/deploy.yml)
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
| [tests/README.md](tests/README.md) | Documentation des tests unitaires et d'intégration |

## 🏗️ Architecture du projet

```
home-credit-scoring/
├── 📁 api/                     # API FastAPI de scoring
│   ├── main.py                 # Endpoints REST
│   ├── models.py               # Schémas Pydantic
│   ├── requirements.txt        # Dépendances API
│   └── Dockerfile              # Containerisation
├── 📁 data/                    # Données (non versionnées)
│   ├── raw/                    # Données brutes
│   └── processed/              # Données prétraitées
├── 📁 models/                  # Modèles entraînés
├── 📁 notebooks/               # Analyses et expérimentations
│   ├── 01_EDA.ipynb           # Analyse exploratoire
│   ├── 02_Preprocessing_Features.ipynb
│   ├── 03_Model_Training_MLflow.ipynb
│   └── 04_Drift_Evidently.ipynb
├── 📁 reports/                 # Rapports générés
│   ├── figures/                # Visualisations
│   └── drift/                  # Rapports Evidently
├── 📁 scripts/                 # Scripts utilitaires
│   └── download_data.py        # Téléchargement Kaggle
├── 📁 src/                     # Code source principal
│   ├── __init__.py
│   ├── preprocessing.py        # Pipeline de prétraitement
│   ├── train.py               # Entraînement avec MLflow
│   ├── inference.py           # Prédictions
│   ├── metrics.py             # Métriques et coût métier
│   └── feature_importance.py   # Explications SHAP
├── 📁 streamlit_app/          # Interface utilisateur
│   ├── app.py                 # Application Streamlit
│   └── requirements.txt
├── 📁 tests/                   # Tests unitaires
│   ├── test_cost.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── 📁 .github/workflows/       # CI/CD
│   ├── ci.yml                 # Intégration continue
│   └── deploy.yml             # Déploiement continu
├── environment.yml             # Environnement Conda
├── pyproject.toml             # Configuration projet
├── setup.py                   # Installation
└── README.md                  # Ce fichier
```

## 🚀 Démarrage rapide

### Prérequis

- Python 3.10+
- Conda ou pip
- Docker (optionnel, pour le déploiement)
- Compte Kaggle (pour les données)

### Installation

```bash
# Cloner le repository
git clone https://github.com/username/home-credit-scoring.git
cd home-credit-scoring

# Créer l'environnement conda
conda env create -f environment.yml
conda activate home-credit

# Ou avec pip
pip install -e .
```

### Télécharger les données

```bash
# Configurer les credentials Kaggle
# Créer ~/.kaggle/kaggle.json avec votre API key

# Télécharger les données
python scripts/download_data.py
```

### Entraîner le modèle

```bash
# Avec MLflow tracking
python -c "from src.train import train_with_mlflow; train_with_mlflow()"

# Voir les expériences MLflow (port 5002 car 5000/5001 utilisés par AirPlay sur macOS)
python run.py mlflow
# Ouvre http://localhost:5002
```

### Lancer la stack complète (recommandé)

```bash
# Méthode 1: Script unifié (API + Dashboard)
python run.py all

# Méthode 2: Services séparés
python run.py api        # API sur http://localhost:8000
python run.py dashboard  # Dashboard sur http://localhost:8501
python run.py mlflow     # MLflow UI sur http://localhost:5002
```

### Ports par défaut

| Service | Port | URL |
|---------|------|-----|
| API FastAPI | 8000 | http://localhost:8000 |
| Dashboard Streamlit | 8501 | http://localhost:8501 |
| MLflow UI | 5002 | http://localhost:5002 |

### Lancer avec Docker

Le projet utilise **3 Dockerfiles distincts** pour chaque service :

#### 1. API (api/Dockerfile)
```bash
docker build -t home-credit-api -f api/Dockerfile .
docker run -p 8000:8000 home-credit-api
```
- **Port** : 8000
- **Base** : python:3.10-slim
- **Contient** : 
  - ✅ Modèle LightGBM (`models/lgbm_model.joblib`) - **inclus dans l'image**
  - ✅ Preprocessor (`models/preprocessor.joblib`) - **inclus dans l'image**
  - ✅ Configuration du modèle (`models/model_config.json`)
  - ✅ **Données téléchargées automatiquement** depuis S3 OpenClassrooms lors du build
  - Code API FastAPI
  - Code source (`src/`, `api/`)

#### 2. Dashboard (streamlit_app/Dockerfile)
```bash
docker build -t home-credit-dashboard -f streamlit_app/Dockerfile .
docker run -p 8501:8501 \
  -e API_URL=https://votre-api.onrender.com \
  -e MLFLOW_URL=https://votre-mlflow.onrender.com \
  home-credit-dashboard
```
- **Port** : 8501
- **Base** : python:3.10-slim
- **Variables obligatoires** : `API_URL` (API FastAPI), `MLFLOW_URL` (Interface MLflow)
- **Contient** : 
  - Application Streamlit avec 5 onglets (Scoring, Comparaison, Import/Simulation, Drift, Documentation)
  - Modèles pour fallback local si l'API est indisponible
  - ✅ **Données téléchargées automatiquement** depuis S3 OpenClassrooms lors du build
  - **Barre latérale enrichie** : Navigation, État des services, Infos modèle, **Statistiques descriptives du dataset**

#### 3. MLflow (mlflow/Dockerfile)
```bash
docker build -t home-credit-mlflow -f mlflow/Dockerfile .
docker run -p 5000:5000 home-credit-mlflow
```
- **Port** : 5000
- **Base** : python:3.10-slim
- **Contient** : MLflow UI avec les runs d'expérimentation (mlruns/ copié lors du build)

> 📝 **Notes** : 
> - Les **données sont téléchargées automatiquement** depuis le bucket S3 OpenClassrooms lors du build Docker (pas de COPY local).
> - **MLflow** : Les runs existants dans `mlruns/` sont copiés dans l'image Docker et accessibles en lecture seule sur Render. Nouvelles expériences non persistantes (tier gratuit).

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

### 5. 🌐 API REST

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/health` | GET | Health check (vérifie que les modèles sont chargés) |
| `/predict` | POST | Prédiction unique |
| `/predict/batch` | POST | Prédictions en batch |
| `/predict/explain` | POST | Prédiction + SHAP |
| `/model/info` | GET | Infos du modèle (seuil, version, features) |
| `/model/features` | GET | Liste des features |

**Note** : L'API charge automatiquement les modèles au démarrage depuis `/app/models/` dans Docker.

### 6. 🔄 CI/CD

- Tests automatisés sur chaque PR (**les tests doivent passer avant le déploiement**)
- Linting et formatage du code
- **Build Docker automatique** des 3 services (API, Dashboard, MLflow)
- **Push des images vers GitHub Container Registry (GHCR)**
- **Déploiement MANUEL sur Render** (tier gratuit - Manual Deploy)

> ⚠️ **Important** : 
> - Le workflow CI/CD **build automatiquement** les images Docker après chaque push sur `main`
> - Les images sont poussées vers GHCR et sont prêtes à être déployées
> - Le **déploiement sur Render est MANUEL** via le bouton "Manual Deploy" (tier gratuit)
> - Le workflow ne s'exécute que si tous les tests CI réussissent

Pour le guide complet de déploiement, consultez [RENDER_SETUP.md](RENDER_SETUP.md).

## 📁 Données

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

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=src --cov=api --cov-report=html

# Tests spécifiques
pytest tests/test_cost.py -v        # Tests coût métier
pytest tests/test_preprocessing.py -v  # Tests prétraitement
pytest tests/test_api.py -v         # Tests API
```

**Note** : Tests simples et rapides en CI/CD - Aucun test de déploiement (Render testé manuellement).

## 🔁 CI/CD et Déploiement

### Architecture CI/CD

Le projet utilise un **workflow GitHub Actions unifié** ([ci-cd.yml](.github/workflows/ci-cd.yml)) :

1. **Lint** - Vérification du code (non bloquant)
   - black, isort, flake8

2. **Test** - Tests unitaires (BLOQUANT)
   - pytest avec couverture
   - Tests API

3. **Build & Push** - Publication des images Docker
   - **S'exécute uniquement si les tests passent**
   - Build des 3 images Docker (API, Dashboard, MLflow)
   - Push vers GitHub Container Registry (GHCR)

4. **Summary** - Résumé du déploiement
   - Instructions pour le déploiement manuel sur Render

### Flux de déploiement

```
Push sur main → Tests → ✅ Succès → Build Docker → Push GHCR → Manual Deploy Render
                      → ❌ Échec → Arrêt (pas de build)
```

### Configuration Render (render.yaml)

Le fichier `render.yaml` définit les 3 services avec Blueprint :

| Service | Port | Health Check | Variables |
|---------|------|--------------|-----------|
| **API** | 8000 | `/health` | `PORT=8000` |
| **Dashboard** | 8501 | `/_stcore/health` | `API_URL`, `MLFLOW_URL` |
| **MLflow** | 5000 | `/` | `PORT=5000` |

#### Déploiement avec Blueprint

1. Allez sur [dashboard.render.com](https://dashboard.render.com)
2. Cliquez **New** → **Blueprint**
3. Connectez votre repo GitHub
4. Render détecte automatiquement `render.yaml`
5. Les 3 services sont créés automatiquement

#### Variables d'environnement Dashboard (à configurer après déploiement)

```bash
API_URL=https://home-credit-scoring-api.onrender.com
MLFLOW_URL=https://home-credit-scoring-mlflow.onrender.com
```

> ⚠️ **Important** : Après le premier déploiement, mettez à jour `API_URL` et `MLFLOW_URL` avec les vraies URLs de vos services Render.

### Secrets GitHub requis

Aucun secret supplémentaire n'est nécessaire. Le workflow utilise `GITHUB_TOKEN` automatique pour publier sur GHCR.

### Variables d'environnement

| Variable | Service | Description | Défaut |
|----------|---------|-------------|--------|
| `PORT` | API/Dashboard | Port d'écoute | 8000 / 8501 |
| `API_URL` | Dashboard | URL de l'API | `http://localhost:8000` |
| `MODEL_PATH` | API | Chemin du modèle | `./models/lgbm_model.joblib` |
| `THRESHOLD` | API | Seuil de décision | `0.44` |

## 📖 Documentation API

La documentation interactive est disponible via :
- **Swagger UI** : `http://localhost:8000/docs` - Tests des endpoints directement
- **ReDoc** : `http://localhost:8000/redoc` - Documentation complète

### Exemple de requête (17 features minimal)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
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
- ✅ **Format du JSON flexible** - Accepte `{"features": {...}}`, `{"data": {...}}` ou format plat
- ⚠️ **Seuil par défaut : 0.44** - Optimisé pour minimiser le coût métier (FN=10, FP=1)

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :

1. Forker le repository
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commiter vos changements (`git commit -m 'Add amazing feature'`)
4. Pusher la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

**Réalisé dans le cadre du projet OpenClassrooms "Réalisez un dashboard et assurez une veille technique"**
