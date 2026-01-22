# 🏦 Home Credit Scoring - Projet MLOps Complet

[![CI - Tests & Linting](https://github.com/username/home-credit-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/username/home-credit-scoring/actions/workflows/ci.yml)
[![CD - Deploy](https://github.com/username/home-credit-scoring/actions/workflows/deploy.yml/badge.svg)](https://github.com/username/home-credit-scoring/actions/workflows/deploy.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Projet complet de **scoring de crédit** basé sur le dataset [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) de Kaggle. Ce projet met en œuvre les meilleures pratiques **MLOps** pour construire, déployer et monitorer un modèle de Machine Learning en production.

### 🎯 Objectif métier

Prédire la **probabilité de défaut de paiement** d'un client demandant un crédit, en optimisant le coût métier avec :
- **Coût d'un Faux Négatif (FN)** : 10 (accepter un client qui fera défaut)
- **Coût d'un Faux Positif (FP)** : 1 (refuser un bon client)

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

```bash
# Build et run API
docker build -t home-credit-api -f api/Dockerfile .
docker run -p 8000:8000 home-credit-api

# Build et run Dashboard
docker build -t home-credit-dashboard -f streamlit_app/Dockerfile .
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 home-credit-dashboard
```

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
| `/health` | GET | Health check |
| `/predict` | POST | Prédiction unique |
| `/predict/batch` | POST | Prédictions en batch |
| `/predict/explain` | POST | Prédiction + SHAP |
| `/model/info` | GET | Infos du modèle |
| `/model/features` | GET | Liste des features |

### 6. 🔄 CI/CD

- Tests automatisés sur chaque PR
- Linting et formatage du code
- Build Docker automatique
- Déploiement sur Render/Railway

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

## 🔁 CI/CD et Déploiement

### Architecture CI/CD

Le projet utilise **2 workflows GitHub Actions séparés** pour la maintenabilité :

1. **CI (`ci.yml`)** - Intégration Continue
   - Linting (black, isort, flake8)
   - Tests unitaires (pytest)
   - Tests API
   - Build Docker (API + Dashboard)
   - Analyse de sécurité (bandit, safety)

2. **CD (`deploy.yml`)** - Déploiement Continu
   - **S'exécute uniquement si la CI réussit**
   - Build et push des images vers GitHub Container Registry
   - Déploiement automatique sur Render (API et Dashboard)
   - Tests de fumée post-déploiement

### Flux de déploiement

```
Push sur main → CI (tests) → ✅ Succès → CD (deploy) → Render
                           → ❌ Échec → Pas de déploiement
```

### Configuration Render (gratuit)

#### 1. Déployer l'API

| Paramètre | Valeur |
|-----------|--------|
| Type | Web Service |
| Environment | Docker |
| Dockerfile Path | `api/Dockerfile` |
| Health Check Path | `/health` |
| Port | 8000 |

#### 2. Déployer le Dashboard

| Paramètre | Valeur |
|-----------|--------|
| Type | Web Service |
| Environment | Docker |
| Dockerfile Path | `streamlit_app/Dockerfile` |
| Health Check Path | `/_stcore/health` |
| Port | 8501 |

**Variable d'environnement requise pour le Dashboard:**
```
API_URL=https://votre-api.onrender.com
```

### Secrets GitHub requis

Pour activer le déploiement automatique, configurez ces secrets dans GitHub :

| Secret | Description |
|--------|-------------|
| `RENDER_API_KEY` | Clé API Render (Account Settings → API Keys) |
| `RENDER_API_SERVICE_ID` | ID du service API (visible dans l'URL Render) |
| `RENDER_DASHBOARD_SERVICE_ID` | ID du service Dashboard |

### Variables d'environnement

| Variable | Service | Description | Défaut |
|----------|---------|-------------|--------|
| `PORT` | API/Dashboard | Port d'écoute | 8000 / 8501 |
| `API_URL` | Dashboard | URL de l'API | `http://localhost:8000` |
| `MODEL_PATH` | API | Chemin du modèle | `./models/lgbm_model.joblib` |
| `THRESHOLD` | API | Seuil de décision | `0.44` |

## 📖 Documentation API

La documentation interactive est disponible via :
- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "AMT_INCOME_TOTAL": 150000,
      "AMT_CREDIT": 500000,
      "AMT_ANNUITY": 25000,
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_2": 0.6,
      "EXT_SOURCE_3": 0.55
    }
  }'
```

### Exemple de réponse

```json
{
  "probability": 0.23,
  "prediction": 0,
  "decision": "approved",
  "threshold": 0.35,
  "confidence": "high"
}
```

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
