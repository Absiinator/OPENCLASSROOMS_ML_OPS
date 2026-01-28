# Guide de Configuration Render pour Déploiement Automatique

Ce guide vous explique comment configurer Render pour le déploiement automatique de votre API et Dashboard.

## 📋 Prérequis

1. **Compte Render** : Créez un compte gratuit sur [render.com](https://render.com)
2. **Compte GitHub** : Votre repo doit être sur GitHub (déjà fait ✅)
3. **Images Docker** : Les images seront dans GitHub Container Registry (GHCR)

## � Architecture Docker

Le projet utilise 3 Dockerfiles distincts pour les 3 services :

### API (`api/Dockerfile`)
- **Base** : `python:3.10-slim`
- **Port** : 8000
- **Contenu** : Code source (`src/`, `api/`), modèles pré-entraînés (`models/`)
- **Variables d'env par défaut** :
  - `PORT=8000`
  - `PYTHONPATH=/app`
- **Health check** : `/health`
- **Commande** : `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Dashboard (`streamlit_app/Dockerfile`)
- **Base** : `python:3.10-slim`
- **Port** : 8501
- **Contenu** : App Streamlit (`app.py`), sources (`src/`), modèles (fallback local)
- **Variables d'env par défaut** :
  - `PORT=8501`
  - `API_URL=http://localhost:8000`
  - `MLFLOW_URL=http://localhost:5002`
- **Health check** : `/_stcore/health`
- **Commande** : `streamlit run app.py --server.port=$PORT`

### MLflow (`mlflow/Dockerfile`)
- **Base** : `python:3.10-slim`
- **Port** : 5000
- **Contenu** : Expériences MLflow (`mlruns/`)
- **Variables d'env par défaut** :
  - `PORT=5000`
- **Commande** : `mlflow server --host 0.0.0.0 --port $PORT`

⚠️ **Note importante** : Les données volumineuses (`data/`) ne sont **PAS** incluses dans les images Docker. Le Dashboard gère gracieusement leur absence (feature de comparaison désactivée).
### Résumé des variables d'environnement par service

| Service | Variable | Valeur par défaut | À configurer sur Render |
|---------|----------|-------------------|-------------------------|
| **API** | `PORT` | 8000 | Automatique (Render) |
| **API** | `HOST` | 0.0.0.0 | ✅ Optionnel |
| **Dashboard** | `PORT` | 8501 | Automatique (Render) |
| **Dashboard** | `API_URL` | http://localhost:8000 | ✅ **Obligatoire** : `https://votre-api.onrender.com` |
| **Dashboard** | `MLFLOW_URL` | http://localhost:5002 | ✅ **Obligatoire** : `https://votre-mlflow.onrender.com` |
| **MLflow** | `PORT` | 5000 | Automatique (Render) |
---

## �🚀 Étape 1 : Configuration API sur Render

### 1.1 Créer un nouveau Web Service

1. Connectez-vous à [dashboard.render.com](https://dashboard.render.com)
2. Cliquez sur **"New +"** → **"Web Service"**
3. Sélectionnez **"Deploy an existing image from a registry"**

### 1.2 Configurer l'image Docker

**Image URL** :
```
ghcr.io/absiinator/openclassrooms-ml-ops-api:latest
```

**Paramètres du service** :
- **Name** : `home-credit-api` (ou votre choix)
- **Region** : Europe (Frankfurt) ou proche de vous
- **Instance Type** : **Free** (pour commencer)

### 1.3 Variables d'environnement (optionnel pour l'API)

Ajoutez ces variables si nécessaire :
```bash
PORT=8000
HOST=0.0.0.0
```

### 1.4 Récupérer l'API Key pour le déploiement automatique

1. Allez dans **Account Settings** (icône utilisateur en haut à droite)
2. Cliquez sur **"API Keys"** dans le menu gauche
3. Cliquez sur **"Create API Key"**
4. Donnez un nom : `GitHub Actions Deploy`
5. **COPIEZ LA CLÉ** (vous ne la reverrez plus !)

### 1.5 Récupérer le Service ID

1. Ouvrez votre service API créé
2. Dans l'URL, copiez l'ID (exemple : `srv-xxxxxxxxxxxxx`)
   ```
   https://dashboard.render.com/web/srv-xxxxxxxxxxxxx
                                      ^^^^^^^^^^^^^^^^
   ```

## 🎨 Étape 2 : Configuration Dashboard sur Render

### 2.1 Créer un nouveau Web Service

Répétez les étapes 1.1 à 1.3 avec ces paramètres :

**Image URL** :
```
ghcr.io/absiinator/openclassrooms-ml-ops-dashboard:latest
```

**Paramètres du service** :
- **Name** : `home-credit-dashboard`
- **Region** : Europe (Frankfurt)
- **Instance Type** : **Free**

### 2.2 Variables d'environnement Dashboard

**Obligatoire** - Ajoutez ces variables :
```bash
API_URL=https://home-credit-api.onrender.com
MLFLOW_URL=https://home-credit-mlflow.onrender.com
```

**Optionnel** - Configuration Streamlit (déjà définies dans le Dockerfile) :
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

⚠️ **Important** : 
- Remplacez `home-credit-api.onrender.com` par l'URL réelle de votre API
- Vous ajouterez `MLFLOW_URL` après avoir créé le service MLflow (étape 2b)

### 2.3 Récupérer le Service ID Dashboard

Même procédure que 1.5, copiez le Service ID du Dashboard.

## � Étape 2b : Configuration MLflow sur Render

### 2b.1 Créer un service MLflow

MLflow permet de tracker les expériences et stocker les modèles.

1. Cliquez sur **"New +"** → **"Web Service"**
2. Sélectionnez **"Deploy an existing image from a registry"**

**Image URL** :

```
ghcr.io/absiinator/openclassrooms-ml-ops-mlflow:latest
```

ℹ️ Le Dockerfile MLflow (`mlflow/Dockerfile`) est déjà configuré et l'image sera automatiquement construite par GitHub Actions.

### 2b.2 Paramètres du service MLflow

| Paramètre | Valeur |
|-----------|--------|
| **Name** | `home-credit-mlflow` |
| **Region** | Europe (Frankfurt) |
| **Instance Type** | Free |
| **Port** | 5000 (ou `$PORT`) |

### 2b.3 Récupérer le Service ID MLflow

Même procédure que pour l'API et le Dashboard.

### 2b.4 Ajouter MLFLOW_URL au Dashboard

**Important** : Retournez au service Dashboard créé à l'étape 2 et ajoutez cette variable d'environnement :

```bash
MLFLOW_URL=https://home-credit-mlflow.onrender.com
```

⚠️ Remplacez par l'URL réelle de votre service MLflow sur Render.

## 🔐 Étape 3 : Configuration GitHub Secrets

### 3.1 Ajouter les secrets dans GitHub

1. Allez sur votre repo GitHub
2. **Settings** → **Secrets and variables** → **Actions**
3. Cliquez sur **"New repository secret"**

**Secrets à ajouter** :

| Nom | Valeur | Description |
|-----|--------|-------------|
| `RENDER_API_KEY` | Votre clé API Render | Clé copiée à l'étape 1.4 |
| `RENDER_API_SERVICE_ID` | `srv-xxxxxxxxxxxxx` | Service ID de l'API (étape 1.5) |
| `RENDER_DASHBOARD_SERVICE_ID` | `srv-xxxxxxxxxxxxx` | Service ID du Dashboard (étape 2.3) |
| `RENDER_MLFLOW_SERVICE_ID` | `srv-xxxxxxxxxxxxx` | Service ID de MLflow (étape 2b.4) |

### 3.2 Vérifier les secrets

Dans **Settings → Secrets → Actions**, vous devriez voir :
```
RENDER_API_KEY
RENDER_API_SERVICE_ID
RENDER_DASHBOARD_SERVICE_ID
RENDER_MLFLOW_SERVICE_ID
```

## ✅ Étape 4 : Tester le Déploiement

### 4.1 Premier déploiement manuel sur Render

1. Retournez dans chaque service sur Render
2. Cliquez sur **"Manual Deploy"** → **"Deploy latest commit"**
3. Attendez que le build se termine (⏱️ ~5-10 minutes)

### 4.2 Vérifier que les services fonctionnent

**API** :
```bash
curl https://votre-api.onrender.com/health
```

Devrait retourner :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Dashboard** :
Ouvrez `https://votre-dashboard.onrender.com` dans votre navigateur.

### 4.3 Tester le déploiement automatique

1. Faites un commit et push sur `main` :
   ```bash
   git add .
   git commit -m "test: trigger CD pipeline"
   git push origin main
   ```

2. Vérifiez dans **Actions** sur GitHub :
   - CI devrait passer ✅
   - CD devrait se déclencher automatiquement ✅
   - Les images Docker devraient être publiées ✅
   - Render devrait redéployer automatiquement ✅

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

## 🔗 Variables d'environnement GitHub

Pour que les URLs soient disponibles comme variables d'environnement dans les workflows CI/CD, ajoutez-les comme **variables** (pas secrets) dans GitHub :

1. Allez dans **Settings** → **Secrets and variables** → **Actions**
2. Cliquez sur l'onglet **"Variables"**
3. Ajoutez les variables suivantes :

| Nom | Valeur |
|-----|--------|
| `RENDER_API_URL` | `https://home-credit-api.onrender.com` |
| `RENDER_DASHBOARD_URL` | `https://home-credit-dashboard.onrender.com` |
| `RENDER_MLFLOW_URL` | `https://home-credit-mlflow.onrender.com` |

Ces variables peuvent ensuite être utilisées dans les workflows avec `${{ vars.RENDER_API_URL }}`.

## 📝 Notes Importantes

### ⚠️ Limitations du Plan Gratuit

- **Sleep après 15 min d'inactivité** : Premier appel prend ~30-60s
- **750h/mois** par service gratuit
- **Pas de custom domain** sur le plan gratuit

### 🔄 Workflow de Déploiement

```mermaid
graph LR
    A[Push sur main] --> B[CI Tests]
    B --> C{Tests OK?}
    C -->|Oui| D[Build Docker Images]
    D --> E[Push vers GHCR]
    E --> F[Trigger Render Deploy]
    F --> G[API déployée]
    F --> H[Dashboard déployé]
    F --> I[MLflow déployé]
    C -->|Non| J[Arrêt - Pas de déploiement]
```

### 🐛 Dépannage

**Problème : Le déploiement échoue**
- Vérifiez les logs dans Render Dashboard
- Vérifiez que les secrets GitHub sont corrects
- Vérifiez que les images sont publiques dans GHCR

**Problème : Dashboard ne peut pas joindre l'API**
- Vérifiez la variable `API_URL` dans le Dashboard
- Vérifiez que l'API est bien déployée et répond

**Problème : "Model not loaded"**
- Normal si les modèles ne sont pas inclus dans l'image Docker
- Le Dashboard utilise automatiquement le fallback local si l'API ne répond pas

---

## ✅ Checklist Finale

- [ ] Compte Render créé
- [ ] Web Service API créé
- [ ] Web Service Dashboard créé  
- [ ] Web Service MLflow créé
- [ ] API Key Render générée
- [ ] Service IDs copiés (API, Dashboard, MLflow)
- [ ] Secrets GitHub configurés (`RENDER_API_KEY`, `RENDER_API_SERVICE_ID`, `RENDER_DASHBOARD_SERVICE_ID`, `RENDER_MLFLOW_SERVICE_ID`)
- [ ] Variables d'env configurées sur chaque service Render
- [ ] Variables GitHub configurées (URLs de déploiement - optionnel)
- [ ] Premier déploiement manuel réussi
- [ ] API répond sur `/health`
- [ ] Dashboard accessible
- [ ] MLflow UI accessible
- [ ] Déploiement automatique testé
- [ ] URLs finales documentées

**Félicitations ! Votre pipeline CI/CD avec MLflow est opérationnel ! 🎉**
