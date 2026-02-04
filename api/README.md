# API Home Credit Scoring

## Objectif

Cette API expose le modèle de scoring crédit afin de calculer :
- la probabilité de défaut d’un client
- la décision binaire (ACCEPTÉ / REFUSÉ) selon un seuil métier optimisé

## Structure du dossier

- `main.py` : endpoints FastAPI (prédiction, batch, explication, infos modèle)
- `models.py` : schémas Pydantic (requêtes / réponses)
- `requirements.txt` : liste des packages utilisés par l’API
- `Dockerfile` : image Docker compatible Render (plan gratuit)

## Artefacts requis

L’API charge automatiquement les artefacts présents dans `models/` à la racine du projet :
- `models/lgbm_model.joblib`
- `models/preprocessor.joblib`
- `models/model_config.json`

## Endpoints principaux

- `/health` : état de l’API et chargement du modèle
- `/predict` : prédiction d’un client (format flexible)
- `/predict/batch` : prédictions multiples
- `/predict/explain` : explication locale des features
- `/model/info` : informations modèle
- `/model/features` : importance des variables

## Données et rapports

- Les données sont téléchargées dans l’image Docker au build (pas de Git LFS)
- Le rapport Evidently est servi par `/data/drift` si `reports/` est présent

## Liens utiles

- [README principal](../README.md)
- [Guide Render](../RENDER_SETUP.md)
- [README Dashboard](../streamlit_app/README.md)
- [README MLflow](../mlflow/README.md)
