# Dashboard Streamlit - Home Credit Scoring

## Objectif

Interface interactive pour expliquer les décisions d’octroi de crédit :
- score et probabilité
- comparaison client vs population (section dans le scoring, possible sans prédiction)
- radar par défaut basé sur `reports/feature_importance.csv` (issues des notebooks)
- rapport de data drift (Evidently)
- accessibilité WCAG
- liens de documentation et statuts API/MLflow dans la sidebar

## Structure

- `app.py` : application Streamlit
- `api_client.py` : appels API (health, predict, explain)
- `constants.py` : features requises et textes d’explication
- `requirements.txt` : dépendances
- `Dockerfile` : image Docker compatible Render (plan gratuit)

## Variables d’environnement

- `API_URL` : URL de l’API FastAPI (obligatoire, route `/predict`)
- `MLFLOW_URL` : URL de l’UI MLflow (obligatoire)
- `PORT` : port d’écoute Streamlit

## Liens utiles

- [README principal](../README.md)
- [Guide Render](../RENDER_SETUP.md)
- [README API](../api/README.md)
- [README MLflow](../mlflow/README.md)
