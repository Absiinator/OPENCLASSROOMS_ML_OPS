# Dashboard Streamlit - Home Credit Scoring

## Objectif

Interface interactive pour expliquer les décisions d’octroi de crédit :
- score et probabilité
- comparaison client vs population (section dans le scoring, possible sans prédiction)
- page dédiée d’analyse du dataset (statistiques clés + infos modèle + exigences + checklist CE)
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

## Libellés & mapping des variables

- Les menus/tableaux affichent des libellés explicites (même pour les features agrégées).
- Le mapping est défini dans `app.py` :
  - `FEATURE_LABELS`, `FEATURE_LABEL_OVERRIDES`
  - `PREFIX_LABELS`, `TOKEN_LABELS`, `STAT_SUFFIXES`
- Les libellés sont **uniquement UI** : l’API reçoit toujours les **noms de colonnes**.
- Le payload d’inférence utilise **uniquement** la clé `features`.

## Format d’inférence (frontend → API)

Le dashboard envoie un **dictionnaire `features`** contenant les colonnes d’origine Home Credit.
Le modèle accepte :
- **Minimum recommandé** : 17 features clés (finances, scores externes, ratios) :
  - `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`
  - `DAYS_BIRTH`, `DAYS_EMPLOYED`
  - `CNT_CHILDREN`, `CODE_GENDER_M`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`
  - `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `REGION_RATING_CLIENT`
  - `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_MEAN`
- **Optionnelles** : catégories (`NAME_*`, `OCCUPATION_TYPE`, etc.), flags (`FLAG_*`), agrégées (`BUREAU_`, `PREV_`, `INST_`, `POS_`, `CC_`).

Le pipeline API :
- ajoute les ratios via `create_application_features()`
- complète les colonnes manquantes en `NaN` puis imputation

➡️ Les tables auxiliaires ne sont **pas** agrégées au runtime : si vous souhaitez les utiliser, fournissez-les déjà agrégées.
- Les variables importantes sont visibles directement.
- Les variables avancées (moins importantes) sont regroupées dans un **accordéon**.
- Les variables numériques bornées utilisent des **sliders** ; les variables importantes bornées ont aussi une **saisie brute**.
- Les catégories multi‑tags (si détectées) utilisent des cases à cocher avec bouton de réinitialisation.

## Liens utiles

- [README principal](../README.md)
- [Guide Render](../RENDER_SETUP.md)
- [README API](../api/README.md)
- [README MLflow](../mlflow/README.md)

## Notebooks (GitHub)

- https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/01_EDA.ipynb
- https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/02_Preprocessing_Features.ipynb
- https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/03_Model_Training_MLflow.ipynb
- https://github.com/Absiinator/OPENCLASSROOMS_ML_OPS/blob/main/notebooks/04_Drift_Evidently.ipynb
