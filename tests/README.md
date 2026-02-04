# Tests Home Credit Scoring

Tests simples et rapides pour valider les composants essentiels en CI/CD.

**Objectif** : Exécution rapide (<1 min) sans déploiement
**Scope** : Validation des features critiques uniquement
**Exclus** : Tests de déploiement sur Render

## Structure

```
tests/
├── conftest.py            # Configuration pytest et fixtures
├── test_cost.py           # Tests des fonctions de coût métier
├── test_preprocessing.py  # Tests du prétraitement
└── test_api.py           # Tests de l'API FastAPI
```

## Exécution des tests

Les tests sont exécutables via `pytest` (intégrés au workflow CI/CD).
Les fichiers couvrent le coût métier, le prétraitement et l’API.

## Conventions

- Un fichier de test par module source
- Tests organisés en classes par fonctionnalité
- Noms descriptifs: `test_<fonction>_<scenario>`
- Fixtures partagées dans `conftest.py`

## Liens utiles

- [README principal](../README.md)
- [README API](../api/README.md)
