# Tests Home Credit Scoring

Ce dossier contient les tests unitaires et d'intégration pour le projet de scoring crédit.

## Structure

```
tests/
├── conftest.py            # Configuration pytest et fixtures
├── test_cost.py           # Tests des fonctions de coût métier
├── test_preprocessing.py  # Tests du prétraitement
└── test_api.py           # Tests de l'API FastAPI
```

## Exécution des tests

### Tous les tests
```bash
pytest tests/ -v
```

### Tests spécifiques
```bash
# Tests de coût métier
pytest tests/test_cost.py -v

# Tests de prétraitement
pytest tests/test_preprocessing.py -v

# Tests API
pytest tests/test_api.py -v
```

### Avec couverture de code
```bash
pytest tests/ -v --cov=src --cov=api --cov-report=html
```

### Tests rapides uniquement
```bash
pytest tests/ -v -m "not slow"
```

## Conventions

- Un fichier de test par module source
- Tests organisés en classes par fonctionnalité
- Noms descriptifs: `test_<fonction>_<scenario>`
- Fixtures partagées dans `conftest.py`
