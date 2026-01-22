import os
import time
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Vérifie que l'endpoint /health répond et retourne un statut valide."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


@pytest.mark.parametrize("source", [None])
def test_data_describe(source):
    """Teste la capacité à charger et décrire les données via /data/describe.
    Ce test échouera en CI si les données ne sont pas accessibles — volontaire.
    """
    params = {}
    if source:
        params['source'] = source
    resp = client.get("/data/describe", params=params, timeout=60)
    # On attend 200 si les données sont accessibles
    assert resp.status_code == 200
    body = resp.json()
    assert "n_rows" in body
    assert "n_columns" in body
    assert "describe" in body
*** End Patch