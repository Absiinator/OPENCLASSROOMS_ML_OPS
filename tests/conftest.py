"""
Configuration des tests pytest.
"""

import pytest
import sys
from pathlib import Path

# Ajouter src et api au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "api"))


@pytest.fixture(scope="session")
def project_root():
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_path(project_root):
    """Retourne le chemin des données."""
    return project_root / "data"


@pytest.fixture(scope="session")
def models_path(project_root):
    """Retourne le chemin des modèles."""
    return project_root / "models"
