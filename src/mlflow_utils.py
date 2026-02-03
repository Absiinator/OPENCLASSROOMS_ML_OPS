"""
Utilitaires MLflow - Normalisation des chemins pour reproductibilité
=====================================================================

Ce module garantit que les meta.yaml contiennent des chemins relatifs,
fonctionnant identiquement en local et en Docker.

Fonctionnalités:
- Normalisation des chemins absolus → relatifs
- Validation de la structure MLflow
- Support pour runs futurs générés par les notebooks
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class MLflowPathNormalizer:
    """
    Normalise les chemins dans les fichiers meta.yaml de MLflow.
    
    Convertit les chemins absolus (ex: /Users/.../) en chemins relatifs
    qui fonctionnent indépendamment de l'environnement (local ou Docker).
    
    Exigences couvertes:
    - CE1: Pipeline reproductible (mêmes chemins partout)
    - CE2: Stockage centralisé (artifacts accessibles)
    - CE3: Formalisation des résultats (structure stable)
    """
    
    def __init__(self, mlflow_root: Path):
        """
        Initialise le normaliseur.
        
        Args:
            mlflow_root: Chemin vers le répertoire mlruns (ex: /app/mlruns ou ./notebooks/mlruns)
        """
        self.mlflow_root = Path(mlflow_root).resolve()
        
        if not self.mlflow_root.exists():
            raise ValueError(f"MLflow root n'existe pas: {self.mlflow_root}")
    
    def normalize_meta_yaml(self, meta_path: Path) -> bool:
        """
        Normalise un fichier meta.yaml spécifique.
        
        Convertit:
        - artifact_location: /Users/.../mlruns/446... → artifact_location: ./446...
        - artifact_uri: /Users/.../mlruns/446.../789.../artifacts → artifact_uri: ./446.../789.../artifacts
        
        Args:
            meta_path: Chemin vers le fichier meta.yaml
            
        Returns:
            True si le fichier a été modifié, False sinon
        """
        if not meta_path.exists():
            return False
        
        try:
            with open(meta_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Pattern 1: Supprimer les chemins absolus qui contiennent "mlruns"
            # /some/path/mlruns/ → ./
            # Patterns à gérer:
            # - /Users/jeffreylepage/Desktop/OPENCLASSROOMS/.../mlruns/...
            # - /app/mlruns/...
            # - /home/user/.../mlruns/...
            
            # Remplacer: artifact_location: /...mlruns/XXXXX
            # Par: artifact_location: ./XXXXX (ou juste le répertoire ID)
            content = re.sub(
                r'artifact_location:\s*/.*?mlruns/([^/\s]+)',
                r'artifact_location: ./\1',
                content
            )
            
            # Remplacer: artifact_uri: /...mlruns/XXX/YYY/artifacts
            # Par: artifact_uri: ./XXX/YYY/artifacts
            content = re.sub(
                r'artifact_uri:\s*/.*?mlruns/(.+)',
                r'artifact_uri: ./\1',
                content
            )
            
            # Aussi gérer les cas où c'est déjà un chemin relatif
            # Assurer la cohérence (./path vs path)
            content = re.sub(
                r'artifact_location:\s*/mlruns/([^/\s]+)',
                r'artifact_location: ./\1',
                content
            )
            
            if content != original_content:
                with open(meta_path, 'w') as f:
                    f.write(content)
                print(f"✅ Normalisé: {meta_path.relative_to(self.mlflow_root)}")
                return True
            
            return False
        
        except Exception as e:
            print(f"⚠️  Erreur lors de la normalisation de {meta_path}: {e}")
            return False
    
    def normalize_all(self) -> Dict[str, int]:
        """
        Normalise TOUS les fichiers meta.yaml du répertoire MLflow.
        
        Returns:
            Dict avec statistiques: {'modified': N, 'checked': N, 'errors': N}
        """
        stats = {'modified': 0, 'checked': 0, 'errors': 0}
        
        # Trouver tous les meta.yaml
        meta_files = list(self.mlflow_root.glob('**/meta.yaml'))
        
        if not meta_files:
            print(f"⚠️  Aucun meta.yaml trouvé dans {self.mlflow_root}")
            return stats
        
        print(f"📊 Normalisation de {len(meta_files)} fichiers meta.yaml...")
        
        for meta_path in meta_files:
            stats['checked'] += 1
            try:
                if self.normalize_meta_yaml(meta_path):
                    stats['modified'] += 1
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ {meta_path}: {e}")
        
        return stats
    
    def validate_structure(self) -> bool:
        """
        Valide la structure du répertoire MLflow.
        
        Vérifie:
        - Existence de répertoires d'expériences (IDs numériques)
        - Existence de meta.yaml
        - Cohérence des chemins
        
        Returns:
            True si la structure est valide
        """
        print(f"\n🔍 Validation de la structure MLflow...")
        
        valid = True
        
        # Vérifier la structure de base
        for item in self.mlflow_root.iterdir():
            if item.is_dir() and item.name not in ['.trash', 'models']:
                # C'est probablement une expérience
                meta_file = item / 'meta.yaml'
                if not meta_file.exists():
                    print(f"⚠️  Manque meta.yaml pour expérience: {item.name}")
                    valid = False
        
        if valid:
            print("✅ Structure valide")
        
        return valid


def normalize_mlflow_paths(mlflow_root: Optional[Path] = None) -> Dict[str, int]:
    """
    Fonction de commodité pour normaliser les chemins MLflow.
    
    Args:
        mlflow_root: Chemin vers mlruns (défaut: ./notebooks/mlruns ou ./mlruns)
        
    Returns:
        Statistiques de normalisation
    """
    if mlflow_root is None:
        # Déterminer le chemin automatiquement
        candidates = [
            Path.cwd() / 'notebooks' / 'mlruns',
            Path.cwd() / 'mlruns'
        ]
        for candidate in candidates:
            if candidate.exists():
                mlflow_root = candidate
                break
        
        if mlflow_root is None:
            raise ValueError("Impossible de trouver mlruns. Spécifiez mlflow_root explicitement.")
    
    normalizer = MLflowPathNormalizer(mlflow_root)
    normalizer.validate_structure()
    stats = normalizer.normalize_all()
    
    print(f"\n📈 Résumé de normalisation:")
    print(f"   Fichiers vérifiés: {stats['checked']}")
    print(f"   Fichiers modifiés: {stats['modified']}")
    print(f"   Erreurs: {stats['errors']}")
    
    return stats


if __name__ == "__main__":
    # Exemple d'utilisation
    import sys
    
    if len(sys.argv) > 1:
        mlflow_root = Path(sys.argv[1])
    else:
        mlflow_root = None
    
    try:
        stats = normalize_mlflow_paths(mlflow_root)
        sys.exit(0 if stats['errors'] == 0 else 1)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)
