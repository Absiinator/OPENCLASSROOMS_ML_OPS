#!/usr/bin/env python3
"""
Script de téléchargement et extraction des données Home Credit.
Ce script télécharge l'archive ZIP depuis le lien OC et l'extrait dans le dossier data/.
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path


DATA_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip"

def get_project_root() -> Path:
    """Retourne la racine du projet."""
    return Path(__file__).parent.parent


def download_data(url: str = DATA_URL, force: bool = False) -> Path:
    """
    Télécharge et extrait les données Home Credit.
    
    Args:
        url: URL de l'archive ZIP
        force: Si True, re-télécharge même si les données existent
        
    Returns:
        Chemin vers le dossier des données
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    zip_path = data_dir / "home-credit-data.zip"
    
    # Créer le dossier data s'il n'existe pas
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Vérifier si les données existent déjà
    required_files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "credit_card_balance.csv",
        "POS_CASH_balance.csv",
        "HomeCredit_columns_description.csv"
    ]
    
    if not force:
        existing_files = [f for f in required_files if (data_dir / f).exists()]
        if len(existing_files) == len(required_files):
            print("✅ Toutes les données sont déjà présentes dans data/")
            return data_dir
    
    # Télécharger l'archive
    print(f"📥 Téléchargement des données depuis {url[:50]}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"✅ Téléchargement terminé: {zip_path}")
    except Exception as e:
        print(f"❌ Erreur de téléchargement: {e}")
        raise
    
    # Extraire l'archive
    print("📦 Extraction de l'archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Lister les fichiers
            file_list = zip_ref.namelist()
            print(f"   Fichiers dans l'archive: {len(file_list)}")
            
            # Extraire dans un dossier temporaire
            temp_extract = data_dir / "temp_extract"
            zip_ref.extractall(temp_extract)
            
            # Trouver et déplacer les fichiers CSV
            for root, dirs, files in os.walk(temp_extract):
                for file in files:
                    if file.endswith('.csv'):
                        src = Path(root) / file
                        dst = data_dir / file
                        shutil.move(str(src), str(dst))
                        print(f"   ✓ {file}")
            
            # Nettoyer
            shutil.rmtree(temp_extract)
            
        # Supprimer le ZIP
        zip_path.unlink()
        print("✅ Extraction terminée et nettoyage effectué")
        
    except Exception as e:
        print(f"❌ Erreur d'extraction: {e}")
        raise
    
    # Vérification finale
    print("\n📊 Vérification des fichiers:")
    for f in required_files:
        file_path = data_dir / f
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {f} ({size_mb:.2f} MB)")
        else:
            print(f"   ✗ {f} MANQUANT")
    
    return data_dir


def copy_existing_data():
    """
    Copie les données du dossier source vers data/ si disponibles.
    Utile si les données sont déjà présentes dans le workspace.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    # Chemin vers les données existantes dans le workspace
    source_dir = project_root.parent / "Projet+Mise+en+prod+-+home-credit-default-risk"
    
    if source_dir.exists():
        print(f"📂 Données trouvées dans {source_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(source_dir.glob("*.csv"))
        for csv_file in csv_files:
            dst = data_dir / csv_file.name
            if not dst.exists():
                shutil.copy2(csv_file, dst)
                print(f"   ✓ Copié: {csv_file.name}")
            else:
                print(f"   - Existe déjà: {csv_file.name}")
        
        return True
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Télécharge les données Home Credit")
    parser.add_argument("--force", action="store_true", help="Force le re-téléchargement")
    parser.add_argument("--copy-local", action="store_true", help="Copie depuis les données locales si disponibles")
    
    args = parser.parse_args()
    
    if args.copy_local:
        if copy_existing_data():
            print("✅ Données copiées depuis le dossier local")
        else:
            print("⚠️ Pas de données locales trouvées, téléchargement...")
            download_data(force=args.force)
    else:
        download_data(force=args.force)
