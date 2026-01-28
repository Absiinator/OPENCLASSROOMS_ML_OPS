#!/usr/bin/env python3
"""
Script de lancement principal - Home Credit Scoring
====================================================

Ce script permet de lancer les différents composants du projet:
- Modèle de scoring (entraînement)
- API de prédiction
- Dashboard Streamlit
- Interface MLflow UI

Usage:
    python run.py [commande] [options]

Commandes disponibles:
    train       Entraîner le modèle de scoring
    api         Lancer l'API FastAPI
    dashboard   Lancer le dashboard Streamlit
    mlflow      Lancer l'interface MLflow UI
    test        Exécuter les tests unitaires
    all         Lancer API + Dashboard + MLflow (dev mode complet)

Exemples:
    python run.py train --sample 0.3
    python run.py api --port 8000
    python run.py dashboard --port 8501
    python run.py mlflow --port 5002
    python run.py all                          # Lance tout!
"""

import argparse
import subprocess
import sys
import os
import socket
from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.resolve()

# Ports par défaut
DEFAULT_API_PORT = 8000
DEFAULT_DASHBOARD_PORT = 8501
DEFAULT_MLFLOW_PORT = 5002  # 5000/5001 utilisés par AirPlay sur macOS


def is_port_available(port: int) -> bool:
    """Vérifie si un port est disponible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Trouve un port disponible à partir du port de départ."""
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port
    raise RuntimeError(f"Aucun port disponible trouvé entre {start_port} et {start_port + max_attempts - 1}")
API_DIR = PROJECT_ROOT / "api"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
SRC_DIR = PROJECT_ROOT / "src"


def check_environment():
    """Vérifie que l'environnement est correctement configuré."""
    required_dirs = [API_DIR, STREAMLIT_DIR, SRC_DIR]
    for d in required_dirs:
        if not d.exists():
            print(f"❌ Dossier manquant: {d}")
            return False
    
    # Vérifier les fichiers essentiels
    required_files = [
        PROJECT_ROOT / "models" / "lgbm_model.joblib",
        PROJECT_ROOT / "models" / "preprocessor.joblib",
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("⚠️  Fichiers de modèle manquants. Exécutez d'abord 'python run.py train'")
        for f in missing:
            print(f"   - {f}")
        return False
    
    return True


def run_train(args):
    """Entraîne le modèle de scoring."""
    print("=" * 60)
    print("        ENTRAÎNEMENT DU MODÈLE DE SCORING")
    print("=" * 60)
    
    cmd = [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from src.train import main
main(sample_frac={sample}, include_supplementary={supplementary})
""".format(
        sample=args.sample if args.sample else "None",
        supplementary="True" if not args.no_supplementary else "False"
    )]
    
    os.chdir(PROJECT_ROOT)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_api(args):
    """Lance l'API FastAPI."""
    print("=" * 60)
    print("        LANCEMENT DE L'API FASTAPI")
    print("=" * 60)
    
    if not check_environment():
        print("\n❌ Environnement non prêt. Entraînez d'abord le modèle.")
        return 1
    
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    
    print(f"\n🚀 API disponible sur: http://{host}:{port}")
    print(f"📖 Documentation: http://{host}:{port}/docs")
    print("\nCtrl+C pour arrêter\n")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    os.chdir(PROJECT_ROOT)
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_dashboard(args):
    """Lance le dashboard Streamlit."""
    print("=" * 60)
    print("        LANCEMENT DU DASHBOARD STREAMLIT")
    print("=" * 60)
    
    port = args.port or 8501
    api_url = args.api_url or f"http://localhost:{args.api_port or 8000}"
    
    print(f"\n🚀 Dashboard disponible sur: http://localhost:{port}")
    print(f"🔗 API URL configurée: {api_url}")
    print("\nCtrl+C pour arrêter\n")
    
    # Configurer l'URL de l'API
    os.environ["API_URL"] = api_url
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(STREAMLIT_DIR / "app.py"),
        "--server.port", str(port),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_mlflow(args):
    """Lance l'interface MLflow UI."""
    print("=" * 60)
    print("        LANCEMENT DE L'INTERFACE MLFLOW")
    print("=" * 60)
    
    # Priorité: notebooks/mlruns (contient les expériences) puis mlruns racine
    tracking_uri = PROJECT_ROOT / "notebooks" / "mlruns"
    if not tracking_uri.exists():
        tracking_uri = PROJECT_ROOT / "mlruns"
        if not tracking_uri.exists():
            print("\n⚠️  Aucun dossier mlruns trouvé.")
            print("   Exécutez d'abord le notebook 03_Model_Training_MLflow.ipynb")
            print(f"   ou créez le dossier: {tracking_uri}")
            return 1
    
    # Résolution dynamique du port
    requested_port = args.port if args.port else DEFAULT_MLFLOW_PORT
    if not is_port_available(requested_port):
        print(f"\n⚠️  Port {requested_port} occupé, recherche d'un port disponible...")
        try:
            port = find_available_port(requested_port)
            print(f"✅ Port {port} trouvé et disponible")
        except RuntimeError as e:
            print(f"\n❌ {e}")
            return 1
    else:
        port = requested_port
    
    print(f"\n🚀 MLflow UI disponible sur: http://localhost:{port}")
    print(f"📁 Tracking URI: {tracking_uri}")
    print("\nCtrl+C pour arrêter\n")
    
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", f"file://{tracking_uri}",
        "--port", str(port),
        "--host", "127.0.0.1"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_tests(args):
    """Exécute les tests unitaires."""
    print("=" * 60)
    print("        EXÉCUTION DES TESTS UNITAIRES")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov=api", "--cov-report=html"])
    
    os.chdir(PROJECT_ROOT)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_all(args):
    """Lance API + Dashboard + MLflow en mode développement."""
    import threading
    import time
    import socket
    import urllib.request
    import urllib.error
    
    print("=" * 60)
    print("    MODE DÉVELOPPEMENT (API + DASHBOARD + MLFLOW)")
    print("=" * 60)
    
    if not check_environment():
        print("\n❌ Environnement non prêt. Entraînez d'abord le modèle.")
        return 1
    
    api_port = args.api_port or 8000
    dashboard_port = args.dashboard_port or 8501
    mlflow_port = getattr(args, 'mlflow_port', None) or DEFAULT_MLFLOW_PORT
    wait_timeout = getattr(args, 'wait_health_timeout', 30)
    wait_interval = getattr(args, 'wait_health_interval', 1)
    ci_mode = getattr(args, 'ci', False)
    
    print(f"\n🚀 Lancement de l'API sur le port {api_port}...")
    print(f"🚀 Lancement du Dashboard sur le port {dashboard_port}...")
    print(f"🚀 Lancement de MLflow UI sur le port {mlflow_port}...")
    print("\nCtrl+C pour arrêter tous les services\n")
    
    # Configurer l'environnement
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    os.environ["API_URL"] = f"http://localhost:{api_port}"
    os.environ["MLFLOW_URL"] = f"http://localhost:{mlflow_port}"
    
    # Lancer l'API en arrière-plan
    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", str(api_port),
    ]
    
    api_process = subprocess.Popen(api_cmd, cwd=PROJECT_ROOT)

    # Attendre que l'endpoint /health réponde
    def wait_for_health(host: str, port: int, timeout: int = 30, interval: int = 1) -> bool:
        url = f"http://{host}:{port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                # utiliser urllib pour compatibilité sans dépendances
                with urllib.request.urlopen(url, timeout=interval) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(interval)
        return False

    ready = wait_for_health('localhost', api_port, timeout=wait_timeout, interval=wait_interval)
    if ready:
        print(f"✅ API répond sur http://localhost:{api_port}/health")
    else:
        print(f"⚠️  L'API n'a pas répondu dans les {wait_timeout}s. Continuer...")

    # En mode CI, on s'arrête après vérification de l'API pour laisser le job CI lancer les étapes suivantes
    if ci_mode:
        print("Mode CI activé: l'API est lancée et répond (ou le timeout est atteint). Fin du processus.")
        return 0
    
    # Déterminer le répertoire MLflow
    tracking_uri = PROJECT_ROOT / "notebooks" / "mlruns"
    if not tracking_uri.exists():
        tracking_uri = PROJECT_ROOT / "mlruns"
    
    # Lancer MLflow UI
    mlflow_cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", f"file://{tracking_uri}",
        "--port", str(mlflow_port),
        "--host", "127.0.0.1"
    ]
    
    mlflow_process = subprocess.Popen(mlflow_cmd, cwd=PROJECT_ROOT)
    print(f"✅ MLflow UI lancé sur http://localhost:{mlflow_port}")
    
    # Lancer le dashboard
    dashboard_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(STREAMLIT_DIR / "app.py"),
        "--server.port", str(dashboard_port),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        dashboard_process = subprocess.Popen(dashboard_cmd, cwd=PROJECT_ROOT)
        print(f"✅ Dashboard lancé sur http://localhost:{dashboard_port}")
        print(f"\n📊 Services disponibles:")
        print(f"   - API: http://localhost:{api_port}")
        print(f"   - Dashboard: http://localhost:{dashboard_port}")
        print(f"   - MLflow: http://localhost:{mlflow_port}")
        print(f"   - API Docs: http://localhost:{api_port}/docs")
        api_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt des services...")
        api_process.terminate()
        mlflow_process.terminate()
        dashboard_process.terminate()
        api_process.wait()
        mlflow_process.wait()
        dashboard_process.wait()
    
    return 0


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Home Credit Scoring - Script de lancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples d'utilisation:
  python run.py train                    # Entraîner le modèle
  python run.py train --sample 0.3       # Entraîner sur 30% des données
  python run.py api                      # Lancer l'API sur le port {DEFAULT_API_PORT}
  python run.py dashboard                # Lancer le dashboard sur le port {DEFAULT_DASHBOARD_PORT}
  python run.py mlflow                   # Lancer MLflow UI sur le port {DEFAULT_MLFLOW_PORT}
  python run.py test                     # Exécuter les tests
  python run.py all                      # Lancer API + Dashboard + MLflow (complet!)
  
  python run.py all --api-port 9000 --dashboard-port 9501 --mlflow-port 9002
                                         # Lancer sur des ports personnalisés
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande train
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument("--sample", type=float, default=None,
                             help="Fraction des données (ex: 0.3 pour 30%%)")
    train_parser.add_argument("--no-supplementary", action="store_true",
                             help="Ne pas inclure les tables supplémentaires")
    
    # Commande api
    api_parser = subparsers.add_parser("api", help="Lancer l'API FastAPI")
    api_parser.add_argument("--port", type=int, default=DEFAULT_API_PORT, help=f"Port (défaut: {DEFAULT_API_PORT})")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host (défaut: 0.0.0.0)")
    api_parser.add_argument("--reload", action="store_true", help="Mode rechargement auto")
    
    # Commande dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Lancer le dashboard Streamlit")
    dash_parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT, help=f"Port (défaut: {DEFAULT_DASHBOARD_PORT})")
    dash_parser.add_argument("--api-url", help=f"URL de l'API (défaut: http://localhost:{DEFAULT_API_PORT})")
    dash_parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help=f"Port de l'API (défaut: {DEFAULT_API_PORT})")
    
    # Commande mlflow
    mlflow_parser = subparsers.add_parser("mlflow", help="Lancer MLflow UI")
    mlflow_parser.add_argument("--port", type=int, default=None, help=f"Port (défaut: {DEFAULT_MLFLOW_PORT})")
    
    # Commande test
    test_parser = subparsers.add_parser("test", help="Exécuter les tests")
    test_parser.add_argument("--coverage", action="store_true", help="Avec couverture de code")
    
    # Commande all
    all_parser = subparsers.add_parser("all", help="Lancer API + Dashboard + MLflow")
    all_parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help=f"Port API (défaut: {DEFAULT_API_PORT})")
    all_parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT, help=f"Port Dashboard (défaut: {DEFAULT_DASHBOARD_PORT})")
    all_parser.add_argument("--mlflow-port", type=int, default=DEFAULT_MLFLOW_PORT, help=f"Port MLflow (défaut: {DEFAULT_MLFLOW_PORT})")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Exécuter la commande
    commands = {
        "train": run_train,
        "api": run_api,
        "dashboard": run_dashboard,
        "mlflow": run_mlflow,
        "test": run_tests,
        "all": run_all
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
