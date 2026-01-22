#!/usr/bin/env bash
# Lance l'API et le dashboard ensemble en utilisant le script `run.py`
# Usage: 1) Assurez-vous d'avoir l'environnement conda 'home-credit-scoring'
#        2) Exécutez: ./launch_all.sh

set -euo pipefail

ENV_NAME="home-credit-scoring"

echo "Vérification de l'environnement conda '${ENV_NAME}'..."
if ! conda env list | grep -q "${ENV_NAME}"; then
  echo "Environnement '${ENV_NAME}' introuvable. Créez-le avec: conda env create -f environment.yml"
  exit 1
fi

echo "Lancement de l'API et du dashboard (mode développement)..."
conda run -n "${ENV_NAME}" --no-capture-output python run.py all

echo "Services arrêtés."
