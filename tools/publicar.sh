#!/usr/bin/env bash
# Publica el contenido de la carpeta colab/ en https://github.com/pscarg/electro2027
#
# El repositorio git vive FUERA de Dropbox (por defecto en ~/repos/electro2027),
# para evitar conflictos entre la sincronización de Dropbox y la carpeta .git.
# Este script copia colab/ al repositorio, hace un commit y lo sube.
#
# Uso:   bash tools/publicar.sh "mensaje del commit"
set -euo pipefail
ORIGEN="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${ELECTRO2027_REPO:-$HOME/repos/electro2027}"
MENSAJE="${1:-Actualización de notebooks}"

rsync -a --delete --exclude '.git' --exclude '__pycache__' --exclude '.ipynb_checkpoints' "$ORIGEN"/ "$REPO"/
cd "$REPO"
git add -A
if git diff --cached --quiet; then
    echo "No hay cambios para publicar."
    exit 0
fi
git commit -q -m "$MENSAJE"
git push -q origin main
echo "Publicado en https://github.com/pscarg/electro2027"
