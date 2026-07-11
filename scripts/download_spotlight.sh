#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_VENV="${SPOTLIGHT_TOOL_VENV:-/tmp/vdt3-spotlight-tools}"
DATA_DIR="${SPOTLIGHT_DATA_DIR:-$ROOT_DIR/data/spotlight}"
FOLDER_URL="https://drive.google.com/drive/folders/1x7WnU6q9EodacUdh3iXRmi9FWrMD3rME?usp=sharing"

if [[ ! -x "$TOOL_VENV/bin/gdown" ]]; then
  python3 -m venv "$TOOL_VENV"
  "$TOOL_VENV/bin/pip" install -q gdown
fi

mkdir -p "$DATA_DIR"
"$TOOL_VENV/bin/gdown" --folder --continue "$FOLDER_URL" -O "$DATA_DIR/"

python3 -m telco_mas.spotlight.readiness --root "$DATA_DIR"
