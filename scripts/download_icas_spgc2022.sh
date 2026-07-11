#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ICAS_SPGC_DATA_DIR:-$ROOT_DIR/data/icas_spgc2022}"
REPO_DIR="$DATA_DIR/SPGC_aiops_bjtu"
REPO_URL="https://github.com/zxuan000/SPGC_aiops_bjtu.git"

mkdir -p "$DATA_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

python3 -m telco_mas.icas_spgc.readiness --root "$REPO_DIR/all_file"
