#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TELECOM_BENCH_DATA_DIR:-$ROOT_DIR/data/telecom_bench}"
REPO_DIR="$DATA_DIR/TeleCom-Bench"
REPO_URL="https://github.com/ZTE-AICloud/TeleCom-Bench.git"

mkdir -p "$DATA_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

python3 -m telco_mas.tnrca.readiness \
  --root "$REPO_DIR/datasets/Knowledge_Application/Root_Cause_Diagnosis"
