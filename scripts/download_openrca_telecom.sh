#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_VENV="${OPENRCA_TOOL_VENV:-/tmp/vdt3-tools}"
OUT="${OPENRCA_TELECOM_ZIP:-/tmp/OpenRCA_Telecom.zip}"
DATA_DIR="${OPENRCA_DATA_DIR:-$ROOT_DIR/data/openrca}"
FILE_ID="1cyOKpqyAP4fy-QiJ6a_cKuwR7D46zyVe"
URL="https://drive.google.com/uc?id=${FILE_ID}"

if [[ ! -x "$TOOL_VENV/bin/gdown" ]]; then
  python3 -m venv "$TOOL_VENV"
  "$TOOL_VENV/bin/pip" install -q gdown
fi

"$TOOL_VENV/bin/gdown" --continue "$URL" -O "$OUT"

if [[ "${1:-}" == "--extract" ]]; then
  mkdir -p "$DATA_DIR"
  python3 - <<'PY' "$OUT" "$DATA_DIR"
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit(f"Corrupt zip entry: {bad}")
    archive.extractall(out_dir)
print(f"Extracted {zip_path} to {out_dir}")
PY
fi
