#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_VENV="${TELECOMTS_TOOL_VENV:-/tmp/vdt3-tools}"
DATA_DIR="${TELECOMTS_DATA_DIR:-$ROOT_DIR/data/telecomts}"
REPO_ID="AliMaatouk/TelecomTS"

if [[ ! -x "$TOOL_VENV/bin/python" ]]; then
  python3 -m venv "$TOOL_VENV"
fi

if ! "$TOOL_VENV/bin/python" - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
then
  "$TOOL_VENV/bin/pip" install -q huggingface_hub
fi

"$TOOL_VENV/bin/python" - <<'PY' "$DATA_DIR" "$REPO_ID"
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

out_dir = Path(sys.argv[1]).expanduser().resolve()
repo_id = sys.argv[2]
out_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=out_dir,
    allow_patterns=["README.md", "anomalous/**/processed/chunked.jsonl"],
)

files = sorted(out_dir.glob("anomalous/**/processed/chunked.jsonl"))
synthetic = [path for path in files if "/synthetic/" in path.as_posix()]
jammer = [path for path in files if "/jammer/" in path.as_posix()]
if len(synthetic) != 9 or len(jammer) != 3:
    raise SystemExit(
        "Unexpected TelecomTS anomaly layout: "
        f"expected 9 synthetic and 3 jammer files, found {len(synthetic)} and {len(jammer)}"
    )

total_bytes = sum(path.stat().st_size for path in files)
print(
    f"TelecomTS anomaly subset is ready under {out_dir} "
    f"({len(files)} files, {total_bytes} bytes)"
)
PY
