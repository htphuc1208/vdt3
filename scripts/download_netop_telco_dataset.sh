#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_VENV="${NETOP_TOOL_VENV:-/tmp/vdt3-tools}"
DATASET="${1:-}"

case "$DATASET" in
  telelogs)
    REPO_ID="netop/TeleLogs"
    DATA_DIR="${TELELOGS_DATA_DIR:-$ROOT_DIR/data/telelogs}"
    ALLOW_PATTERNS=("README.md" "troubleshooting/**" "*.json" "**/*.json")
    ;;
  telelogs_agent)
    REPO_ID="netop/TeleLogsAgent"
    DATA_DIR="${TELELOGS_AGENT_DATA_DIR:-$ROOT_DIR/data/telelogs_agent}"
    ALLOW_PATTERNS=("README.md" "TS1/test.json" "TS2/test.json" "TS3/test.json" "fastapi_server.py" "fastmcp_server.py" "benchmark.py")
    ;;
  challenge|telco_challenge|telco_troubleshooting_challenge)
    REPO_ID="netop/Telco-Troubleshooting-Agentic-Challenge"
    DATA_DIR="${TELCO_CHALLENGE_DATA_DIR:-$ROOT_DIR/data/telco_troubleshooting_challenge}"
    ALLOW_PATTERNS=("README.md" "**/*.md" "**/*.json" "**/*.csv" "**/*.py")
    ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/download_netop_telco_dataset.sh telelogs|telelogs_agent|telco_troubleshooting_challenge

These NetOp/Huawei datasets are gated on Hugging Face. Accept the dataset terms
in a browser, then export HF_TOKEN or HUGGING_FACE_HUB_TOKEN before rerunning.
EOF
    exit 2
    ;;
esac

if [[ ! -x "$TOOL_VENV/bin/python" ]]; then
  python3 -m venv "$TOOL_VENV"
fi

if ! "$TOOL_VENV/bin/python" - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
then
  "$TOOL_VENV/bin/pip" install -q huggingface_hub
fi

"$TOOL_VENV/bin/python" - <<'PY' "$DATA_DIR" "$REPO_ID" "${ALLOW_PATTERNS[@]}"
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

out_dir = Path(sys.argv[1]).expanduser().resolve()
repo_id = sys.argv[2]
allow_patterns = list(sys.argv[3:])
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

out_dir.mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=allow_patterns,
        token=token,
    )
except GatedRepoError as exc:
    raise SystemExit(
        f"{repo_id} is gated on Hugging Face. Accept the dataset conditions, "
        "then rerun with HF_TOKEN or HUGGING_FACE_HUB_TOKEN set.\n"
        f"Original error: {exc}"
    )
except HfHubHTTPError as exc:
    raise SystemExit(f"Failed to download {repo_id}: {exc}")

files = [
    path for path in out_dir.rglob("*")
    if path.is_file() and path.name != "README.md" and ".cache" not in path.parts
]
if not files:
    raise SystemExit(f"Download completed but no data files were found under {out_dir}")

print(f"{repo_id} files are ready under {out_dir} ({len(files)} data files)")
PY
