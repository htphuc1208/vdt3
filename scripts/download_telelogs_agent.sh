#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_VENV="${TELELOGS_TOOL_VENV:-/tmp/vdt3-tools}"
DATA_DIR="${TELELOGS_AGENT_DATA_DIR:-$ROOT_DIR/data/telelogs_agent}"
REPO_ID="${TELELOGS_AGENT_REPO_ID:-netop/TeleLogsAgent}"

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
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

out_dir = Path(sys.argv[1]).expanduser().resolve()
repo_id = sys.argv[2]
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

out_dir.mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=[
            "TS1/test.json",
            "TS2/test.json",
            "TS3/test.json",
            "README.md",
            "fastapi_server.py",
            "fastmcp_server.py",
            "benchmark.py",
        ],
        token=token,
    )
except GatedRepoError as exc:
    raise SystemExit(
        "TeleLogsAgent is a gated Hugging Face dataset. Log in/accept the dataset "
        "conditions at https://huggingface.co/datasets/netop/TeleLogsAgent, then "
        "rerun with HF_TOKEN or HUGGING_FACE_HUB_TOKEN set.\n"
        f"Original error: {exc}"
    )
except HfHubHTTPError as exc:
    raise SystemExit(f"Failed to download {repo_id}: {exc}")

missing = [
    str(out_dir / name / "test.json")
    for name in ("TS1", "TS2", "TS3")
    if not (out_dir / name / "test.json").exists()
]
if missing:
    raise SystemExit("Download completed but required files are missing: " + ", ".join(missing))

print(f"TeleLogsAgent files are ready under {out_dir}")
PY
