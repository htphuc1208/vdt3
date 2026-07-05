"""Resource-bounded Docker sandbox for RCA-Agent generated Python."""
from __future__ import annotations

import json
import hashlib
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any

from .prepared import PreparedOpenRCA

DEFAULT_IMAGE_REPOSITORY = "telco-openrca-executor"


class SandboxError(RuntimeError):
    pass


class DockerPythonSandbox:
    def __init__(
        self,
        prepared: PreparedOpenRCA,
        row_id: int,
        *,
        image: str | None = None,
        cell_timeout_s: int = 120,
    ) -> None:
        self.prepared = prepared
        self.row_id = int(row_id)
        self.image = image or source_image_tag()
        self.cell_timeout_s = cell_timeout_s
        self.process: subprocess.Popen[str] | None = None

    @staticmethod
    def ensure_image(
        *,
        image: str | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        root = Path(project_root or Path(__file__).resolve().parents[2])
        image = image or source_image_tag(root)
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode == 0:
            return
        build = subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(root / "docker" / "openrca-executor.Dockerfile"),
                "-t",
                image,
                str(root),
            ],
            check=False,
        )
        if build.returncode != 0:
            raise SandboxError("Failed to build the OpenRCA executor image")

    def start(self) -> None:
        if self.process is not None:
            return
        self.ensure_image(image=self.image)
        row_dir = self.prepared.row_dir(self.row_id)
        stats_path = self.prepared.metric_stats_path(self.row_id)
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--memory",
            "3g",
            "--cpus",
            "2",
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            f"{row_dir}:/telemetry/row:ro",
            "-v",
            f"{stats_path}:/telemetry/metric_stats.parquet:ro",
            self.image,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={"PATH": os.environ.get("PATH", "")},
        )

    def execute(self, code: str) -> dict[str, Any]:
        self.start()
        assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
        request = json.dumps({"op": "execute", "code": code, "timeout_s": self.cell_timeout_s})
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select(
            [self.process.stdout],
            [],
            [],
            self.cell_timeout_s + 10,
        )
        if not ready:
            self.close(kill=True)
            raise SandboxError("RCA-Agent sandbox did not return before its deadline")
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read(2000) if self.process.stderr is not None else ""
            raise SandboxError(f"RCA-Agent sandbox stopped unexpectedly: {stderr}")
        return json.loads(line)

    def reset(self) -> None:
        if self.process is None:
            return
        self._request({"op": "reset"})

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def close(self, *, kill: bool = False) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            if kill:
                self.process.kill()
            else:
                self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        finally:
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(kill=exc is not None)
        return False


def source_image_tag(project_root: str | Path | None = None) -> str:
    root = Path(project_root or Path(__file__).resolve().parents[2])
    digest = hashlib.sha256()
    for path in (
        root / "docker" / "openrca-executor.Dockerfile",
        root / "telco_mas" / "openrca" / "sandbox_kernel.py",
    ):
        digest.update(path.read_bytes())
    return f"{DEFAULT_IMAGE_REPOSITORY}:{digest.hexdigest()[:16]}"
