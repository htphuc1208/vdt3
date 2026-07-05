"""JSON-lines stateful Python kernel used only inside the RCA-Agent container."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import signal
import sys
import traceback
from typing import Any

ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "statistics",
    "datetime",
    "pytz",
    "json",
    "re",
}
DENIED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "help",
    "input",
    "open",
    "quit",
}
DENIED_ATTRIBUTES = {
    "system",
    "popen",
    "spawn",
    "fork",
    "remove",
    "unlink",
    "rename",
    "replace",
    "rmdir",
    "mkdir",
    "makedirs",
    "write_text",
    "write_bytes",
    "to_csv",
    "to_json",
    "to_parquet",
    "to_pickle",
}
MAX_OUTPUT = 64_000


def validate_code(code: str) -> ast.Module:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS:
                    raise PermissionError(f"import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] not in ALLOWED_IMPORTS:
                raise PermissionError(f"import is not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            raise PermissionError(f"name is not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in DENIED_ATTRIBUTES:
            raise PermissionError(f"attribute is not allowed: {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise PermissionError("global/nonlocal statements are not allowed")
    return tree


def execute(code: str, namespace: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    tree = validate_code(code)
    body = list(tree.body)
    final_expression = body.pop() if body and isinstance(body[-1], ast.Expr) else None
    stdout = io.StringIO()

    def timed_out(_signum, _frame):
        raise TimeoutError("sandbox cell timed out")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(max(1, min(int(timeout_s), 300)))
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]), "<rca-agent>", "exec"), namespace)
            result = None
            if final_expression is not None:
                result = eval(compile(ast.Expression(final_expression.value), "<rca-agent>", "eval"), namespace)
        text = stdout.getvalue()
        if result is not None:
            text += ("\n" if text else "") + repr(result)
        return {"ok": True, "output": text[:MAX_OUTPUT], "truncated": len(text) > MAX_OUTPUT}
    except Exception:
        text = stdout.getvalue() + traceback.format_exc(limit=8)
        return {"ok": False, "output": text[:MAX_OUTPUT], "truncated": len(text) > MAX_OUTPUT}
    finally:
        signal.alarm(0)


def main() -> int:
    namespace: dict[str, Any] = {"__name__": "__rca_agent__"}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "reset":
                namespace = {"__name__": "__rca_agent__"}
                response = {"ok": True, "output": "reset"}
            elif operation == "execute":
                response = execute(
                    str(request.get("code") or ""),
                    namespace,
                    int(request.get("timeout_s") or 120),
                )
            else:
                response = {"ok": False, "output": f"unknown operation: {operation}"}
        except Exception:
            response = {"ok": False, "output": traceback.format_exc(limit=4)[:MAX_OUTPUT]}
        print(json.dumps(response, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
