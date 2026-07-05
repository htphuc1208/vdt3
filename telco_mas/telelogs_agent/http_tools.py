"""HTTP-tool adapter for the official TeleLogsAgent FastAPI server."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


Transport = Callable[[str, str, dict[str, Any], dict[str, str], float], Any]


FALLBACK_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("/scenario", "Return the current TeleLogsAgent scenario description and context."),
    ("/signaling-plane-event-log", "Return signaling-plane event logs for the scenario when available."),
    ("/throughput-logs", "Return user-plane throughput logs for the scenario."),
    ("/cell-info", "Return serving and neighboring cell configuration information."),
    ("/gnodeb-location", "Return gNodeB location information."),
    ("/user-location", "Return user location information."),
    ("/user-speed", "Return user speed or mobility information."),
    ("/serving-cell-pci", "Return the serving cell PCI time series or value."),
    ("/serving-cell-rsrp", "Return serving cell RSRP measurements."),
    ("/serving-cell-sinr", "Return serving cell SINR measurements."),
    ("/rbs-allocated-to-user", "Return allocated resource blocks for the user."),
    ("/neighboring-cells-pci", "Return neighboring cell PCI information."),
    ("/neighboring-cell-rsrp", "Return neighboring cell RSRP measurements."),
    ("/beam-scenario-info", "Return beam scenario information when available."),
)


class TeleLogsHTTPError(RuntimeError):
    """Raised when the TeleLogsAgent FastAPI tool server cannot be used."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ToolEndpoint:
    name: str
    path: str
    description: str
    parameters: dict[str, Any]


class TeleLogsHTTPClient:
    """Expose TeleLogsAgent FastAPI endpoints as OpenAI-style function tools."""

    def __init__(
        self,
        base_url: str = "http://localhost:7861",
        *,
        scenario_id: str,
        timeout: float = 20.0,
        max_result_chars: int = 16_000,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.scenario_id = str(scenario_id)
        self.timeout = timeout
        self.max_result_chars = max_result_chars
        self._transport = transport
        self._tool_paths: dict[str, str] = {}

    def tools_spec(self) -> list[dict[str, Any]]:
        endpoints = self._discover_endpoints()
        specs = []
        self._tool_paths = {}
        for endpoint in endpoints:
            name = _safe_tool_name(endpoint.name)
            self._tool_paths[name] = endpoint.path
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": endpoint.description,
                        "parameters": endpoint.parameters,
                    },
                }
            )
        return specs

    def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        if not self._tool_paths:
            self.tools_spec()
        path = self._tool_paths.get(name) or _path_from_name(name)
        args = arguments or {}
        try:
            payload = self._request_json("GET", path, args)
        except TeleLogsHTTPError as exc:
            if exc.status not in {400, 405, 422}:
                raise
            payload = self._request_json("POST", path, args)
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        if self.max_result_chars > 0 and len(text) > self.max_result_chars:
            omitted = len(text) - self.max_result_chars
            text = text[: self.max_result_chars] + f"... [truncated {omitted} chars]"
        return text

    def _discover_endpoints(self) -> list[ToolEndpoint]:
        try:
            payload = self._request_json("GET", "/tools", {})
        except TeleLogsHTTPError as exc:
            if exc.status not in {404, 405}:
                raise
            return _fallback_endpoints()
        endpoints = _parse_tools_payload(payload)
        return endpoints or _fallback_endpoints()

    def _request_json(self, method: str, path: str, params: dict[str, Any]) -> Any:
        headers = {"Accept": "application/json", "X-Scenario-Id": self.scenario_id}
        if self._transport is not None:
            return self._transport(method, _normalise_path(path), params, headers, self.timeout)

        url = self.base_url + _normalise_path(path)
        body = None
        if method.upper() == "GET":
            query = urllib.parse.urlencode(_query_params(params), doseq=True)
            if query:
                url += "?" + query
        else:
            body = json.dumps(params or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user-supplied benchmark URL
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise TeleLogsHTTPError(
                f"TeleLogsAgent server returned HTTP {exc.code} for {method} {path}: {detail}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise TeleLogsHTTPError(
                f"Cannot reach TeleLogsAgent server at {self.base_url}: {exc.reason}"
            ) from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}


def _parse_tools_payload(payload: Any) -> list[ToolEndpoint]:
    items = _tool_items(payload)
    endpoints: list[ToolEndpoint] = []
    seen: set[str] = set()
    for item in items:
        endpoint = _endpoint_from_tool_item(item)
        if endpoint is None or endpoint.path in seen:
            continue
        seen.add(endpoint.path)
        endpoints.append(endpoint)
    return endpoints


def _tool_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tools", "available_tools", "functions", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(value, dict) for value in payload.values()):
            return list(payload.values())
    return []


def _endpoint_from_tool_item(item: Any) -> ToolEndpoint | None:
    if isinstance(item, str):
        path = _path_from_name(item)
        return ToolEndpoint(
            name=item,
            path=path,
            description=f"Call the TeleLogsAgent FastAPI endpoint {path}.",
            parameters=_generic_parameters(),
        )
    if not isinstance(item, dict):
        return None
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    source = function or item
    raw_name = (
        source.get("name")
        or item.get("name")
        or source.get("endpoint")
        or item.get("endpoint")
        or source.get("path")
        or item.get("path")
    )
    if not raw_name:
        return None
    path = (
        item.get("path")
        or source.get("path")
        or item.get("endpoint")
        or source.get("endpoint")
        or _path_from_name(str(raw_name))
    )
    description = (
        source.get("description")
        or item.get("description")
        or f"Call the TeleLogsAgent FastAPI endpoint {_normalise_path(str(path))}."
    )
    parameters = source.get("parameters") or source.get("input_schema") or item.get("parameters")
    if not isinstance(parameters, dict):
        parameters = _generic_parameters()
    return ToolEndpoint(
        name=str(raw_name),
        path=_normalise_path(str(path)),
        description=str(description),
        parameters=parameters,
    )


def _fallback_endpoints() -> list[ToolEndpoint]:
    return [
        ToolEndpoint(
            name=path.strip("/"),
            path=path,
            description=description,
            parameters=_generic_parameters(),
        )
        for path, description in FALLBACK_ENDPOINTS
    ]


def _safe_tool_name(name: str) -> str:
    cleaned = name.strip().strip("/")
    cleaned = cleaned.replace("-", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "telelogs_tool"
    if cleaned[0].isdigit():
        cleaned = "tool_" + cleaned
    return cleaned


def _path_from_name(name: str) -> str:
    text = name.strip()
    if text.startswith("/"):
        return _normalise_path(text)
    return _normalise_path("/" + text.strip("/").replace("_", "-"))


def _normalise_path(path: str) -> str:
    path = path.strip()
    if not path.startswith("/"):
        path = "/" + path
    return path


def _query_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            out[key] = value
        else:
            out[key] = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return out


def _generic_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
