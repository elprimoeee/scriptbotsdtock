"""Small, dependency-free client for the Trading 212 practice API.

The client is deliberately locked to the demo host.  Trading 212's market-order
endpoint is not idempotent, so POST requests are never retried here.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEMO_BASE_URL = "https://demo.trading212.com"


class Trading212Error(RuntimeError):
    """A sanitized Trading 212 API failure."""


def load_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without echoing credentials."""

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class Trading212Credentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_env_file(cls, path: Path) -> "Trading212Credentials":
        values = load_env_file(path)
        api_key = values.get("212_API_KEY_ID") or values.get("TRADING212_API_KEY_ID")
        api_secret = values.get("212_API_KEY_SECRET") or values.get("TRADING212_API_KEY_SECRET")
        if not api_key or not api_secret:
            raise Trading212Error(
                "Missing 212_API_KEY_ID/212_API_KEY_SECRET (or TRADING212_* equivalents)"
            )
        return cls(api_key=api_key, api_secret=api_secret)


class Trading212PracticeClient:
    """Read account state and place orders against Trading 212 demo only."""

    def __init__(
        self,
        credentials: Trading212Credentials,
        *,
        base_url: str = DEMO_BASE_URL,
        timeout_seconds: float = 20.0,
    ) -> None:
        normalized = base_url.rstrip("/")
        if normalized != DEMO_BASE_URL:
            raise ValueError("This client is locked to the Trading 212 demo host")
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds
        token = base64.b64encode(
            f"{credentials.api_key}:{credentials.api_secret}".encode("utf-8")
        ).decode("ascii")
        self._authorization = f"Basic {token}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        retry_gets: int = 2,
    ) -> Any:
        if not path.startswith("/api/"):
            raise ValueError("Trading 212 API paths must start with /api/")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": self._authorization, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        attempts = 1 + (retry_gets if method == "GET" else 0)
        for attempt in range(attempts):
            request = Request(
                f"{self.base_url}{path}", data=body, headers=headers, method=method
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    content = response.read()
                    return json.loads(content) if content else None
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if method == "GET" and exc.code in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise Trading212Error(f"Trading 212 returned HTTP {exc.code}: {detail}") from None
            except (URLError, TimeoutError) as exc:
                if method == "GET" and attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise Trading212Error(f"Trading 212 request failed: {exc.reason if isinstance(exc, URLError) else exc}") from None
        raise Trading212Error("Trading 212 request failed")

    def account_summary(self) -> dict[str, Any]:
        return self._request("GET", "/api/v0/equity/account/summary")

    def instruments(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v0/equity/metadata/instruments")

    def positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v0/equity/positions")

    def pending_orders(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v0/equity/orders")

    def market_order(
        self, *, ticker: str, quantity: float, extended_hours: bool = False
    ) -> dict[str, Any]:
        if not ticker or quantity == 0:
            raise ValueError("A ticker and non-zero quantity are required")
        return self._request(
            "POST",
            "/api/v0/equity/orders/market",
            payload={
                "ticker": ticker,
                "quantity": quantity,
                "extendedHours": extended_hours,
            },
            retry_gets=0,
        )
