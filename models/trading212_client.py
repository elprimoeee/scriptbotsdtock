"""Small, dependency-free client locked to the Trading 212 demo API.

GET requests may be retried. Order submissions are sent once because the API
does not make them idempotent.
"""

from __future__ import annotations

import base64
import json
import math
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
    """Use Trading 212's Invest/Stocks equity demo API; this client is not for CFDs."""

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
        path: str,
        *,
        retry_gets: int = 2,
    ) -> Any:
        if not path.startswith("/api/v0/equity/"):
            raise ValueError("Trading 212 API paths must be equity API paths")
        headers = {"Authorization": self._authorization, "Accept": "application/json"}

        attempts = 1 + retry_gets
        for attempt in range(attempts):
            request = Request(
                f"{self.base_url}{path}", headers=headers, method="GET"
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    content = response.read()
                    return json.loads(content) if content else None
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in {408, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise Trading212Error(f"Trading 212 returned HTTP {exc.code}: {detail}") from None
            except (URLError, TimeoutError) as exc:
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise Trading212Error(f"Trading 212 request failed: {exc.reason if isinstance(exc, URLError) else exc}") from None
        raise Trading212Error("Trading 212 request failed")

    def account_summary(self) -> dict[str, Any]:
        return self._request("/api/v0/equity/account/summary")

    def instruments(self) -> list[dict[str, Any]]:
        return self._request("/api/v0/equity/metadata/instruments")

    def positions(self) -> list[dict[str, Any]]:
        return self._request("/api/v0/equity/positions")

    def pending_orders(self) -> list[dict[str, Any]]:
        return self._request("/api/v0/equity/orders")

    def order_by_id(self, order_id: int) -> dict[str, Any]:
        if isinstance(order_id, bool) or int(order_id) < 1:
            raise ValueError("order_id must be a positive integer")
        return self._request(f"/api/v0/equity/orders/{int(order_id)}")

    def market_order(self, *, ticker: str, quantity: float) -> dict[str, Any]:
        """Submit one demo market order. POST requests are never retried."""

        try:
            normalized_quantity = float(quantity)
        except (TypeError, ValueError):
            raise ValueError("quantity must be a finite, non-zero number") from None
        if not ticker or not ticker.strip():
            raise ValueError("ticker is required")
        if not math.isfinite(normalized_quantity) or normalized_quantity == 0:
            raise ValueError("quantity must be a finite, non-zero number")

        payload = {
            "ticker": ticker.strip(),
            "quantity": normalized_quantity,
            "extendedHours": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/v0/equity/orders/market",
            data=body,
            headers={
                "Authorization": self._authorization,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code >= 500 or exc.code == 408:
                raise Trading212Error(
                    f"Trading 212 returned HTTP {exc.code}; the order outcome may be unknown. "
                    "Check demo order history before retrying manually. "
                    f"{detail}"
                ) from None
            raise Trading212Error(
                f"Trading 212 returned HTTP {exc.code}: {detail}"
            ) from None
        except json.JSONDecodeError:
            raise Trading212Error(
                "Trading 212 returned an unreadable order response; the order outcome may be unknown. "
                "Check demo order history before retrying manually."
            ) from None
        except (URLError, TimeoutError) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            raise Trading212Error(
                "Trading 212 demo order request failed; the outcome may be unknown. "
                "Check demo order history before retrying manually. "
                f"({reason})"
            ) from None
