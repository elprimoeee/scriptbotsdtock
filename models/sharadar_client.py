"""Small, secret-safe client for the Sharadar CSV API."""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from models.trading212_client import load_env_file


API_BASE_URL = "https://api.sharadar.com/v1.0"


class SharadarError(RuntimeError):
    """A sanitized Sharadar failure which never contains the API key."""


@dataclass(frozen=True)
class SharadarCredentials:
    api_key: str

    @classmethod
    def from_env_file(cls, path: Path) -> "SharadarCredentials":
        api_key = load_env_file(path).get("SHARADAR_API_KEY")
        if not api_key:
            raise SharadarError("Missing SHARADAR_API_KEY in the environment file")
        return cls(api_key=api_key)


class SharadarClient:
    """Retrieve paginated Sharadar tables without logging credentialed URLs."""

    def __init__(self, credentials: SharadarCredentials, *, base_url: str = API_BASE_URL,
                 timeout_seconds: float = 60.0, page_size: int = 100_000) -> None:
        if not credentials.api_key:
            raise ValueError("A non-empty Sharadar API key is required")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self._api_key = credentials.api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.page_size = page_size

    def _page(self, table: str, params: Mapping[str, object]) -> list[dict[str, str]]:
        query = {key: value for key, value in params.items() if value is not None}
        query.update({"api_key": self._api_key, "format": "csv"})
        request = Request(
            f"{self.base_url}/data/{table}?{urlencode(query)}",
            headers={"Accept": "text/csv", "User-Agent": "stock-research-bot/1.0"},
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    content = response.read().decode("utf-8-sig")
                return list(csv.DictReader(io.StringIO(content))) if content.strip() else []
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise SharadarError(
                    f"Sharadar table {table!r} returned HTTP {exc.code}: {detail}"
                ) from None
            except (URLError, TimeoutError) as exc:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                reason = exc.reason if isinstance(exc, URLError) else "request timed out"
                raise SharadarError(f"Sharadar table {table!r} request failed: {reason}") from None
        raise SharadarError(f"Sharadar table {table!r} request failed")

    def table(self, table: str, *, fields: Iterable[str] | None = None,
              tickers: Iterable[str] | None = None, start: str | None = None,
              end: str | None = None, **filters: object) -> list[dict[str, str]]:
        """Return all matching rows, following the API's offset pagination."""

        normalized_table = table.strip().lower()
        if not normalized_table or not normalized_table.replace("_", "").isalnum():
            raise ValueError("Invalid Sharadar table name")
        params: dict[str, object] = {
            "fields": ",".join(fields) if fields else None,
            "ticker": ",".join(dict.fromkeys(str(t).strip().upper() for t in tickers)) if tickers else None,
            "from": start,
            "to": end,
            "limit": self.page_size,
            "sort": filters.pop("sort", "date.asc"),
            **filters,
        }
        rows: list[dict[str, str]] = []
        offset = 0
        while True:
            page = self._page(normalized_table, {**params, "skip": offset})
            rows.extend(page)
            if len(page) < self.page_size:
                return rows
            offset += len(page)
