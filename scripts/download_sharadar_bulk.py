#!/usr/bin/env python3
"""Download full-history Sharadar tables as compressed CSV archives."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sharadar_client import API_BASE_URL, SharadarCredentials, SharadarError


DEFAULT_TABLES = ("stocks", "daily", "fundamentals", "sp500")
CHUNK_SIZE = 1024
RANGE_SIZE = 32 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", default=list(DEFAULT_TABLES))
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/raw/sharadar_bulk"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def _safe_table_name(table: str) -> str:
    normalized = table.strip().lower()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError(f"Invalid Sharadar table name: {table!r}")
    return normalized


def _valid_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except zipfile.BadZipFile:
        return False


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _bulk_download_url(table: str, api_key: str) -> str:
    query = urlencode({"api_key": api_key, "years": "full"})
    request = Request(
        f"{API_BASE_URL}/data/{table}?{query}",
        headers={"Accept": "application/zip", "User-Agent": "stock-research-bot/1.0"},
    )
    try:
        response = build_opener(_NoRedirect).open(request, timeout=60.0)
    except HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise SharadarError(
                f"Sharadar bulk table {table!r} returned HTTP {exc.code}: {detail}"
            ) from None
        response = exc
    location = response.headers.get("Location")
    parsed = urlparse(location or "")
    if parsed.scheme != "https" or not parsed.netloc:
        raise SharadarError(f"Sharadar bulk table {table!r} returned no secure download URL")
    return location


def _archive_size(download_url: str, table: str) -> int:
    request = Request(download_url, headers={"Range": "bytes=0-0"})
    try:
        with urlopen(request, timeout=60.0) as response:
            content_range = response.headers.get("Content-Range", "")
            response.read(1)
    except (HTTPError, URLError, TimeoutError) as exc:
        code = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else "network error"
        raise SharadarError(f"Sharadar bulk table {table!r} size probe failed: {code}") from None
    try:
        return int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError):
        raise SharadarError(f"Sharadar bulk table {table!r} returned no archive size") from None


def _download_range(download_url: str, table: str, start: int, end: int) -> bytes:
    request = Request(download_url, headers={"Range": f"bytes={start}-{end}"})
    expected = end - start + 1
    try:
        with urlopen(request, timeout=180.0) as response:
            if response.status != 206:
                raise SharadarError(
                    f"Sharadar bulk table {table!r} ignored byte range {start}-{end}"
                )
            payload = bytearray()
            while len(payload) < expected:
                chunk = response.read(min(CHUNK_SIZE, expected - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
    except HTTPError as exc:
        raise SharadarError(
            f"Sharadar bulk table {table!r} range {start}-{end} returned HTTP {exc.code}"
        ) from None
    except (OSError, URLError, TimeoutError) as exc:
        if isinstance(exc, URLError):
            reason = exc.reason
        elif isinstance(exc, TimeoutError):
            reason = "request timed out"
        else:
            reason = str(exc)
        raise SharadarError(
            f"Sharadar bulk table {table!r} range {start}-{end} failed: {reason}"
        ) from None
    written = len(payload)
    if written != expected:
        raise SharadarError(
            f"Sharadar bulk table {table!r} range was truncated: expected {expected:,}, got {written:,}"
        )
    return bytes(payload)


def download_table(table: str, api_key: str, output_dir: Path, *, force: bool,
                   workers: int) -> dict[str, object]:
    table = _safe_table_name(table)
    destination = output_dir / f"{table}.csv.zip"
    partial = destination.with_suffix(destination.suffix + ".part")
    if not force and _valid_zip(destination):
        print(f"{table}: keeping existing {destination} ({destination.stat().st_size:,} bytes)", flush=True)
        return {"table": table, "path": str(destination), "bytes": destination.stat().st_size, "status": "existing"}

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{table}: requesting full-history archive", flush=True)
    download_url = _bulk_download_url(table, api_key)
    expected = _archive_size(download_url, table)
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > expected:
        partial.unlink()
        downloaded = 0
    elif downloaded < expected and downloaded % RANGE_SIZE:
        downloaded -= downloaded % RANGE_SIZE
        with partial.open("r+b") as output:
            output.truncate(downloaded)
    with partial.open("ab") as output, ThreadPoolExecutor(max_workers=workers) as pool:
        while downloaded < expected:
            ranges: list[tuple[int, int]] = []
            cursor = downloaded
            for _ in range(workers * 4):
                if cursor >= expected:
                    break
                end = min(cursor + RANGE_SIZE, expected) - 1
                ranges.append((cursor, end))
                cursor = end + 1
            for attempt in range(2):
                try:
                    payloads = list(pool.map(
                        lambda byte_range: _download_range(
                            download_url, table, byte_range[0], byte_range[1]
                        ),
                        ranges,
                    ))
                    break
                except SharadarError:
                    if attempt:
                        raise
                    download_url = _bulk_download_url(table, api_key)
            for payload in payloads:
                output.write(payload)
                downloaded += len(payload)
            output.flush()
            print(f"{table}: {downloaded / 2**20:,.0f}/{expected / 2**20:,.0f} MiB", flush=True)

    if not _valid_zip(partial):
        raise SharadarError(f"Sharadar bulk table {table!r} did not return a valid ZIP archive")
    os.replace(partial, destination)
    print(f"{table}: saved {downloaded / 2**20:,.1f} MiB to {destination}", flush=True)
    return {"table": table, "path": str(destination), "bytes": downloaded, "status": "downloaded"}


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    credentials = SharadarCredentials.from_env_file(args.env_file)
    tables = list(dict.fromkeys(_safe_table_name(table) for table in args.tables))
    manifest = [
        download_table(
            table, credentials.api_key, args.output_dir,
            force=args.force, workers=args.workers,
        )
        for table in tables
    ]
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"downloaded_at": datetime.now(timezone.utc).isoformat(), "history": "full", "files": manifest},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
