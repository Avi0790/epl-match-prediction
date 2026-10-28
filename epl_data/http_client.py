"""Small sequential HTTP client: caching, immutable raw versions, and bounded retries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from .store import utc_now


class AccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Download:
    path: Path
    observed_at: str
    url: str
    cached: bool


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # API keys must never be forwarded to another host through a redirect.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AccessError("Redirect refused for an authenticated request")


class HttpClient:
    def __init__(self, raw_dir: Path, interval: float = 2.0, timeout: float = 30.0,
                 retries: int = 2, request_budget: int = 100):
        if interval < 0 or timeout <= 0 or retries < 0 or request_budget < 1:
            raise ValueError("Invalid HTTP client settings")
        self.root = Path(raw_dir)
        self.interval, self.timeout, self.retries = interval, timeout, retries
        self.budget, self.requests = request_budget, 0
        self._last = 0.0

    def get(self, source: str, url: str, *, refresh: bool = False,
            headers: dict | None = None) -> Download:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise AccessError("Only HTTPS URLs without embedded credentials are accepted")
        # Credentials only belong in headers, never query strings or persisted metadata.
        if any(x in urllib.parse.parse_qs(parsed.query) for x in ("key", "token", "apikey", "api_key")):
            raise AccessError("Credentials in URLs are not supported")
        key = hashlib.sha256(url.encode()).hexdigest()
        directory = self.root / source
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / f"{key}.json"
        if manifest.exists() and not refresh:
            info = json.loads(manifest.read_text())
            path = directory / info["filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError("Raw cache checksum mismatch; inspect the file before refreshing")
            return Download(path, info["observed_at"], url, True)
        request_headers = {"User-Agent": "EPLResearchIngest/0.1 (academic data ingestion)", **(headers or {})}
        opener = urllib.request.build_opener(_NoRedirect()) if headers else urllib.request.build_opener()
        for attempt in range(self.retries + 1):
            if self.requests >= self.budget:
                raise AccessError(f"Per-run request budget ({self.budget}) exhausted; cached progress is retained")
            time.sleep(max(0.0, self.interval - (time.monotonic() - self._last)))
            self.requests += 1
            self._last = time.monotonic()
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    payload = response.read(25_000_001)
                    if len(payload) > 25_000_000:
                        raise AccessError("Response exceeds 25 MB; narrow the request")
                    if not payload:
                        raise AccessError("Provider returned an empty response")
                observed = utc_now()
                digest = hashlib.sha256(payload).hexdigest()
                filename = f"{key}-{digest}.raw"
                path = directory / filename
                if not path.exists():
                    temp = path.with_suffix(".tmp")
                    temp.write_bytes(payload)
                    temp.replace(path)
                # Reuse only an unchanged consecutive snapshot. A -> B -> A is a new revision.
                previous = json.loads(manifest.read_text()) if manifest.exists() else {}
                if previous.get("sha256") == digest:
                    observed = previous["observed_at"]
                stamp = observed.replace(":", "").replace(".", "")
                history = directory / f"{key}-{digest}-{stamp}.metadata.json"
                info = {"url": url, "sha256": digest, "filename": filename,
                        "observed_at": observed, "last_checked_at": utc_now(), "bytes": len(payload)}
                for target in (history, manifest):
                    temp = target.with_suffix(".tmp")
                    temp.write_text(json.dumps(info, indent=2) + "\n")
                    temp.replace(target)
                return Download(path, observed, url, False)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise AccessError(f"{source}: HTTP {exc.code}; authorized access is required") from None
                if exc.code == 404:
                    raise AccessError(f"{source}: HTTP 404; dataset/season not available") from None
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise AccessError(f"{source}: HTTP {exc.code}; no data ingested") from None
                delay = min(2 ** (attempt + 1), 30)
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                if delay > 60:
                    raise AccessError("Provider requested a long cooldown; retry after Retry-After expires") from None
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt == self.retries:
                    raise AccessError(f"{source}: network failure ({type(exc).__name__})") from None
                time.sleep(min(2 ** (attempt + 1), 30))
        raise AssertionError("Retry loop exhausted")
