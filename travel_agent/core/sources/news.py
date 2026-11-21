"""GDELT news adapter with a conservative rate limiter and RSS fallback."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
import threading
import xml.etree.ElementTree as ET
from urllib.parse import quote

from ..http import request, source_result, http_mode
from ..models import NewsItem, SourceResult


_last_gdelt_request = 0.0
_gdelt_lock = threading.Lock()


def _published(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            for pattern in ("%Y%m%d%H%M%S", "%Y%m%d%H%M"):
                try:
                    return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return None


def parse_gdelt(payload: dict) -> list[NewsItem]:
    return [NewsItem(title=item.get("title", ""), url=item.get("url", ""), source=item.get("domain"), published_at=_published(item.get("seendate")), provider="gdelt") for item in payload.get("articles", [])]


def parse_google_rss(xml_text: str) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.findall("./channel/item"):
        items.append(NewsItem(title=item.findtext("title", ""), url=item.findtext("link", ""), source=item.findtext("source"), published_at=_published(item.findtext("pubDate")), provider="google_news"))
    return items


def _gdelt(query: str) -> list[NewsItem]:
    global _last_gdelt_request
    with _gdelt_lock:
        wait = 5.5 - (time.monotonic() - _last_gdelt_request)
        if wait > 0:
            time.sleep(wait)
        _last_gdelt_request = time.monotonic()
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote(query)}&mode=artlist&format=json"
    body = request("GET", url)[1]
    return parse_gdelt(json.loads(body))


def search(query: str) -> SourceResult[list[NewsItem]]:
    gdelt_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote(query)}&mode=artlist&format=json"
    rss_url = f"https://news.google.com/rss/search?q={quote(query + ' when:3d')}&hl=en-US&gl=US&ceid=US:en"

    def load() -> list[NewsItem]:
        try:
            return parse_google_rss(request("GET", rss_url, timeout=5.0)[1])
        except Exception as rss_error:
            if (http_mode.get() or "").lower() == "replay":
                raise RuntimeError("news not captured in this snapshot") from rss_error
            try:
                return _gdelt(query)
            except Exception as gdelt_error:
                raise RuntimeError(f"news unavailable: RSS={rss_error}; GDELT={gdelt_error}") from gdelt_error

    return source_result("news", gdelt_url, load)
