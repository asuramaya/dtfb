"""
Fetch manifest: skip re-scraping a vehicle that's already been pulled down,
so re-running dtfb on the same URL (a duplicate line in a batch file, an
accidental re-run, a URL list that overlaps a previous run) doesn't burn a
full Cloudflare-bypass browser session + CLIP/rembg pipeline for nothing.

Keyed by VIN when one can be pulled straight out of the URL (Tomball
Ford's VDP URLs embed it as a path segment: /vehicle/<VIN>/...), which is
what makes the skip actually save compute -- no fetch needed to know the
identity of the vehicle. Falls back to the normalized URL as the key for
any URL that doesn't shake out a VIN, which still dedupes exact repeats
within/across runs, just can't skip *before* the first fetch reveals
whatever this URL turns out to be.

One manifest per output directory (<out_root>/manifest.json), since
--out is what defines "one batch of listings" here.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")

MANIFEST_FILENAME = "manifest.json"


def extract_vin_from_url(url: str) -> str | None:
    """Best-effort VIN pull from the URL path alone -- no fetch required.
    Tomball Ford's VDP URLs always embed it (/vehicle/<VIN>/...); this is
    just a generic 17-char VIN-shaped token search so it keeps working if
    the path layout ever shifts slightly."""
    m = _VIN_RE.search(urlsplit(url).path)
    return m.group(1) if m else None


def _normalize_url(url: str) -> str:
    """Strips query string/fragment and a trailing slash -- just enough to
    catch the same page requested two slightly different ways, not a full
    canonicalization."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def dedup_key(url: str) -> str:
    vin = extract_vin_from_url(url)
    return f"vin:{vin}" if vin else f"url:{_normalize_url(url)}"


def _manifest_path(out_root: Path) -> Path:
    return Path(out_root) / MANIFEST_FILENAME


def load_manifest(out_root: Path) -> dict:
    path = _manifest_path(out_root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(out_root: Path, data: dict) -> None:
    """Writes via a temp file + atomic rename, not in place -- a plain
    write_text() left the file truncated (confirmed: simulated the exact
    bytes a killed process leaves behind) if the process dies mid-write,
    and load_manifest()'s corrupt-JSON fallback then silently treats the
    WHOLE manifest as empty, not just the interrupted entry. That means
    every vehicle already fetched looks brand new to the next run --
    wasteful (a full re-scrape + re-process of everything), not
    destructive to the scraped data itself, but avoidable for the cost of
    a rename."""
    path = _manifest_path(out_root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def already_fetched(out_root: Path, url: str) -> dict | None:
    """Returns the manifest entry if this URL/VIN was already fetched AND
    its output folder is still there with at least details.json in it --
    self-healing against a manifest entry whose folder got deleted or
    moved by hand, so that doesn't permanently and silently skip a vehicle."""
    manifest = load_manifest(out_root)
    entry = manifest.get(dedup_key(url))
    if entry is None:
        return None
    folder = Path(out_root) / entry["folder"]
    if not (folder / "details.json").exists():
        return None
    return entry


def find_delisted(out_root: Path, live_vins: set[str]) -> list[dict]:
    """Manifest entries whose VIN isn't in `live_vins` -- i.e. vehicles
    this pipeline scraped at some point that the dealer's site no longer
    lists. `live_vins` should come from a FULL, unfiltered listing crawl
    (see listing.py::expand_listing_url against an all-inventory URL) --
    passing anything narrower would misreport every vehicle outside that
    filter as delisted. Only compares entries keyed by VIN (url: keys
    never map onto a VIN, so there is nothing to compare); returns each
    entry plus its VIN so a caller can flag it without a second manifest
    read. Advisory, like everything else in this module -- callers decide
    what "delisted" should mean for their output (a warning, an archive
    move, ...); this only detects it."""
    manifest = load_manifest(out_root)
    out = []
    for key, entry in manifest.items():
        if not key.startswith("vin:"):
            continue
        vin = entry.get("vin") or key.removeprefix("vin:")
        if vin not in live_vins:
            out.append(entry)
    return out


def record_fetch(out_root: Path, url: str, folder_name: str, vin: str | None = None,
                  stock_number: str | None = None) -> None:
    manifest = load_manifest(out_root)
    manifest[dedup_key(url)] = {
        "url": url,
        "folder": folder_name,
        "vin": vin,
        "stock_number": stock_number,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_manifest(out_root, manifest)
