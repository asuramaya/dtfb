"""
Dealer-specific configuration — one place for all the values that change when
pointing lotstretcher at a different dealership.

Every module that previously hardcoded a greeting, address, city tag, or
dealer-specific behaviour now reads it from here. The loader tries, in order:

  1. Environment variables (LOTSTRETCHER_DEALER_NAME, LOTSTRETCHER_DEALER_GREETING, etc.)
  2. A JSON file at `--dealer-config` or `$LOTSTRETCHER_CONFIG`
  3. Built-in defaults (the Tomball Ford originals)

Usage:
    from lotstretcher.dealer_config import load

    cfg = load()
    print(cfg.dealer_greeting)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DealerConfig:
    """All the per-dealer values the pipeline needs."""

    # -- Post boilerplate ---------------------------------------------------
    dealer_name: str = "Tomball Ford"
    dealer_greeting: str = "Ask for Hector Chavez!"
    dealer_address: str = "22702 TX-249, Tomball, TX 77375"

    # -- Social-media tags (Instagram / Threads) ----------------------------
    city_tags: list[str] = field(default_factory=lambda: [
        "Tomball", "TomballTX", "Houston", "HoustonCars", "TomballFord",
    ])

    # -- Which manufacturer "more details" resolvers are available -----------
    # Keyed by lowercased make name.  Each entry is a module path:
    #   "package.module:function"
    # The function receives the Vehicle and returns a URL or None.
    manufacturer_links: dict[str, str] = field(default_factory=lambda: {
        "ford": "facebook_post:ford_qr_link",
    })

    # -- Border / branding --------------------------------------------------
    # Default border tag used when composing hero images.
    default_border_tag: str = "dealer-frame"

    # -- Dealer website (for the scraper) -----------------------------------
    # Used as hints; the scraper is CMS-agnostic and reads the page's
    # embedded data regardless of domain.
    dealer_domain: Optional[str] = None
    inventory_url: Optional[str] = None

    # Named inventory scopes (e.g. "used" / "new" / "all", but the names
    # are whatever the dealer's own site uses -- there's no fixed set).
    # Lets `--scope used` stand in for the dealer's actual used-inventory
    # URL without every user of this tool having to remember or retype
    # it. Deliberately EMPTY by default, unlike this class's other
    # Tomball-originals defaults -- these values become live HTTP
    # requests, so silently inheriting a *different* dealer's real
    # inventory URL would mean scraping the wrong site by accident, not
    # just a wrong caption. A user must set these themselves (config file
    # or LOTSTRETCHER_INVENTORY_URL_<SCOPE> env vars, see load()) before `--scope`
    # does anything; resolve_scope_url() below refuses loudly if it's not
    # configured rather than guessing.
    inventory_urls: dict[str, str] = field(default_factory=dict)

    # -- Recraft API key (AI background generation) -------------------------
    recraft_api_key: Optional[str] = None


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(f"LOTSTRETCHER_{key}", default)


def _json_path() -> Path | None:
    """Return the config file path, if one was given."""
    explicit = os.environ.get("LOTSTRETCHER_CONFIG")
    if explicit:
        return Path(explicit)
    # Common locations
    for candidate in ("lotstretcher-config.json", "config.json", ".lotstretcher-config.json"):
        p = Path(candidate)
        if p.exists():
            return p
    return None


def load(path: str | Path | None = None) -> DealerConfig:
    """Load dealer configuration, merging env vars over file defaults.

    Priority (highest wins):
      1. Explicit environment variables
      2. Config file fields (JSON)
      3. Built-in defaults
    """
    cfg = DealerConfig()

    # Layer 1: file
    src = Path(path) if path else _json_path()
    if src and src.exists():
        raw = json.loads(src.read_text(encoding="utf-8"))
        for key in ("dealer_name", "dealer_greeting", "dealer_address",
                     "default_border_tag", "dealer_domain", "inventory_url"):
            if raw.get(key):
                setattr(cfg, key, raw[key])
        if raw.get("city_tags"):
            cfg.city_tags = raw["city_tags"]
        if raw.get("manufacturer_links"):
            cfg.manufacturer_links.update(raw["manufacturer_links"])
        if raw.get("inventory_urls"):
            cfg.inventory_urls.update(raw["inventory_urls"])

    # Layer 2: env vars
    for env_key, attr in [
        ("DEALER_NAME", "dealer_name"),
        ("DEALER_GREETING", "dealer_greeting"),
        ("DEALER_ADDRESS", "dealer_address"),
        ("DEFAULT_BORDER_TAG", "default_border_tag"),
        ("DEALER_DOMAIN", "dealer_domain"),
        ("INVENTORY_URL", "inventory_url"),
        ("RECRAFT_API_KEY", "recraft_api_key"),
    ]:
        val = _env(env_key)
        if val is not None:
            setattr(cfg, attr, val)

    # LOTSTRETCHER_INVENTORY_URL_<SCOPE> -- scanned rather than a fixed list since
    # scope names are open-ended (whatever a dealer's own site calls its
    # inventory sections, not just "used"/"new"/"all"). Wins over the same
    # scope's config-file entry, matching every other env-over-file field
    # above.
    prefix = "LOTSTRETCHER_INVENTORY_URL_"
    for env_key, val in os.environ.items():
        if env_key.startswith(prefix) and val:
            cfg.inventory_urls[env_key[len(prefix):].lower()] = val

    return cfg


def resolve_scope_url(cfg: DealerConfig, scope: str) -> str:
    """The URL for a named inventory scope (e.g. "used"), or a refusal
    that names what's actually configured -- never a guess. Scope names
    and their URLs are entirely dealer-defined (see inventory_urls'
    docstring); this only ever reads what a user configured."""
    url = cfg.inventory_urls.get(scope)
    if url:
        return url
    if cfg.inventory_urls:
        available = ", ".join(sorted(cfg.inventory_urls))
        raise ValueError(f"no inventory URL configured for scope {scope!r} -- configured scopes: {available} "
                          f"(dealer-config.json's inventory_urls, or LOTSTRETCHER_INVENTORY_URL_{scope.upper()})")
    raise ValueError(f"no inventory scopes configured at all -- add an \"inventory_urls\" object to your "
                      f"dealer-config.json (e.g. {{\"used\": \"https://yourdealer.com/inventory/used/\"}}) "
                      f"or set LOTSTRETCHER_INVENTORY_URL_{scope.upper()}, or just pass the URL directly with "
                      f"--inventory-url")


# Module-level convenience — import and use directly when no custom path is
# needed.  Lazy-loaded so importing this module doesn't immediately parse a
# config file (which might not exist yet during install / first import).
_cached: DealerConfig | None = None


def get() -> DealerConfig:
    global _cached
    if _cached is None:
        _cached = load()
    return _cached


def reload(path: str | Path | None = None) -> DealerConfig:
    """Force-reload config (useful in tests or after a config-file change)."""
    global _cached
    _cached = load(path)
    return _cached
