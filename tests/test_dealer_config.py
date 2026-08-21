"""Tests for the dealer configuration layer."""

import json
from pathlib import Path

import pytest

from dealer_config import DealerConfig, load, reload


def test_defaults() -> None:
    """Built-in defaults are the Tomball Ford originals."""
    cfg = load()
    assert cfg.dealer_name == "Tomball Ford"
    assert cfg.dealer_greeting == "Ask for Hector Chavez!"
    assert cfg.dealer_address == "22702 TX-249, Tomball, TX 77375"
    assert "Tomball" in cfg.city_tags
    assert "ford" in cfg.manufacturer_links
    # Reset cache so other tests start clean too
    reload()


def test_from_json_file() -> None:
    """A JSON config file overrides defaults."""
    reload()  # reset to defaults first

    config = {
        "dealer_name": "Another Dealer",
        "dealer_greeting": "Ask for Jane!",
        "dealer_address": "456 Oak St, Othertown, ST 67890",
        "city_tags": ["Othertown", "OthertownCars"],
    }
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    cfg = reload(config_path)
    assert cfg.dealer_name == "Another Dealer"
    assert cfg.dealer_greeting == "Ask for Jane!"
    assert cfg.dealer_address == "456 Oak St, Othertown, ST 67890"
    assert cfg.city_tags == ["Othertown", "OthertownCars"]
    assert cfg.dealer_domain is None

    reload()  # clean up


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override file values."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps({"dealer_greeting": "Ask from file!"}), encoding="utf-8")

    monkeypatch.setenv("DTFB_DEALER_GREETING", "Ask from env!")
    monkeypatch.delenv("DTFB_CONFIG", raising=False)

    cfg = reload(config_path)
    assert cfg.dealer_greeting == "Ask from env!"
    reload()


def test_dealer_config_dataclass() -> None:
    """The dataclass has all expected fields."""
    cfg = DealerConfig()
    assert hasattr(cfg, "dealer_name")
    assert hasattr(cfg, "dealer_greeting")
    assert hasattr(cfg, "dealer_address")
    assert hasattr(cfg, "city_tags")
    assert hasattr(cfg, "manufacturer_links")
    assert hasattr(cfg, "default_border_tag")
    assert hasattr(cfg, "dealer_domain")
    assert hasattr(cfg, "inventory_url")
    assert hasattr(cfg, "recraft_api_key")


def test_reload_clears_cache() -> None:
    """reload() clears the cached config."""
    from dealer_config import get as _get
    cfg1 = _get()
    cfg2 = reload()
    assert cfg2.dealer_name == cfg1.dealer_name
