"""Tests for the fetch manifest."""

import json
from pathlib import Path

import pytest

import lotstretcher.manifest as fetch_manifest


SAMPLE_MANIFEST = {
    "vin:1HGCY1F24SA123456": {
        "vin": "1HGCY1F24SA123456",
        "url": "https://www.tomballford.com/vehicle/1HGCY1F24SA123456/used/",
        "fetched_at": "2025-01-15T10:00:00+0000",
        "folder": "2025-Honda-Accord-LX-SA123456",
    },
    "vin:1FTYE1C8XTKB49701": {
        "vin": "1FTYE1C8XTKB49701",
        "url": "https://www.tomballford.com/vehicle/1FTYE1C8XTKB49701/",
        "fetched_at": "2025-01-16T12:00:00+0000",
        "folder": "2026-Ford-Transit-KB49701",
    },
}

# Non-VIN URL entry for url-key fallback tests
NON_VIN_ENTRY: dict = {
    "url:https://example.com/inventory": {
        "vin": None,
        "url": "https://example.com/inventory",
        "fetched_at": "2025-01-17T12:00:00+0000",
        "folder": "inventory-snapshot",
    },
}


def _write_manifest(tmp_path: Path, data: dict = SAMPLE_MANIFEST) -> None:
    """Write manifest.json and create each vehicle folder with details.json."""
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    for entry in data.values():
        folder = tmp_path / entry["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "details.json").write_text("{}")


def test_extract_vin_from_url() -> None:
    """VIN extracted from a standard DealerInspire VDP URL."""
    url = "https://www.tomballford.com/vehicle/1HGCY1F24SA123456/used/"
    assert fetch_manifest.extract_vin_from_url(url) == "1HGCY1F24SA123456"


def test_extract_vin_from_url_no_match() -> None:
    """No VIN when the path has no 17-char token."""
    assert fetch_manifest.extract_vin_from_url("https://www.example.com/") is None


def test_already_fetched_vin(tmp_path: Path) -> None:
    """A manifest entry with a matching VIN yields already_fetched."""
    _write_manifest(tmp_path)
    url = "https://www.tomballford.com/vehicle/1HGCY1F24SA123456/different-slug/"
    result = fetch_manifest.already_fetched(tmp_path, url)
    assert result is not None
    assert result["vin"] == "1HGCY1F24SA123456"


def test_already_fetched_not_found(tmp_path: Path) -> None:
    """A VIN with no manifest entry returns None."""
    _write_manifest(tmp_path)
    url = "https://www.otherdealer.com/vehicle/2ABCDEFGHIJKLMNOP/"
    assert fetch_manifest.already_fetched(tmp_path, url) is None


def test_already_fetched_by_vin(tmp_path: Path) -> None:
    """Same VIN across different URL paths still matches."""
    _write_manifest(tmp_path)
    url = "https://www.tomballford.com/vehicle/1FTYE1C8XTKB49701/new/"
    result = fetch_manifest.already_fetched(tmp_path, url)
    assert result is not None
    assert result["folder"] == "2026-Ford-Transit-KB49701"


def test_already_fetched_by_normalized_url(tmp_path: Path) -> None:
    """Non-VIN URLs fall back to normalized URL matching."""
    all_data = {**SAMPLE_MANIFEST, **NON_VIN_ENTRY}
    _write_manifest(tmp_path, all_data)
    url = "https://example.com/inventory/"
    result = fetch_manifest.already_fetched(tmp_path, url)
    assert result is not None
    assert result["folder"] == "inventory-snapshot"


def test_already_fetched_self_healing(tmp_path: Path) -> None:
    """If the vehicle folder is missing, already_fetched returns None."""
    (tmp_path / "manifest.json").write_text(json.dumps({
        "vin:1HGCY1F24SA123456": {
            "vin": "1HGCY1F24SA123456",
            "url": "https://example.com/vehicle/1HGCY1F24SA123456/",
            "fetched_at": "2025-01-15T10:00:00+0000",
            "folder": "non-existent-folder",
        },
    }))
    url = "https://example.com/vehicle/1HGCY1F24SA123456/"
    assert fetch_manifest.already_fetched(tmp_path, url) is None


def test_find_delisted(tmp_path: Path) -> None:
    """Vehicles in the manifest but not in the live VIN set are delisted."""
    _write_manifest(tmp_path)
    live_vins = {"1FTYE1C8XTKB49701"}  # Transit still live, Accord is missing
    delisted = fetch_manifest.find_delisted(tmp_path, live_vins)
    folders = [d["folder"] for d in delisted]
    assert "2025-Honda-Accord-LX-SA123456" in folders
    assert "2026-Ford-Transit-KB49701" not in folders


def test_dedup_key_vin() -> None:
    """A VIN-shaped URL produces a vin: key."""
    key = fetch_manifest.dedup_key("https://www.tomballford.com/vehicle/1HGCY1F24SA123456/")
    assert key == "vin:1HGCY1F24SA123456"


def test_dedup_key_url() -> None:
    """A non-VIN URL produces a url: key."""
    key = fetch_manifest.dedup_key("https://www.tomballford.com/inventory/")
    assert key.startswith("url:")
