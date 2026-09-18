"""Shared fixtures for lotstretcher tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_manifest(tmp_path: Path) -> Path:
    """A minimal manifest with two vehicles."""
    manifest = {
        "vin:1HGCY1F24SA123456": {
            "vin": "1HGCY1F24SA123456",
            "url": "https://example.com/vehicle/1HGCY1F24SA123456/",
            "fetched_at": "2025-01-15T10:00:00+0000",
            "folder": "2025-Honda-Accord-LX-SA123456",
        },
        "url:https://example.com/vehicle/OTHER/": {
            "vin": None,
            "url": "https://example.com/vehicle/OTHER/",
            "fetched_at": "2025-01-16T12:00:00+0000",
            "folder": "2025-Other-Vehicle-OTHER",
        },
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p
