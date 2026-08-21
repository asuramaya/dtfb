"""
Recraft API asset generation -- backgrounds and (later) borders, generated
on demand and saved into assets/ with a manifest.json entry, so the same
lookup in imaging/assets.py works whether an asset was hand-picked or
AI-generated. Nothing here runs automatically; it's called explicitly via
generate_assets.py (a paid API call per image, on purpose never a silent
side effect of the main scraping pipeline).

Requires RECRAFT_API_KEY in .env at the repo root (gitignored -- never log,
print, or commit that key; _api_key() only ever reads it).

API reference (verified against recraft.ai/docs, Aug 2026): POST
https://external.api.recraft.ai/v1/images/generations, Bearer auth, JSON
body {prompt, model, size, n, response_format}. Default model is
recraftv4_1 ($0.035/image) rather than the Pro tier (6x the cost) -- bump
via the model= kwarg if a specific generation needs the higher quality.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"
ASSETS_DIR = ROOT / "assets"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"

API_URL = "https://external.api.recraft.ai/v1/images/generations"
DEFAULT_MODEL = "recraftv4_1"
DEFAULT_SIZE = "1024x1024"


def _load_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _api_key() -> str:
    key = os.environ.get("RECRAFT_API_KEY") or _load_env().get("RECRAFT_API_KEY")
    if not key:
        raise RuntimeError("RECRAFT_API_KEY not set (expected in .env at the repo root)")
    return key


def generate_image(prompt: str, model: str = DEFAULT_MODEL, size: str = DEFAULT_SIZE,
                    style: str | None = None) -> bytes:
    """One paid API call. Returns raw image bytes (Recraft returns a URL;
    this downloads it so callers don't have to)."""
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    body = {"prompt": prompt, "model": model, "size": size, "n": 1, "response_format": "url"}
    if style:
        body["style"] = style

    resp = requests.post(API_URL, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    image_url = resp.json()["data"][0]["url"]

    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    return img_resp.content


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "asset"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"backgrounds": [], "borders": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def _add_asset(category: str, dest: Path, name: str, tags: list[str], prompt: str) -> None:
    manifest = _load_manifest()
    manifest.setdefault(category, []).append({
        "file": f"{category}/{dest.name}",
        "name": name,
        "tags": [*tags, "ai-generated"],
        "prompt": prompt,
    })
    _save_manifest(manifest)


def generate_background(prompt: str, name: str, tags: list[str] | None = None, **kwargs) -> Path:
    content = generate_image(prompt, **kwargs)
    dest = ASSETS_DIR / "backgrounds" / f"{_slugify(name)}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    _add_asset("backgrounds", dest, name, tags or [], prompt)
    return dest
