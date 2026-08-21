"""
Lookup helper for composition media (backgrounds, borders, and the video/
audio the animated hero uses) in ../assets/, indexed by
assets/manifest.json so files can have illegible names on disk (or come
from a future AI-generation step) and still be found by theme.

Add new media by dropping the file in the matching assets/ subfolder and
adding an entry to manifest.json with a name and a few tags -- no code
changes needed. Keeping this as a flat lookup (not a class, not a
database) on purpose: the moment we wire in AI-generated backgrounds or
borders, this is the one place that needs to grow, and it should stay
easy to see the whole picture.

Every lookup here is category-agnostic (find_one/resolve_arg take the
category as a string), so the per-category list_*() helpers below are
conveniences, not the mechanism -- a new category needs a manifest key
and nothing else.

Media a composition depends on lives HERE, in the repo, not wherever it
happened to be downloaded to. The flag clip and the audio loop were
passed as absolute paths out of ~/Downloads for a while, which meant the
video pipeline silently depended on one machine's home directory and on
files that could be renamed or cleaned up at any time.
"""
from __future__ import annotations

import json
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"backgrounds": [], "borders": []}
    return json.loads(MANIFEST_PATH.read_text())


def _list(category: str, tag: str | None = None) -> list[dict]:
    entries = _load_manifest().get(category, [])
    if tag:
        entries = [e for e in entries if tag in e.get("tags", [])]
    return entries


def list_backgrounds(tag: str | None = None) -> list[dict]:
    return _list("backgrounds", tag)


def list_borders(tag: str | None = None) -> list[dict]:
    return _list("borders", tag)


def list_videos(tag: str | None = None) -> list[dict]:
    return _list("videos", tag)


def list_audio(tag: str | None = None) -> list[dict]:
    return _list("audio", tag)


def resolve(entry: dict) -> Path:
    return ASSETS_DIR / entry["file"]


def entry_for(category: str, value: str | None, default_name: str | None = None) -> dict:
    """The manifest ENTRY resolve_arg() would resolve to, rather than just
    its path -- for callers that need the metadata alongside the file
    (hero_video_cli wants the audio entry's `bars`, which is a property of
    that specific recording, not something to hardcode)."""
    entries = _load_manifest().get(category, [])
    if not entries:
        raise ValueError(f"No {category} in assets/manifest.json yet.")
    path = resolve_arg(category, value, default_name=default_name)
    for e in entries:
        if resolve(e) == path:
            return e
    raise ValueError(f"No {category} entry matches {path}")


def find_one(category: str, name_or_tag: str) -> Path:
    """Find a single asset by exact name (case-insensitive) or by tag.
    Raises if there's no match or more than one -- composing should fail
    loudly on an ambiguous pick, not silently guess."""
    entries = _load_manifest().get(category, [])
    by_name = [e for e in entries if e.get("name", "").lower() == name_or_tag.lower()]
    if len(by_name) == 1:
        return resolve(by_name[0])
    by_tag = [e for e in entries if name_or_tag in e.get("tags", [])]
    if len(by_tag) == 1:
        return resolve(by_tag[0])
    if not by_name and not by_tag:
        raise ValueError(f"No {category} match {name_or_tag!r} by name or tag "
                          f"(run with --list-assets to see what's available)")
    raise ValueError(f"{name_or_tag!r} matches multiple {category}, be more specific")


def resolve_arg(category: str, value: str | None, default_name: str | None = None) -> Path:
    """CLI-friendly asset lookup shared by dtfb.py and compose_cli.py: an
    exact name, a tag, or unset. Unset falls back to `default_name` (e.g.
    "American Flag") if that entry exists, else the first entry in the
    manifest -- so composing still works with zero flags even before a
    named default has been added."""
    entries = _load_manifest().get(category, [])
    if not entries:
        raise ValueError(f"No {category} in assets/manifest.json yet -- add one (see generate_assets.py) first.")
    if value is not None:
        return find_one(category, value)
    if default_name:
        for e in entries:
            if e.get("name", "").lower() == default_name.lower():
                return resolve(e)
    return resolve(entries[0])
