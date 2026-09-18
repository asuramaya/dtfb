"""
Build a vehicle inventory feed (CSV/TSV) from vehicles already scraped by
lotstretcher, in the shape third-party listing platforms (Meta/Facebook
Automotive Inventory Ads, and by convergent convention most others)
expect to ingest.

WHY THIS EXISTS: there is no open, universal automotive inventory feed
standard (confirmed via research, 2026-09) -- every DMS/inventory-
management-system integration (HomeNet, vAuto, Dealer.com) is a bespoke,
gatekept, vendor-credentialed relationship, not something a small tool can
self-serve into. But EXPORTING a feed needs nobody's permission -- you
just produce a file in the shape the destination already expects. That's
what this does: lotstretcher already has everything a feed needs (VIN, price,
mileage, colors, the dealer's own real public photo URLs) sitting in
details.json from the normal scrape -- this just re-shapes it.

FIELD-MAPPING CAVEAT, read before trusting this in production: Meta's own
canonical field-by-field spec for Automotive Inventory Ads lives behind a
JS-rendered, partially login-gated Business Help Center page that could
not be fetched directly while building this. FEED_COLUMNS below is built
from a field set that converged consistently across several independent
secondary sources (vehicle_id, title, url, make/model/year/trim, vin,
mileage.value/mileage.unit, price, state_of_vehicle, body_style,
image[N].url, address) -- reasonably trustworthy as a shape, but NOT
verified against Meta's own feed validator. Run the output through Meta
Commerce Manager's feed validator (or your platform's equivalent) before
relying on it, and adjust FEED_COLUMNS/vehicle_to_row() -- deliberately
kept as one flat mapping table, not scattered logic -- if reality differs.

Usage:
    from lotstretcher.export_feed import build_rows, write_csv
    rows = build_rows(vehicle_folders)
    write_csv(rows, Path("feed.csv"))
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

MAX_IMAGES = 20

# vehicle_to_row() output key -> feed column header. One flat table so a
# real spec mismatch is a one-line fix, not a hunt through logic.
FEED_COLUMNS = [
    "vehicle_id", "title", "description", "url",
    "make", "model", "year", "trim", "vin", "body_style",
    "mileage.value", "mileage.unit",
    "price", "currency", "state_of_vehicle", "availability",
    "exterior_color", "interior_color", "transmission", "drivetrain", "fuel_type",
    "address", "dealer_name", "dealer_phone",
] + [f"image[{i}].url" for i in range(MAX_IMAGES)]

# condition (lotstretcher's own scraped value) -> state_of_vehicle (feed value).
# Certified is sold as a used vehicle everywhere downstream of this feed;
# there is no separate "certified" state_of_vehicle value in any source
# checked.
CONDITION_TO_STATE = {
    "new": "new",
    "used": "used",
    "certified pre-owned": "used",
    "certified": "used",
}


def _vehicle_record(folder: Path) -> dict:
    try:
        data = json.loads((folder / "details.json").read_text())
    except (OSError, ValueError):
        return {}
    return data.get("vehicle", data)


def _clean_price(display_price) -> int | None:
    """display_price on disk is whatever type the dealer's own raw
    payload happened to store it as -- a bare number for this dealer, but
    facebook_post.py's own _to_float() exists because other payloads give
    a formatted string ("$35,382"). A feed's price field needs a bare
    number either way."""
    try:
        return int(round(float(str(display_price).replace(",", "").replace("$", "").lstrip("+"))))
    except (TypeError, ValueError):
        return None


def vehicle_to_row(folder: Path) -> dict | None:
    """One feed row for one vehicle folder, or None if it's missing the
    two fields nothing downstream can work without (price, at least one
    photo) -- mirrors the one hard requirement every source agreed on."""
    v = _vehicle_record(folder)
    if not v:
        return None
    if v.get("delisted_at"):
        return None  # sold/removed -- don't advertise it

    price = _clean_price(v.get("display_price"))
    photos = v.get("photo_urls") or []
    if not price or not photos:
        return None

    row = {
        "vehicle_id": v.get("vin") or v.get("stock_number") or folder.name,
        "title": v.get("title") or f"{v.get('year', '')} {v.get('make', '')} {v.get('model', '')}".strip(),
        # Feed description fields commonly cap in the low thousands of
        # characters; 2000 is a conservative slice of lotstretcher's own
        # dealer_description (real ones run to ~2800 chars, see
        # PKD51678's 2772-char example) that stays well under any
        # limit seen cited without truncating mid-sentence too abruptly.
        "description": (v.get("dealer_description") or "")[:2000],
        "url": v.get("url", ""),
        "make": v.get("make", ""),
        "model": v.get("model", ""),
        "year": v.get("year", ""),
        "trim": v.get("trim", ""),
        "vin": v.get("vin", ""),
        "body_style": v.get("body_type", ""),
        "mileage.value": v.get("mileage", ""),
        "mileage.unit": "MI",
        "price": price,
        "currency": "USD",
        "state_of_vehicle": CONDITION_TO_STATE.get((v.get("condition") or "").lower(), "used"),
        "availability": "in stock",
        "exterior_color": v.get("exterior_color_generic") or v.get("exterior_color_factory", ""),
        "interior_color": v.get("interior_color", ""),
        "transmission": v.get("transmission", ""),
        "drivetrain": v.get("drivetrain", ""),
        "fuel_type": v.get("fuel_type", ""),
        "address": v.get("dealer_address", ""),
        "dealer_name": v.get("dealer_name", ""),
        "dealer_phone": v.get("dealer_phone") or "",
    }
    for i, url in enumerate(photos[:MAX_IMAGES]):
        row[f"image[{i}].url"] = url
    return row


def build_rows(folders: list[Path]) -> list[dict]:
    """One row per vehicle folder with enough data to advertise; folders
    that fail vehicle_to_row()'s checks are silently skipped (sold,
    delisted, or missing the two hard-required fields) -- not an error,
    just not feed-worthy yet."""
    rows = []
    for folder in folders:
        row = vehicle_to_row(folder)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path, delimiter: str = ",") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEED_COLUMNS, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FEED_COLUMNS})
