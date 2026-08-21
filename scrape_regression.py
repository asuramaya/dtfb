#!/usr/bin/env python3
"""
Regression harness for scrape.py's VDP parsing.

scrape.py is the highest-blast-radius module in this repo -- everything
else (photos, composition, posts) works off whatever it extracts -- and
until now it had zero test coverage: a page-layout tweak on the dealer's
site, or a refactor here, could silently corrupt every field with nothing
to catch it before it ships. This snapshots six real rendered VDP pages
(scrape_fixtures/pages/) spanning the shapes that have burned this project
before -- New vs. Used vs. Certified Pre-Owned, gas/diesel/electric,
sedan/SUV/truck/cargo van -- and diffs normalize_vehicle()'s output against
scrape_fixtures/expected.json.

Snapshots, not live fetches: the site is behind a Cloudflare challenge and
production inventory rotates constantly (a VIN scraped today is gone next
month), so a live-fetch test would be flaky by construction and unusable
offline. Pinning the rendered HTML is what makes this deterministic.

The Honda Accord fixture is deliberately a real confirmed 1-owner vehicle,
and the Escape/F-350/Mach-E fixtures deliberately are not -- see
scrape.py::extract_carfax_one_owner's docstring for why that split matters
(it's the only field the analytics blob gets flat-out wrong on every used
vehicle, confirmed False even on ones the page visibly badges "1 OWNER").

Usage:
    python scrape_regression.py

Adding cases: fetch+save a rendered VDP with fetch_rendered_html() (see
scrape.py), drop it in scrape_fixtures/pages/, then regenerate its expected
entry the same way this file's git history did -- run normalize_vehicle()
against it and eyeball the result before pinning it, don't hand-type
expected values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "scrape_fixtures"


# Real VIN/URL pairs from the fixture pages themselves -- pins
# vin_from_url() against actual dealer.com URL shapes, and (with the
# WRONG vin paired in) that process_vehicle()'s "scraped VIN doesn't match
# the URL" guard actually has something real to compare against. That
# guard exists because a dead/mistyped VDP URL can 404 into a page that
# still embeds a DIFFERENT real vehicle's schema.org data (a "similar
# vehicles" widget) -- confirmed real case on tomballford.com -- and
# without this check that silently "succeeds" under the wrong vehicle's
# name instead of failing loudly. See vehicle_pipeline.py::process_vehicle.
VIN_URL_CASES = [
    ("https://www.tomballford.com/vehicle/1FMCU0G67NUC01787/Used-2022-Ford-Escape-Tomball-TX/",
     "1FMCU0G67NUC01787"),
    ("https://www.tomballford.com/vehicle/1FT8W3DM5TEC78576/Used-2026-Ford-F--350SD-Tomball-TX/",
     "1FT8W3DM5TEC78576"),
    ("https://www.tomballford.com/inventory/new-vehicles/", None),  # not a VDP URL at all
]


def check_vin_from_url() -> tuple[int, int]:
    from scrape import vin_from_url

    failures = 0
    for url, expect in VIN_URL_CASES:
        got = vin_from_url(url)
        if got == expect:
            print(f"ok   vin_from_url: {url}")
        else:
            print(f"FAIL vin_from_url: {url} -- expected {expect}, got {got}")
            failures += 1
    return len(VIN_URL_CASES), failures


def main() -> int:
    from scrape import normalize_vehicle

    manifest = json.loads((FIXTURES_DIR / "expected.json").read_text())

    total = failures = 0
    for entry in manifest:
        page = FIXTURES_DIR / "pages" / entry["page"]
        expect = entry["expect"]
        total += 1
        if not page.exists():
            print(f"FAIL {entry['page']}: snapshot page missing")
            failures += 1
            continue

        v = normalize_vehicle("https://example.invalid/regression-fixture", page.read_text())
        got = {k: getattr(v, k) for k in expect if k in v.__dataclass_fields__}
        got["has_carfax_url"] = bool(v.carfax_url)
        got["has_window_sticker_url"] = bool(v.window_sticker_url)
        got["photo_count"] = len(v.photo_urls)
        got["main_features_count"] = len(v.main_features)
        got["warnings"] = v.warnings

        if got == expect:
            print(f"ok   {entry['page']}")
        else:
            diff = {k: (expect[k], got[k]) for k in expect if expect[k] != got[k]}
            print(f"FAIL {entry['page']}: {diff}")
            failures += 1

    n, f = check_vin_from_url()
    total += n
    failures += f

    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
