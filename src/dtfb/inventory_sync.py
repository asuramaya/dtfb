"""
Automated inventory sync — the daemonless cron-friendly companion.

Ties together dtfb's listing expansion, fetch pipeline, manifest tracking,
and delist detection into a single command that's safe to run from cron.

Usage:
    inventory_sync.py                                          # uses defaults
    inventory_sync.py --config /path/to/dealer-config.json
    inventory_sync.py --dry-run                                # preview only

Environment variables (see README.md):
    DTFB_INVENTORY_URL     — dealer inventory listing URL
    DTFB_LISTINGS_ROOT     — where to store output
    DTFB_CONFIG            — path to dealer config file
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dtfb.manifest import already_fetched, find_delisted, load_manifest, record_fetch
from dtfb.listing import expand_listing_url, is_vdp_url
from dtfb.scrape import vin_from_url
from dtfb.vehicle_pipeline import process_vehicle, HeroOptions

import requests
from playwright.sync_api import sync_playwright


def load_config(args) -> dict:
    """Load settings from —config file and/or env vars."""
    config = {}

    # File
    if args.config:
        import dtfb.dealer_config as dc
        dc.reload(args.config)

    # Environment overrides
    import os
    for env_key, attr in [
        ("DTFB_INVENTORY_URL", "inventory_url"),
        ("DTFB_LISTINGS_ROOT", "listings_root"),
    ]:
        val = os.environ.get(env_key)
        if val:
            config[attr] = val

    # CLI overrides
    if args.inventory_url:
        config["inventory_url"] = args.inventory_url
    if args.out:
        config["listings_root"] = args.out

    config.setdefault("inventory_url",
                       os.environ.get("DTFB_INVENTORY_URL",
                                      "https://www.tomballford.com/inventory/all-vehicles/"))
    config.setdefault("listings_root",
                       os.environ.get("DTFB_LISTINGS_ROOT",
                                      str(Path.home() / "Documents" / "listings")))
    return config


def run_sync(config: dict, dry_run: bool = False, headed: bool = False) -> dict:
    """Execute one sync cycle. Returns a summary dict."""
    out_root = Path(config["listings_root"])
    inventory_url = config["inventory_url"]
    out_root.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inventory_url": inventory_url,
        "dry_run": dry_run,
        "new": 0,
        "skipped": 0,
        "failed": 0,
        "delisted": 0,
        "listing_vehicle_count": 0,
    }

    # Step 1: expand listing URL
    print(f"Scanning inventory: {inventory_url}")
    if dry_run:
        print("[dry-run] skipping live listing expansion")
        summary["listing_vehicle_count"] = 0
        return summary

    with sync_playwright() as p:
        try:
            urls, listing_complete, expected_total = expand_listing_url(p, inventory_url, headed=headed)
        except Exception as e:
            print(f"!! Failed to expand listing: {e}", file=sys.stderr)
            summary["error"] = str(e)
            summary["listing_vehicle_count"] = 0
            return summary

    summary["listing_vehicle_count"] = len(urls)
    summary["listing_complete"] = listing_complete
    print(f"  Found {len(urls)} vehicle(s)" +
          (f" (site reports {expected_total})" if expected_total else "") +
          ("" if listing_complete else " (PARTIAL crawl)"))

    if not urls:
        print("  No vehicles found in listing.")
        return summary

    # Step 2: process each vehicle
    session = requests.Session()
    session.headers.update({"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )})

    to_fetch = []
    for url in urls:
        already = already_fetched(out_root, url)
        if already:
            print(f"  [skip] {url} — already fetched at {already['fetched_at']}")
            summary["skipped"] += 1
        else:
            to_fetch.append(url)

    if to_fetch:
        with sync_playwright() as p:
            for i, url in enumerate(to_fetch):
                if i > 0:
                    time.sleep(2)
                try:
                    folder = process_vehicle(p, session, url, out_root)
                    print(f"  [ok]   {folder.parent.name}/{folder.name}")
                    summary["new"] += 1
                except Exception as e:
                    print(f"  [FAIL] {url}: {e}", file=sys.stderr)
                    summary["failed"] += 1

    # Step 3: detect delisted vehicles
    live_vins = {vin_from_url(u) for u in urls} - {None}
    delisted = find_delisted(out_root, live_vins)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    newly_flagged = 0
    for entry in delisted:
        details_path = out_root / entry["folder"] / "details.json"
        try:
            data = json.loads(details_path.read_text())
        except (OSError, ValueError):
            continue
        v = data.get("vehicle", data)
        if v.get("delisted_at"):
            continue
        if not dry_run:
            v["delisted_at"] = now
            details_path.write_text(json.dumps(data, indent=2) + "\n")
        newly_flagged += 1
        print(f"  [delisted] {entry['folder']}" +
              (" (dry-run)" if dry_run else ""))

    summary["delisted"] = newly_flagged
    summary["duration_seconds"] = round(time.time() - start, 1)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to dealer JSON config file")
    parser.add_argument("--inventory-url", help="Override inventory listing URL")
    parser.add_argument("--out", help="Override listings output root")
    parser.add_argument("--dry-run", action="store_true",
                         help="Scan listing and show what would happen, without fetching")
    parser.add_argument("--headed", action="store_true",
                         help="Show browser window (debugging)")
    args = parser.parse_args()

    config = load_config(args)
    summary = run_sync(config, dry_run=args.dry_run, headed=args.headed)

    print()
    print("── Sync summary ──────────────────────────────")
    print(f"  Inventory URL:  {config['inventory_url']}")
    print(f"  Output root:    {config['listings_root']}")
    print(f"  Vehicles found: {summary['listing_vehicle_count']}")
    print(f"  New:            {summary['new']}")
    print(f"  Skipped:        {summary['skipped']}")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Delisted:       {summary['delisted']}")
    print(f"  Duration:       {summary['duration_seconds']}s")
    if summary.get("error"):
        print(f"  ERROR:          {summary['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
