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
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dtfb.manifest import already_fetched, dedup_key, find_delisted, load_manifest, record_fetch
from dtfb.listing import expand_listing_url, is_vdp_url
from dtfb.scrape import vin_from_url
from dtfb.vehicle_pipeline import process_vehicle, HeroOptions
from dtfb.imaging.interior import InteriorSubjectClassifier
from dtfb.imaging.dedupe import DEFAULT_TEMPLATES_DIR, JunkFilter
from dtfb.imaging.classify import (
    AngleClassifier,
    InteriorExteriorTiebreakClassifier,
    SceneClassifier,
    SpareTireClassifier,
    WheelDetailClassifier,
)

# Hero video rendering is pure CPU (ffmpeg + PIL, no GPU model to load --
# see vehicle_pipeline.py::render_vehicle_video's docstring) so it overlaps
# for free with the NEXT vehicle's scrape/CLIP/cutout work, which IS
# GPU-bound. 3 workers matches the 3 video formats rendered per vehicle, so
# one vehicle's whole video job (all 3 formats) can run fully in parallel
# with itself as well as with the next vehicle's synchronous stage.
VIDEO_WORKERS = min(3, os.cpu_count() or 1)
# How many vehicles' worth of video jobs are allowed to be in flight before
# the main loop blocks on the oldest -- bounds memory/disk (an in-flight
# job holds its own cutout PNGs + ffmpeg process) rather than letting a
# slow video queue grow unboundedly across a 150-vehicle run.
MAX_INFLIGHT_VEHICLES = 2

# Matches dtfb.py's own argparse defaults -- inventory_sync.py doesn't
# expose every knob cli.py does (this is the cron-friendly, opinionated
# path, not the fully-configurable one), it just needs to call
# process_vehicle() with the same sane defaults a plain `dtfb <url>` run
# would use.
STICKER_DPI = 200
UPSCALE_MODEL = "swinir"

import requests
from playwright.sync_api import sync_playwright

# Sanity threshold on the delist step, ported from dtfb.py's --sync guard
# (decision this exists to prevent -- confirmed real incident, not
# hypothetical: a listing crawl that silently returned fewer vehicles than
# the site actually has -- a network blip, a stale/cached page, or a
# not-yet-fixed count-parsing bug -- makes every vehicle outside that
# short crawl look "delisted" even though it's still live. A dealer's real
# inventory almost never turns over this fast in one sync cycle, so a
# spike this size means the crawl was the problem, not the inventory.
MAX_DELIST_FRACTION = 0.3


def load_config(args) -> dict:
    """Load settings from —config file and/or env vars.

    inventory_url precedence (highest wins): --inventory-url, DTFB_INVENTORY_URL,
    --scope (resolved against dealer-config.json's/env's inventory_urls),
    dealer-config.json's own bare inventory_url. There's deliberately no
    built-in fallback URL -- unlike the cosmetic dealer_name/greeting/
    address defaults, a wrong inventory URL means scraping a real site by
    accident, so an unconfigured dealer gets a clear error instead of
    silently syncing whichever dealership this tool happened to be
    written against."""
    import dtfb.dealer_config as dc

    config = {}

    # File
    if args.config:
        dc.reload(args.config)
    cfg = dc.get()

    # Environment overrides
    import os
    for env_key, attr in [
        ("DTFB_INVENTORY_URL", "inventory_url"),
        ("DTFB_LISTINGS_ROOT", "listings_root"),
    ]:
        val = os.environ.get(env_key)
        if val:
            config[attr] = val

    if args.scope and "inventory_url" not in config:
        try:
            config["inventory_url"] = dc.resolve_scope_url(cfg, args.scope)
        except ValueError as e:
            raise SystemExit(str(e))

    if cfg.inventory_url:
        config.setdefault("inventory_url", cfg.inventory_url)

    # CLI overrides (highest precedence, including over --scope)
    if args.inventory_url:
        config["inventory_url"] = args.inventory_url
    if args.out:
        config["listings_root"] = args.out

    if "inventory_url" not in config:
        raise SystemExit(
            "No inventory URL configured. Pass one of:\n"
            "  --inventory-url <url>            explicit listing URL\n"
            "  --scope <name>                   e.g. --scope used, resolved via "
            "dealer-config.json's inventory_urls or DTFB_INVENTORY_URL_<NAME>\n"
            "  DTFB_INVENTORY_URL=<url>         env var\n"
            "  \"inventory_url\" in --dealer-config's JSON file")
    config.setdefault("listings_root",
                       os.environ.get("DTFB_LISTINGS_ROOT",
                                      str(Path.home() / "Documents" / "listings")))
    return config


def _already_delisted_folders(out_root: Path, manifest: dict, crawled_buckets: set) -> set:
    """Folder keys (from manifest entries in the given buckets) whose own
    details.json already carries delisted_at from a prior cycle -- see the
    mass-delist fraction comment above for why this matters. Reads every
    crawled-bucket manifest entry's details.json once -- hundreds, not
    thousands, and a small JSON read each; a few hundred ms even on a
    slow disk, not worth restricting to just the "missing" subset for."""
    out = set()
    for k, e in manifest.items():
        if not k.startswith("vin:"):
            continue
        folder = e.get("folder", "")
        if folder.split("/", 1)[0] not in crawled_buckets:
            continue
        try:
            data = json.loads((out_root / folder / "details.json").read_text())
        except (OSError, ValueError):
            continue
        v = data.get("vehicle", data)
        if v.get("delisted_at"):
            out.add(folder)
    return out


def run_sync(config: dict, dry_run: bool = False, headed: bool = False,
             allow_mass_delist: bool = False, vision_seat_check: bool = False) -> dict:
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
        "photos_pending": 0,
        "listing_vehicle_count": 0,
    }

    # Step 1: expand listing URL
    print(f"Scanning inventory: {inventory_url}")
    if dry_run:
        print("[dry-run] skipping live listing expansion")
        summary["listing_vehicle_count"] = 0
        summary["duration_seconds"] = round(time.time() - start, 1)
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

    # process_vehicle() defaults hero_opts=None, which SKIPS hero image/
    # video generation and the vision seat check entirely (confirmed real
    # gap this closes: a bare sync used to only produce raw photos + sticker
    # data, none of the finished deliverables the rest of the tool exists
    # to make). Built once, not per-vehicle, so the CLIP backbone each
    # classifier shares (see imaging/classify.py) is only loaded once for
    # the whole cycle, same reasoning as cli.py's own construction.
    junk_filter = JunkFilter(DEFAULT_TEMPLATES_DIR)
    classifier = SceneClassifier()
    angle_classifier = AngleClassifier()
    wheel_classifier = WheelDetailClassifier()
    spare_classifier = SpareTireClassifier()
    interior_tiebreak_classifier = InteriorExteriorTiebreakClassifier()
    interior_classifier = InteriorSubjectClassifier() if vision_seat_check else None
    # h264_nvenc over the HeroOptions default libx264 -- confirmed real cost
    # of the CPU default: hardware-encoding this machine's idle NVIDIA GPU
    # sits through anyway, x264 software encoding measured at ~45s PER
    # video format (square/vertical/horizontal each re-encode the same
    # edit) via real file-mtime timing on a completed vehicle -- ~66% of
    # total per-vehicle processing time. `ffmpeg -encoders` confirms
    # h264_nvenc is available and a live test encode succeeded on this
    # machine before this was turned on by default.
    #
    # vision_seat_check is deliberately OFF here even when the CLI flag was
    # passed -- it now runs as a single end-of-run batch pass (below)
    # instead of per-vehicle. Per-vehicle, infer_seat_config() unloads the
    # local Ollama model after every call (keep_alive=0, see
    # imaging/seat_vision.py) to avoid sharing GPU VRAM with CLIP/rembg/
    # SAM2 -- correct in isolation, but it means every vehicle pays a full
    # cold model load (measured 18-63s) for one photo's worth of inference.
    # Batching keeps the model warm across the run and unloads it once at
    # the very end, same safety property, paid once instead of N times.
    hero_opts = HeroOptions(interior_classifier=interior_classifier, vision_seat_check=False,
                             video_encoder="h264_nvenc")

    to_fetch = []
    for url in urls:
        already = already_fetched(out_root, url)
        if already:
            print(f"  [skip] {url} — already fetched at {already['fetched_at']}")
            summary["skipped"] += 1
        else:
            to_fetch.append(url)

    new_folders: list[Path] = []

    def _drain_video_futures(pending: list, block: bool) -> None:
        """Log and remove every finished (folder, fmt, future) triple.
        With block=True, waits for the OLDEST to finish first (used to cap
        how many vehicles' worth of video jobs are in flight); with
        block=False, only removes ones already done (used at the very end,
        after everything else has had a chance to finish)."""
        if block and pending:
            pending[0][2].result()
        i = 0
        while i < len(pending):
            folder, fmt, future = pending[i]
            if not future.done():
                i += 1
                continue
            pending.pop(i)
            try:
                report = future.result()
            except Exception as e:
                print(f"    ! hero video ({fmt}) failed for {folder.parent.name}/{folder.name}: {e}",
                      file=sys.stderr)
                continue
            if report is None:
                continue
            fallback_note = (f" [fell back to {report['encoder']}, GPU was too busy for h264_nvenc]"
                              if report.get("encoder") != "h264_nvenc" else "")
            print(f"    hero video ({fmt}) [{folder.parent.name}/{folder.name}]: "
                  f"{Path(report['out_path']).name} ({report['duration_s']}s, "
                  f"{report['file_size_mb']} MB, {report['n_shots']} shots){fallback_note}")

    if to_fetch:
        pending_video_futures: list = []
        # in-flight vehicle count == distinct folders with an unresolved
        # video future, not len(pending_video_futures) (3 formats each).
        with sync_playwright() as p, ProcessPoolExecutor(max_workers=VIDEO_WORKERS) as video_executor:
            for i, url in enumerate(to_fetch):
                if i > 0:
                    time.sleep(2)
                in_flight = len({f for f, _fmt, _fut in pending_video_futures})
                while in_flight >= MAX_INFLIGHT_VEHICLES:
                    _drain_video_futures(pending_video_futures, block=True)
                    in_flight = len({f for f, _fmt, _fut in pending_video_futures})
                try:
                    folder = process_vehicle(p, session, url, out_root, STICKER_DPI, headed, junk_filter,
                                              classifier, False, UPSCALE_MODEL, angle_classifier,
                                              hero_opts, wheel_classifier, spare_classifier,
                                              interior_tiebreak_classifier,
                                              video_executor=video_executor,
                                              pending_video_futures=pending_video_futures)
                    # process_vehicle() already recorded photos_pending in
                    # the manifest (see vehicle_pipeline.py) -- read it back
                    # rather than re-deriving it, so this stays in sync with
                    # whatever the pipeline itself decided. A pending
                    # vehicle skips the seat-vision batch below (no interior
                    # photo to check yet) and doesn't count as a normal
                    # "new" completion -- it'll be retried automatically on
                    # a future sync via already_fetched()'s self-heal.
                    entry = load_manifest(out_root).get(dedup_key(url), {})
                    if entry.get("photos_pending"):
                        print(f"  [pending] {folder.parent.name}/{folder.name} -- no real photos yet")
                        summary["photos_pending"] += 1
                    else:
                        print(f"  [ok]   {folder.parent.name}/{folder.name}")
                        summary["new"] += 1
                        new_folders.append(folder)
                except Exception as e:
                    print(f"  [FAIL] {url}: {e}", file=sys.stderr)
                    summary["failed"] += 1
                _drain_video_futures(pending_video_futures, block=False)

            # Everything's been scraped/processed -- wait out whatever video
            # jobs are still rendering before moving on (the executor's own
            # __exit__ would block here anyway; doing it explicitly lets us
            # log each one as it lands instead of going silent).
            while pending_video_futures:
                _drain_video_futures(pending_video_futures, block=True)

        # Batch seat-vision pass (see hero_opts note above): one photo's
        # worth of inference per new vehicle, model kept warm between calls
        # instead of reloaded per vehicle, unloaded after the last one.
        if vision_seat_check and new_folders:
            from dtfb.imaging.seat_vision import extract_seat_config, ollama_available

            if not ollama_available():
                print("  ! vision seat check requested but Ollama isn't reachable -- skipping batch")
            else:
                for i, folder in enumerate(new_folders):
                    keep_alive = 0 if i == len(new_folders) - 1 else 60
                    try:
                        result = extract_seat_config(folder, classifier=interior_classifier,
                                                      keep_alive=keep_alive)
                    except Exception as e:
                        print(f"    ! vision seat check failed for {folder.parent.name}/{folder.name}: {e}",
                              file=sys.stderr)
                        continue
                    if result is not None:
                        (folder / "vision-equipment.json").write_text(json.dumps(result, indent=2) + "\n")
                        print(f"    vision seat check [{folder.parent.name}/{folder.name}]: "
                              f"{result['config']} ({result['confidence']} confidence, "
                              f"from {result['source_photo']})")

    # Step 3: detect delisted vehicles
    live_vins = {vin_from_url(u) for u in urls} - {None}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Scope the comparison to whichever bucket(s) this crawl actually covered.
    # A used-only (or new-only) inventory URL only ever sees VINs from that
    # bucket, so comparing against the WHOLE manifest makes every vehicle in
    # the other bucket look "delisted" -- that's what tripped the mass-delist
    # valve on the first used-only run (651 new-bucket entries flagged missing
    # out of 861 total). Bucket is derived from the folder prefix manifest
    # entries are already keyed with ("used/..." / "new/..."), matching the
    # same "used" if "used" in url.lower() else "new" rule condition_bucket()
    # falls back to for bare listing URLs (scrape.py).
    crawled_buckets = {("used" if "used" in u.lower() else "new") for u in urls}
    manifest = load_manifest(out_root)

    # Both sides of the mass-delist fraction below have to be "still-active
    # inventory" only, not the whole manifest's history. A vehicle already
    # flagged delisted_at in an EARLIER cycle stays in the manifest forever
    # (nothing ever prunes it) and was still missing from live_vins THIS
    # cycle too, since it's still not listed -- so without this filter it
    # re-counts as "missing" every single run, permanently inflating the
    # fraction as more of the lot naturally turns over across cycles.
    # Confirmed real: after one cycle correctly flagged 70 real sold
    # vehicles, the very next cycle (zero new churn) still measured
    # 68/217 = 31% and tripped the valve on nothing but its own prior work.
    already_delisted = _already_delisted_folders(out_root, manifest, crawled_buckets)
    manifest_vin_count = sum(
        1 for k, e in manifest.items()
        if k.startswith("vin:") and e.get("folder", "").split("/", 1)[0] in crawled_buckets
        and e.get("folder") not in already_delisted
    )
    delisted = [
        e for e in find_delisted(out_root, live_vins)
        if e.get("folder", "").split("/", 1)[0] in crawled_buckets
        and e.get("folder") not in already_delisted
    ]
    if not listing_complete and not allow_mass_delist:
        print(f"  !! Listing crawl was PARTIAL -- refusing to delist anything this cycle. "
              f"A partial crawl can't tell 'no longer listed' apart from 'just wasn't on the "
              f"pages we managed to fetch'. Re-run once the crawl completes cleanly, or pass "
              f"--confirm-mass-delist if you understand the risk.")
        summary["delisted"] = 0
        summary["delist_skipped_reason"] = "partial_crawl"
        summary["duration_seconds"] = round(time.time() - start, 1)
        return summary
    if manifest_vin_count and len(delisted) / manifest_vin_count > MAX_DELIST_FRACTION and not allow_mass_delist:
        print(f"  !! {len(delisted)}/{manifest_vin_count} previously-scraped vehicles are missing "
              f"from this listing crawl -- that's above the {MAX_DELIST_FRACTION:.0%} sanity "
              f"threshold. This almost always means the crawl was too narrow, not that this many "
              f"vehicles actually sold. Refusing to flag anything. Pass --confirm-mass-delist if "
              f"this delisting is genuinely expected.")
        summary["delisted"] = 0
        summary["delist_skipped_reason"] = "mass_delist_threshold"
        summary["duration_seconds"] = round(time.time() - start, 1)
        return summary

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
    parser.add_argument("--scope", help="Named inventory scope (e.g. \"used\", \"new\", \"all\") -- "
                                         "resolved against dealer-config.json's/env's inventory_urls "
                                         "instead of typing the dealer's actual URL every time. Scope "
                                         "names are whatever you configure; there's no fixed set. "
                                         "Overridden by --inventory-url if both are given.")
    parser.add_argument("--out", help="Override listings output root")
    parser.add_argument("--dry-run", action="store_true",
                         help="Scan listing and show what would happen, without fetching")
    parser.add_argument("--headed", action="store_true",
                         help="Show browser window (debugging)")
    parser.add_argument("--vision-seat-check", action="store_true",
                         help="Run the local-vision front-seat-config check (needs `ollama serve` "
                              "with gemma4:e2b pulled). Off by default -- see HeroOptions.vision_seat_check.")
    parser.add_argument("--confirm-mass-delist", action="store_true",
                         help=f"Allow delisting despite a red flag: either a partial listing "
                              f"crawl, or more than {round(MAX_DELIST_FRACTION * 100)}%% of the "
                              "manifest missing from the crawl in one cycle. Off by default -- "
                              "both almost always mean the crawl didn't cover the full "
                              "inventory, not that this many vehicles genuinely sold at once.")
    args = parser.parse_args()

    config = load_config(args)
    summary = run_sync(config, dry_run=args.dry_run, headed=args.headed,
                        allow_mass_delist=args.confirm_mass_delist,
                        vision_seat_check=args.vision_seat_check)

    print()
    print("── Sync summary ──────────────────────────────")
    print(f"  Inventory URL:  {config['inventory_url']}")
    print(f"  Output root:    {config['listings_root']}")
    print(f"  Vehicles found: {summary['listing_vehicle_count']}")
    print(f"  New:            {summary['new']}")
    print(f"  Photos pending: {summary['photos_pending']}" +
          (" (no real photos yet -- will retry next sync)" if summary['photos_pending'] else ""))
    print(f"  Skipped:        {summary['skipped']}")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Delisted:       {summary['delisted']}" +
          (f" (skipped: {summary['delist_skipped_reason']})" if summary.get("delist_skipped_reason") else ""))
    print(f"  Duration:       {summary['duration_seconds']}s")
    if summary.get("error"):
        print(f"  ERROR:          {summary['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
