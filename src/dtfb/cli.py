#!/usr/bin/env python3
"""
dealer-to-fb: pull a Tomball Ford vehicle detail page (VDP) into a folder of
photos, a window sticker screenshot, and a Facebook-ready copy/paste post.

Usage:
    dtfb <vdp-url> [<vdp-url> ...]
    dtfb --file urls.txt
    dtfb <vdp-url> --out listings
    dtfb <filtered-inventory-listing-url>       # expands to every vehicle it lists
    dtfb <listing-url> --dry-run                # preview what it would expand to, without fetching

This file is just the CLI surface -- argument parsing and the batch loop.
The actual work is split across:
    scrape.py            fetch the VDP, parse it into a Vehicle record
    listing.py           expand a filtered inventory listing URL into VDP URLs
    photos.py            download + sort + cutout the photo gallery
    window_sticker.py    download/render/split the window sticker PDF
    facebook_post.py     build the ready-to-paste post text
    vehicle_pipeline.py  process_vehicle(): wires the above together for one URL
    manifest.py          tracks completed fetches so re-running is a no-op
    imaging/             the CV pipeline (CLIP sorting, cutouts, compose, sticker PDF parsing)
See each module's own docstring for how its piece works.

Each vehicle gets one output folder, organized so source material and
finished deliverables don't mix:
    <vehicle-folder>/
        images/                  originals + cutouts (see imaging/ above)
        window-sticker.pdf       source document, if one was found
        window-sticker.json      structured data parsed from it (imaging/sticker.py)
        bundle/                  everything meant to actually be posted/shared:
            hero.png               the composed hero shot, square
            hero-portrait.png      same hero at 4:5 for Instagram/Facebook feed
            hero-video.mp4         animated hero, square (Marketplace/feed)
            hero-video-vertical.mp4    same edit, 9:16 (Reels/Stories/TikTok/Shorts)
            hero-video-horizontal.mp4  same edit, 16:9 (YouTube)
            framed/                every exterior cutout solo-composed just as big,
                                     including any confirmed wheel money shots
                                     (see imaging/wheel.py -- most vehicles won't have one)
            window-sticker-*a/b.png  rendered sticker, split into 2 readable panels
            facebook.txt            ready-to-paste Marketplace/Facebook post
            threads.txt             same vehicle inside Threads' 500-char cap
            instagram.txt           hook-first caption + hashtag block
        details.json             full scraped Vehicle record
<out>/manifest.json       cumulative fetch record across all runs (see --force)
<out>/run-summary.json    what happened in THIS run (see write_batch_summary())
<out>/runs.jsonl         append-only history of every run's outcomes, one row per URL
<out>/photo-hashes.json  per-vehicle gallery hashes, for spotting stock renders reused
                          across listings (see imaging/dedupe.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

import dealer_config
import dtfb.manifest as fetch_manifest
from dtfb.imaging import assets
from dtfb.imaging.classify import (
    AngleClassifier,
    InteriorExteriorTiebreakClassifier,
    SceneClassifier,
    SpareTireClassifier,
    WheelDetailClassifier,
)
from dtfb.imaging.interior import InteriorSubjectClassifier
from dtfb.imaging.dedupe import DEFAULT_TEMPLATES_DIR, JunkFilter
from dtfb.listing import expand_listing_url, is_vdp_url
from dtfb.scrape import USER_AGENT, vin_from_url
from dtfb.vehicle_pipeline import HeroOptions, log, process_vehicle


def expand_urls(raw_urls: list[str], headed: bool) -> tuple[list[str], list[str]]:
    """Vehicle detail page URLs pass through unchanged; anything else (a
    filtered inventory/listing/search-results page -- see listing.py) gets
    crawled and replaced with every VDP URL it links to, so the caller
    always ends up with a flat list of individual vehicles to process.
    De-duplicated by URL in case a vehicle shows up in more than one
    listing, or was also passed explicitly.

    Returns (all_urls, listing_derived_urls, listing_complete) -- the
    second list is the subset that came from actually crawling a listing
    page, as opposed to being named directly on the command line. --sync
    (see main()) needs that distinction: a listing crawl is a true
    snapshot of "what the site currently has", so anything previously
    scraped and missing from it is fair to call delisted -- but a
    hand-picked VDP URL says nothing about the rest of the inventory, and
    treating it as a full snapshot would misreport everything else as
    delisted.

    `listing_complete` is False if ANY listing crawl this call made was
    truncated by an error partway through (see expand_listing_url's
    docstring for the confirmed real incident this guards against) --
    --sync must refuse to delist anything when this is False, since
    "previously scraped but not on the pages we managed to fetch" is not
    the same thing as "no longer listed"."""
    listing_urls = [u for u in raw_urls if not is_vdp_url(u)]
    urls = [u for u in raw_urls if is_vdp_url(u)]
    from_listing = []
    listing_complete = True

    if listing_urls:
        with sync_playwright() as p:
            for lurl in listing_urls:
                log(f"==> expanding listing: {lurl}")
                try:
                    expanded, complete, expected_total = expand_listing_url(p, lurl, headed=headed)
                    total_note = f" (site reports {expected_total})" if expected_total is not None else ""
                    log(f"    found {len(expanded)} vehicle(s){total_note}" +
                        ("" if complete else " (PARTIAL crawl)"))
                    urls.extend(expanded)
                    from_listing.extend(expanded)
                    listing_complete = listing_complete and complete
                except Exception as e:
                    log(f"    !! failed to expand listing: {e}")
                    listing_complete = False

    seen = set()
    all_urls = [u for u in urls if not (u in seen or seen.add(u))]
    seen2 = set()
    listing_urls_out = [u for u in from_listing if not (u in seen2 or seen2.add(u))]
    return all_urls, listing_urls_out, listing_complete


def resolve_video_formats(requested: list[str] | None) -> tuple[str, ...]:
    """Which aspects to render. All three by default -- the same edit
    blocked for each frame shape, so one scrape covers Marketplace, Reels
    and YouTube without a second pass. Costs roughly 3x the render time
    (~3.5min vs ~1.2min per vehicle on GPU), so narrow it with
    --video-format square when that matters."""
    from dtfb.imaging.compose.hero_video import VIDEO_FORMATS

    if not requested:
        return tuple(VIDEO_FORMATS)
    if "all" in requested:
        return tuple(VIDEO_FORMATS)
    unknown = [f for f in requested if f not in VIDEO_FORMATS]
    if unknown:
        sys.exit(f"Unknown --video-format {unknown[0]!r}; choose from {', '.join(VIDEO_FORMATS)} or 'all'")
    return tuple(dict.fromkeys(requested))


def resolve_hero_formats(requested: list[str] | None) -> tuple[str, ...]:
    """Hero still shapes. Square (Marketplace) plus 4:5 portrait (the
    tallest in-feed render Instagram and Facebook allow) by default;
    vertical is Stories-only and the vertical VIDEO serves that better."""
    from dtfb.imaging.compose.pipeline import HERO_STILL_FORMATS

    if not requested:
        return ("square", "portrait")
    if "all" in requested:
        return tuple(HERO_STILL_FORMATS)
    unknown = [f for f in requested if f not in HERO_STILL_FORMATS]
    if unknown:
        sys.exit(f"Unknown --hero-format {unknown[0]!r}; choose from "
                 f"{', '.join(HERO_STILL_FORMATS)} or 'all'")
    return tuple(dict.fromkeys(requested))


def _read_summary_fields(folder: Path) -> dict:
    try:
        data = json.loads((folder / "details.json").read_text())
        return {"vin": data.get("vin"), "title": data.get("title"), "condition": data.get("condition")}
    except (OSError, json.JSONDecodeError):
        return {}


RUN_LOG_FILENAME = "runs.jsonl"

# --sync's safety valve -- see the check itself for why this exists.
MAX_DELIST_FRACTION = 0.3


def append_run_log(out_root: Path, results: list[dict]) -> None:
    """Append this run's outcomes to a durable log, one JSON object per URL.

    run-summary.json is overwritten every run by design (it answers "how
    did my LAST batch go"), and manifest.json only records SUCCESSES --
    record_fetch runs after a vehicle completes. So before this, a failure
    left no trace anywhere the moment the next run started, and the
    question "how often does this fail, and why" was unanswerable. A
    confirmed real failure (a CUDA OOM that killed a whole scrape) had
    already been lost that way.

    Append-only JSONL rather than a growing JSON document so a crashed run
    can't corrupt the history it's writing into.
    """
    with (out_root / RUN_LOG_FILENAME).open("a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({"run_at": r.get("_run_at"),
                                  **{k: v for k, v in r.items() if not k.startswith("_")}}) + "\n")


def write_batch_summary(out_root: Path, results: list[dict]) -> Path:
    """One record of what THIS run did (success/failed/skipped per URL),
    distinct from manifest.json's cumulative all-time fetch record -- lets
    you check "how did my last batch go" without diffing manifest.json or
    opening every vehicle folder. Always written when at least one URL was
    processed (including a run that's 100% skips). The per-run history
    lives in runs.jsonl -- see append_run_log()."""
    summary_path = out_root / "run-summary.json"
    summary_path.write_text(json.dumps({
        "run_at": results[0]["_run_at"] if results else None,
        "results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
    }, indent=2))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*", help="Vehicle detail page URL(s)")
    parser.add_argument("--file", help="Text file with one VDP URL per line")
    parser.add_argument("--out", default=str(Path.home() / "Documents" / "listings"),
                         help="Output directory (default: ~/Documents/listings)")
    parser.add_argument("--sticker-dpi", type=int, default=200, help="Window sticker PNG render DPI (default: 200)")
    parser.add_argument("--headed", action="store_true", help="Show the browser window (debugging)")
    parser.add_argument("--no-junk-filter", action="store_true",
                         help="Skip perceptual-hash filtering of known marketing/junk images (see imaging/templates/)")
    parser.add_argument("--junk-templates-dir", default=str(DEFAULT_TEMPLATES_DIR),
                         help="Directory of known junk/marketing reference images (default: imaging/templates/)")
    parser.add_argument("--no-photo-sort", action="store_true",
                         help="Skip CLIP/rembg exterior-interior sorting, transparent cutouts, angle "
                              "tagging, and interior letterbox cropping (faster, no GPU/ML deps used, "
                              "flat images/ layout)")
    parser.add_argument("--no-interiors", action="store_true",
                         help="Skip interior post-processing (bundle/interior/). By default every "
                              "interior photo is white-balanced and exposure-corrected -- the "
                              "photograph itself, at original resolution, with nothing drawn on it.")
    parser.add_argument("--interior-captions", action="store_true",
                         help="Also print a feature name onto interior photos that clearly depict "
                              "one the listing confirms. Off by default: text on a listing photo is "
                              "a merchandising decision, and it costs a CLIP pass. See "
                              "imaging/interior.py.")
    parser.add_argument("--vision-seat-check", action="store_true",
                         help="Ask a local vision model (gemma4:e2b via Ollama) whether the front row "
                              "is bucket seats with a console or a bench, from a real interior photo. "
                              "Fills a real gap: some window stickers never disclose front-seat "
                              "configuration in text at all. Off by default -- needs `ollama serve` "
                              "running locally with gemma4:e2b pulled; quietly skipped (not a failure) "
                              "if that's not available. Writes vehicle_folder/vision-equipment.json. "
                              "See imaging/seat_vision.py.")
    parser.add_argument("--no-strict-cutouts", action="store_true",
                         help="Keep every cutout, even ones the rest of the vehicle's gallery doesn't "
                              "vouch for. By default a cutout that matches neither the shape nor the "
                              "paint of its siblings is dropped and its photo moved to images/interior/ "
                              "-- that catches a shot taken through the windshield or one that cut out "
                              "another car on the lot. See imaging/gallery.py.")
    parser.add_argument("--upscale", action="store_true",
                         help="Run 4x super-resolution on the general exterior cutout gallery (off by "
                              "default: measured across the fleet, those cutouts land in the composed "
                              "hero/video at a median ~1.1x scale, where plain Lanczos already looks "
                              "fine -- 4x SR only pays for itself above ~2x, which is rare here. Wheel "
                              "money shots are the exception and always get it -- they scale ~2.1x on "
                              "average -- see photos.py and imaging/upscale.py.)")
    parser.add_argument("--upscale-model", default="swinir", choices=["swinir", "realesrnet"],
                         help="Model for wheel-shot upscaling (always on) and, with --upscale, the "
                              "general gallery too. swinir: sharper, ~4x slower (default). realesrnet: "
                              "faster, still non-hallucinating but softer.")
    parser.add_argument("--no-hero", action="store_true",
                         help="Skip hero image composition (hero.png + framed/*.png). On by default, "
                              "needs exterior cutouts (i.e. --no-photo-sort was NOT used).")
    parser.add_argument("--background", help="Background name or tag from assets/manifest.json "
                                              "(default: American Flag)")
    parser.add_argument("--border", help="Border name or tag from assets/manifest.json "
                                          "(default: first available)")
    parser.add_argument("--frame", action="store_true",
                         help="Composite the dealer frame onto the composed images (frameless is the default -- "
                              "contact info baked into the frame was getting posts shadowbanned).")
    parser.add_argument("--photo-background", action="store_true",
                         help="Use the shared background photo asset instead of the default per-image gradient "
                              "built from each vehicle's own exterior/interior colors. A single shared backdrop "
                              "across every post is what got flagged, so this is opt-in.")
    parser.add_argument("--no-video", action="store_true",
                         help="Skip the animated hero video (bundle/hero-video.mp4)")
    parser.add_argument("--video-music", action="store_true",
                         help="Score the hero video. Silent is the default; timing is bar/beat-locked either way.")
    parser.add_argument("--video-flag-background", action="store_true",
                         help="Use the backdrop video clip instead of the default rotating vehicle-color gradient.")
    parser.add_argument("--hero-format", action="append", metavar="FORMAT",
                         help="Shape(s) for the hero still; repeatable, or 'all'. square 1254x1254 "
                              "(Marketplace), portrait 1080x1350 (4:5 Instagram/Facebook feed), "
                              "vertical 1080x1920 (Stories). Default: square + portrait. "
                              "framed/ is always square.")
    parser.add_argument("--video-format", action="append", metavar="FORMAT",
                         help="Frame shape for the hero video; repeatable, or 'all'. "
                              "square 1254x1254 (Marketplace/feed), vertical 1080x1920 "
                              "(Reels/Stories/TikTok/Shorts), horizontal 1920x1080 (YouTube). "
                              "Default: all three. The edit is identical in every format -- only "
                              "the blocking changes.")
    parser.add_argument("--nvenc", action="store_true",
                         help="Encode the hero video on the GPU (h264_nvenc).")
    parser.add_argument("--no-glow", action="store_true", help="Disable the glow behind composed car cutouts")
    parser.add_argument("--glow-color", default="white", choices=["white", "blue", "gold", "red"])
    parser.add_argument("--glow-radius", type=int, default=24,
                         help="Glow blur radius in pixels (default: 24)")
    parser.add_argument("--glow-intensity", type=float, default=0.75,
                         help="Glow opacity, 0-1 (default: 0.75)")
    parser.add_argument("--dealer-config",
                         help="Path to a JSON dealer-config file (overrides env vars and defaults)")
    parser.add_argument("--list-assets", action="store_true",
                         help="List available hero-image backgrounds/borders and exit")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch even if this URL/VIN is already in <out>/manifest.json "
                              "from a previous run (default: skip it, no browser/pipeline cost paid)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Expand any listing URLs and print the resulting vehicle URLs, then exit "
                              "without fetching/processing anything -- preview a filter before committing "
                              "to a full run.")
    parser.add_argument("--sync", action="store_true",
                         help="After processing, flag any previously-scraped vehicle that's no longer "
                              "in this listing crawl as delisted (details.json gets a 'delisted_at' "
                              "timestamp; nothing is deleted). Combined with --force off (the default), "
                              "this is the daily-sync shape: new vehicles get the full pipeline, "
                              "already-known ones cost only the listing-page crawl, and sold/removed "
                              "ones get flagged. Requires at least one listing URL -- see manifest.py's "
                              "find_delisted().")
    parser.add_argument("--confirm-mass-delist", action="store_true",
                         help=f"Allow --sync to delist despite a red flag: either flagging more than "
                              f"{round(MAX_DELIST_FRACTION * 100)}%% of the manifest in one run, or the "
                              "listing crawl itself being interrupted partway through. Off by default -- "
                              "both almost always mean the listing crawl didn't actually cover the full "
                              "inventory, not that this many vehicles genuinely sold at once.")
    args = parser.parse_args()

    if args.dealer_config:
        dealer_config.reload(args.dealer_config)


    if args.list_assets:
        print("Backgrounds:")
        for e in assets.list_backgrounds():
            print(f"  {e['name']}  (tags: {', '.join(e.get('tags', []))})")
        print("\nBorders:")
        for e in assets.list_borders():
            print(f"  {e['name']}  (tags: {', '.join(e.get('tags', []))})")
        return

    raw_urls = list(args.urls)
    if args.file:
        raw_urls.extend(
            line.strip() for line in Path(args.file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not raw_urls:
        parser.error("Provide at least one vehicle or listing URL (or --file).")

    urls, listing_derived_urls, listing_complete = expand_urls(raw_urls, args.headed)
    if not urls:
        parser.error("No vehicle URLs found (a listing URL expanded to nothing, or none were provided).")

    if args.sync and not listing_derived_urls:
        parser.error("--sync needs at least one listing/inventory URL (not just individual vehicle "
                      "URLs) -- it works by comparing a full listing crawl against what's already "
                      "been scraped.")

    if args.sync and not listing_complete and not args.confirm_mass_delist:
        parser.error("--sync's listing crawl was interrupted partway through (see the '(PARTIAL crawl)' "
                      "warning above) -- refusing to run the delist step against an incomplete snapshot, "
                      "since anything on the pages that didn't load would look 'delisted' when it isn't. "
                      "Re-run once the crawl completes cleanly, or pass --confirm-mass-delist if you "
                      "understand the risk and want to proceed anyway.")

    if args.dry_run:
        print(f"{len(urls)} vehicle URL(s):")
        for u in urls:
            print(f"  {u}")
        return

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    junk_filter = None if args.no_junk_filter else JunkFilter(Path(args.junk_templates_dir))
    # SceneClassifier()/AngleClassifier()/WheelDetailClassifier()/
    # SpareTireClassifier()/InteriorExteriorTiebreakClassifier() share one
    # CLIP model by default (all fall back to the same default_backbone()
    # singleton) -- one GPU load, five label sets. See imaging/classify.py
    # and imaging/pipeline.py for what InteriorExteriorTiebreakClassifier
    # catches (a genuine exterior shot CLIP's main 5-way pass mislabels
    # "interior" outright).
    classifier = None if args.no_photo_sort else SceneClassifier()
    angle_classifier = None if args.no_photo_sort else AngleClassifier()
    wheel_classifier = None if args.no_photo_sort else WheelDetailClassifier()
    spare_classifier = None if args.no_photo_sort else SpareTireClassifier()
    interior_tiebreak_classifier = None if args.no_photo_sort else InteriorExteriorTiebreakClassifier()
    # Sixth label set on the same backbone -- what an interior photo
    # DEPICTS, for the confirmed feature callouts (imaging/interior.py).
    # Needs the sorted images/interior/ folder, so it's gated on photo
    # sorting like everything else.
    # Only paid for when something that actually needs it is requested --
    # the default interior path is pure image processing and needs no
    # model. --vision-seat-check also wants this classifier (to pick the
    # best candidate photo before spending an Ollama call on it), not just
    # --interior-captions.
    interior_classifier = (InteriorSubjectClassifier()
                            if ((args.interior_captions or args.vision_seat_check)
                                and not args.no_photo_sort and not args.no_interiors) else None)

    hero_opts = HeroOptions(enabled=not args.no_hero, glow=not args.no_glow, glow_color=args.glow_color,
                             glow_radius=args.glow_radius, glow_intensity=args.glow_intensity,
                             gradient=not args.photo_background, video=not args.no_video,
                             video_encoder="h264_nvenc" if args.nvenc else "libx264",
                             video_formats=resolve_video_formats(args.video_format),
                             hero_formats=resolve_hero_formats(args.hero_format),
                             interiors=not args.no_interiors and not args.no_photo_sort,
                             interior_captions=args.interior_captions,
                             interior_classifier=interior_classifier,
                             vision_seat_check=args.vision_seat_check and not args.no_photo_sort
                             and not args.no_interiors)
    if hero_opts.enabled:
        # Left as None, each of these means the generated/absent variant;
        # only resolve an asset when the flag (or an explicit name) asks
        # for one, so the manifest isn't consulted for things we won't use.
        try:
            if args.photo_background or args.background:
                hero_opts.background_path = assets.resolve_arg("backgrounds", args.background,
                                                                 default_name="American Flag")
                hero_opts.gradient = False
            if args.frame or args.border:
                hero_opts.border_path = assets.resolve_arg("borders", args.border)
            if args.video_flag_background:
                hero_opts.video_background = assets.resolve_arg("videos", None,
                                                                 default_name="American Flag Waving")
            if args.video_music:
                entry = assets.entry_for("audio", None, default_name="Its Mine")
                hero_opts.video_audio = assets.resolve(entry)
                hero_opts.video_bars_per_loop = int(entry.get("bars", 4))
        except ValueError as e:
            parser.error(str(e))

    run_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results = []
    to_fetch = []
    for url in urls:
        already = None if args.force else fetch_manifest.already_fetched(out_root, url)
        if already:
            log(f"==> {url}\n    already fetched {already['fetched_at']} -> "
                f"{already['folder']} (--force to re-fetch)")
            results.append({"url": url, "status": "skipped", "folder": already["folder"],
                             "vin": already.get("vin"), "_run_at": run_at})
        else:
            to_fetch.append(url)

    # Only spin up a browser session if there's actually something left to
    # fetch -- entering/exiting sync_playwright() with zero navigations
    # made it throw a harmless-but-noisy asyncio warning on shutdown.
    if to_fetch:
        with sync_playwright() as p:
            for i, url in enumerate(to_fetch):
                if i > 0:
                    time.sleep(2)  # a little pacing helps avoid harsher Cloudflare challenges
                try:
                    folder = process_vehicle(p, session, url, out_root, args.sticker_dpi, args.headed, junk_filter,
                                              classifier, args.upscale, args.upscale_model, angle_classifier,
                                              hero_opts, wheel_classifier, spare_classifier,
                                              interior_tiebreak_classifier, not args.no_strict_cutouts)
                    results.append({"url": url, "status": "success", "folder": folder.name,
                                     "_run_at": run_at, **_read_summary_fields(folder)})
                except Exception as e:
                    log(f"    !! FAILED: {e}")
                    results.append({"url": url, "status": "failed", "error": str(e), "_run_at": run_at})

    summary_path = write_batch_summary(out_root, results)
    append_run_log(out_root, results)

    failures = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    succeeded = [r for r in results if r["status"] == "success"]
    log("")
    log(f"Done: {len(succeeded)}/{len(urls)} succeeded"
        f"{f', {len(skipped)} already fetched (skipped)' if skipped else ''}, output in {out_root.resolve()}")
    log(f"Run summary: {summary_path}")

    if args.sync:
        # Only vehicles missing from a FULL listing crawl are fair to call
        # delisted (see expand_urls()'s docstring) -- listing_derived_urls
        # is exactly that set, restricted to VINs since anything without
        # one can't be matched against the manifest's vin: keys anyway.
        live_vins = {vin_from_url(u) for u in listing_derived_urls} - {None}
        delisted = fetch_manifest.find_delisted(out_root, live_vins)

        # Safety valve: confirmed real failure mode, not hypothetical --
        # pointing --sync at a NARROW/filtered listing URL (a single model
        # search, a typo'd filter) instead of a genuinely full-inventory
        # one makes every other vehicle in the manifest look "missing" and
        # mass-flags real live inventory as sold. A real dealer's day-to-
        # day churn is a handful of vehicles, never a third of the
        # manifest at once -- so a spike that size is far more likely to
        # mean "wrong URL" than "mass sell-off", and the safe default is
        # to refuse and say why rather than write it.
        manifest_vin_count = sum(1 for k in fetch_manifest.load_manifest(out_root) if k.startswith("vin:"))
        if manifest_vin_count and len(delisted) / manifest_vin_count > MAX_DELIST_FRACTION \
                and not args.confirm_mass_delist:
            log(f"\n!! --sync found {len(delisted)}/{manifest_vin_count} previously-scraped vehicles "
                f"missing from this listing crawl -- that's above the {MAX_DELIST_FRACTION:.0%} "
                "sanity threshold. This almost always means the listing URL is narrower than the full "
                "inventory (a filtered search, a typo), not that this many vehicles actually sold. "
                "Refusing to flag anything. If this delisting is genuinely expected, re-run with "
                "--confirm-mass-delist.")
            sys.exit(1)

        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        newly_flagged = []
        for entry in delisted:
            details_path = out_root / entry["folder"] / "details.json"
            try:
                data = json.loads(details_path.read_text())
            except (OSError, ValueError):
                continue
            v = data.get("vehicle", data)
            if v.get("delisted_at"):
                continue  # already flagged in a previous sync, don't overwrite the original timestamp
            v["delisted_at"] = now
            details_path.write_text(json.dumps(data, indent=2) + "\n")
            newly_flagged.append(entry["folder"])
        if newly_flagged:
            log(f"Delisted: {len(newly_flagged)} vehicle(s) no longer in the live listing, flagged:")
            for folder in newly_flagged:
                log(f"  - {folder}")
        else:
            log(f"Delisted: none ({len(delisted)} already flagged from a previous sync)" if delisted
                else "Delisted: none")

    if failures:
        log("Failures:")
        for r in failures:
            log(f"  - {r['url']}: {r['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
