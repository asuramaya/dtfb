"""
Per-vehicle orchestration: fetch the page, normalize it, download photos
and the window sticker, compose the hero/framed images, write the post and
the manifest record. dtfb.py's main() is just the CLI wrapper around
process_vehicle() looped over a URL list.
"""
from __future__ import annotations

import dataclasses
import random
import json
from pathlib import Path

import requests

import dtfb.manifest as fetch_manifest
from dtfb.facebook_post import build_facebook_post, check_pricing_consistency, explain_facebook_post
from dtfb.social_post import build_instagram_caption, build_threads_post
from dtfb.imaging.compose import compose_interiors, compose_vehicle, compose_wheel_shots
from dtfb.photos import download_photos
from dtfb.scrape import (USER_AGENT, condition_bucket, fetch_rendered_html, normalize_vehicle,
                     vehicle_folder_name, vin_from_url)
from dtfb.window_sticker import download_window_sticker


def log(msg: str) -> None:
    print(msg, flush=True)


def new_page(playwright, headed: bool):
    """Launch a fresh browser for one page load.

    Cloudflare's bot score appears to accumulate across navigations in the
    same browser session -- back-to-back VDP fetches in one long-lived
    context reliably start failing the JS challenge after the first one or
    two. A throwaway browser per vehicle costs ~1-2s but has proven far more
    reliable in testing than reusing one context across a whole batch.
    """
    browser = playwright.chromium.launch(
        headless=not headed,
        args=["--disable-blink-features=AutomationControlled"],
    )
    ctx = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
    page = ctx.new_page()
    return browser, page


def video_output_path(folder: Path, fmt: str) -> Path:
    """bundle/hero-video.mp4 stays the square one so nothing that already
    points at it breaks; the other aspects get suffixed siblings."""
    from dtfb.imaging.compose.hero_video import DEFAULT_VIDEO_FORMAT

    name = "hero-video.mp4" if fmt == DEFAULT_VIDEO_FORMAT else f"hero-video-{fmt}.mp4"
    return folder / "bundle" / name


def render_vehicle_video(folder: Path, hero_opts, fmt: str = "square") -> dict | None:
    """One animated hero per vehicle per aspect, from the cutouts just
    produced. Returns None (not an error) when the gallery can't fill the
    3-slot conveyor. Mirrors hero_video_cli.py, which stays the way to
    re-render one vehicle without re-scraping."""
    from dtfb.imaging.compose.hero_video import VIDEO_FORMATS

    spec = VIDEO_FORMATS[fmt]
    from dtfb.imaging.compose import render_hero_video
    from dtfb.imaging.palette import colors_from_details, vehicle_gradient_colors
    from dtfb.imaging.select import order_for_conveyor_start, pick_all_for_carousel

    cutout_dir = folder / "images" / "exterior" / "cutout"
    carousel = pick_all_for_carousel(cutout_dir, wheel_dir=folder / "images" / "exterior" / "wheels")
    if len(carousel) < 3:
        return None
    carousel = order_for_conveyor_start(carousel, cutout_dir)

    gradient_colors = None
    if hero_opts.video_background is None:
        ext, inr = colors_from_details(folder)
        sample = next(iter(sorted(cutout_dir.glob("*.png"))), None)
        start, end = vehicle_gradient_colors(ext, inr, sample)
        gradient_colors = (random.Random(folder.name).uniform(0, 360), start, end)

    return render_hero_video(
        background_video=hero_opts.video_background,
        border_path=hero_opts.border_path,
        carousel_paths=[p for p, _l in carousel],
        carousel_labels=[l for _p, l in carousel],
        audio_path=hero_opts.video_audio,
        bars_per_loop=hero_opts.video_bars_per_loop,
        gradient_colors=gradient_colors,
        out_path=video_output_path(folder, fmt),
        canvas_size=spec["canvas"],
        budget_mb=spec["budget_mb"],
        glow=hero_opts.glow,
        glow_color=hero_opts.glow_color,
        glow_radius=hero_opts.glow_radius,
        glow_intensity=hero_opts.glow_intensity,
        encoder=hero_opts.video_encoder,
    )


@dataclasses.dataclass
class HeroOptions:
    """Bundles compose_vehicle()'s asset/style knobs so process_vehicle()
    doesn't grow another handful of positional params -- see dtfb.py's
    --background/--border/--no-hero/--glow* flags for where these come from."""
    enabled: bool = True
    # background_path/border_path None means "generated gradient" and
    # "frameless" -- the defaults since the shared flag backdrop and the
    # contact-info frame both caused posting problems. dtfb.py's
    # --photo-background/--frame put them back.
    background_path: Path | None = None
    border_path: Path | None = None
    gradient: bool = True
    video: bool = True
    video_background: Path | None = None
    video_audio: Path | None = None
    video_bars_per_loop: int = 4
    video_encoder: str = "libx264"
    # One edit per aspect. Same choreography in all of them -- only the
    # blocking changes, see imaging/compose/hero_video.py::VIDEO_FORMATS.
    video_formats: tuple[str, ...] = ("square", "vertical", "horizontal")
    # Hero still shapes. framed/ stays square regardless -- see
    # imaging/compose/pipeline.py::HERO_STILL_FORMATS.
    hero_formats: tuple[str, ...] = ("square", "portrait")
    # Interior post-processing (exposure fix + confirmed feature callouts,
    # see imaging/interior.py). The classifier is threaded through here
    # rather than constructed per-vehicle so it shares the one CLIP
    # backbone the rest of the run already paid to load.
    interiors: bool = True
    # Text on a listing photo is a merchandising call, not an image-
    # processing one, so the default output is the corrected photograph
    # and nothing else. dtfb.py's --interior-captions turns it back on.
    interior_captions: bool = False
    interior_classifier: object | None = None
    # Local-vision front-seat-config extraction (imaging/seat_vision.py) --
    # off by default. It needs `ollama serve` running with gemma4:e2b
    # pulled, which isn't guaranteed on every machine this runs on; the
    # extraction call itself is already advisory (never fails a scrape,
    # returns None quietly if Ollama isn't reachable), but the DEFAULT
    # being off is a deliberate choice on top of that, same reasoning as
    # interior_captions: an extra ~10s/vehicle and a new local-model
    # dependency shouldn't turn on silently. dtfb.py's --vision-seat-check.
    vision_seat_check: bool = False
    glow: bool = True
    glow_color: str = "white"
    glow_radius: int = 24
    glow_intensity: float = 0.75


def process_vehicle(playwright, session: requests.Session, url: str, out_root: Path,
                     sticker_dpi: int, headed: bool, junk_filter, classifier,
                     upscale_cutouts: bool, upscale_model: str,
                     angle_classifier=None, hero_opts: "HeroOptions | None" = None,
                     wheel_classifier=None, spare_classifier=None,
                     interior_tiebreak_classifier=None,
                     strict_cutouts: bool = True) -> Path:
    log(f"==> {url}")
    browser, page = new_page(playwright, headed)
    try:
        html = fetch_rendered_html(page, url)
    finally:
        browser.close()
    v = normalize_vehicle(url, html)

    # A real vehicle detail page always has at least a VIN, even for a
    # freshly-arrived unit with no price or photos yet published (confirmed
    # case: "Incoming" stock with a VIN but no price/photos still isn't
    # this). VIN + price + photos ALL missing at once means this almost
    # certainly wasn't a single-vehicle page at all -- e.g. a category/
    # listing URL that didn't redirect to one specific VIN. Found in
    # practice: feeding an inventory listing URL in here "succeeded" with
    # an empty folder and a wall of warnings, which reads as a working
    # scrape until you look inside. Fail loudly instead.
    if not v.vin and not v.display_price and not v.photo_urls:
        raise RuntimeError(
            "No VIN, price, or photos found -- this doesn't look like a single "
            "vehicle's page. Check that the URL is a specific vehicle's /vehicle/... "
            "page, not a category/search/listing page."
        )

    # A dead/mistyped VIN or a stale bookmark can 404 or redirect to a
    # generic page that still embeds a DIFFERENT real vehicle's schema.org
    # data -- confirmed real case: tomballford.com's not-found page carries
    # a "similar vehicles" widget, and extract_ldjson_car() has no way to
    # tell that apart from the actual target. The above VIN/price/photos
    # check doesn't catch this because the widget vehicle DOES have a VIN
    # -- just the wrong one. The URL itself is the one thing we can trust:
    # if it names a VIN and the page we scraped has a different one, this
    # was never the vehicle asked for, and saving it under the wrong name
    # is worse than failing loudly.
    expected_vin = vin_from_url(url)
    if expected_vin and v.vin and v.vin.upper() != expected_vin.upper():
        raise RuntimeError(
            f"Scraped VIN {v.vin} does not match the VIN in the requested URL "
            f"({expected_vin}). This page likely 404'd or redirected to an "
            "unrelated vehicle -- check the URL is current, not a stale bookmark "
            "or a mistyped VIN."
        )

    bucket = condition_bucket(v, url)
    folder = out_root / bucket / vehicle_folder_name(v)
    folder.mkdir(parents=True, exist_ok=True)
    log(f"    folder: {folder}")

    photos = download_photos(session, v, folder / "images", junk_filter, classifier, upscale_cutouts, upscale_model,
                              angle_classifier, wheel_classifier, spare_classifier, interior_tiebreak_classifier,
                              strict_cutouts)
    originals = [p for p in photos if "/cutout/" not in p and "/wheels/" not in p]
    cutouts = len([p for p in photos if "/cutout/" in p])
    wheel_cutouts = len([p for p in photos if "/wheels/" in p])
    cutout_note = f" (+{cutouts} cutout{'s' if cutouts != 1 else ''})" if cutouts else ""
    wheel_note = f" (+{wheel_cutouts} wheel shot{'s' if wheel_cutouts != 1 else ''})" if wheel_cutouts else ""
    log(f"    photos: {len(originals)}/{len(v.photo_urls)} saved{cutout_note}{wheel_note}")

    # Does this vehicle's gallery duplicate one already scraped? See
    # imaging/dedupe.py -- advisory only, and wrapped because a warning is
    # never worth failing a scrape over.
    try:
        from dtfb.imaging.dedupe import find_shared_gallery, gallery_hashes, record_gallery_hashes

        exterior_dir = folder / "images" / "exterior"
        hashes = gallery_hashes(exterior_dir)
        if hashes:
            shared = find_shared_gallery(out_root, f"{bucket}/{folder.name}", hashes, exterior_dir)
            for other, count in sorted(shared.items(), key=lambda kv: -kv[1]):
                v.warnings.append(
                    f"{count} of {len(hashes)} exterior photo(s) are the same image as "
                    f"{other} -- these look like manufacturer stock renders rather than "
                    "photos of this vehicle, which Facebook Marketplace prohibits."
                )
            record_gallery_hashes(out_root, f"{bucket}/{folder.name}", hashes)
    except Exception as e:
        log(f"    ! stock-render check failed: {e}")

    sticker_pngs = download_window_sticker(session, v, folder, sticker_dpi)
    if sticker_pngs:
        log(f"    window sticker: {len(sticker_pngs)} page(s) rendered")
    else:
        log("    window sticker: none")

    v.warnings.extend(check_pricing_consistency(v))

    if hero_opts and hero_opts.enabled:
        # A composition hiccup (bad asset, no cutouts, whatever) shouldn't
        # sink an otherwise-successful scrape -- it's a bonus deliverable,
        # not core vehicle data.
        try:
            compose_result = compose_vehicle(
                cutout_dir=folder / "images" / "exterior" / "cutout",
                out_dir=folder / "bundle",
                background_path=hero_opts.background_path,
                border_path=hero_opts.border_path,
                glow=hero_opts.glow,
                glow_color=hero_opts.glow_color,
                glow_radius=hero_opts.glow_radius,
                glow_intensity=hero_opts.glow_intensity,
                gradient=hero_opts.gradient,
                exterior_color=v.exterior_color_factory,
                interior_color=v.interior_color,
                hero_formats=hero_opts.hero_formats,
            )
            if compose_result["hero"]:
                n_framed = len(compose_result["framed"])
                names = ", ".join(p.name for p in compose_result["heroes"].values())
                log(f"    hero image: {names} + {n_framed} bundle/framed/ image{'s' if n_framed != 1 else ''}")
            else:
                log("    hero image: skipped (no exterior cutouts)")
        except Exception as e:
            log(f"    ! hero image composition failed: {e}")
            v.warnings.append(f"Hero image composition failed: {e}")

        try:
            wheel_shots = compose_wheel_shots(
                wheel_cutout_dir=folder / "images" / "exterior" / "wheels",
                out_dir=folder / "bundle",
                background_path=hero_opts.background_path,
                border_path=hero_opts.border_path,
                glow=hero_opts.glow,
                glow_color=hero_opts.glow_color,
                glow_radius=hero_opts.glow_radius,
                glow_intensity=hero_opts.glow_intensity,
                gradient=hero_opts.gradient,
                exterior_color=v.exterior_color_factory,
                interior_color=v.interior_color,
            )
            if wheel_shots:
                log(f"    wheel money shot{'s' if len(wheel_shots) != 1 else ''}: "
                    f"{len(wheel_shots)} in bundle/framed/")
        except Exception as e:
            log(f"    ! wheel shot composition failed: {e}")
            v.warnings.append(f"Wheel shot composition failed: {e}")

        if hero_opts.interiors:
            # Same bonus-deliverable rule. Runs off images/interior/, so it
            # has to come after download_photos() has finished sorting --
            # including any cutout demoted into that folder by
            # imaging/gallery.py, which is exactly the kind of photo (a
            # cabin shot, a van's cargo bay) this is for.
            try:
                processed = compose_interiors(
                    interior_dir=folder / "images" / "interior",
                    out_dir=folder / "bundle",
                    vehicle=dataclasses.asdict(v),
                    classifier=hero_opts.interior_classifier,
                    captions=hero_opts.interior_captions,
                )
                if processed:
                    captioned = sum(1 for p in processed if p["callout"])
                    note = f" ({captioned} captioned)" if hero_opts.interior_captions else ""
                    log(f"    interior: {len(processed)} corrected into bundle/interior/{note}")
            except Exception as e:
                log(f"    ! interior processing failed: {e}")
                v.warnings.append(f"Interior processing failed: {e}")

        if hero_opts.vision_seat_check:
            # Same bonus-deliverable rule, and same "never sink the scrape"
            # guarantee -- imaging/seat_vision.py is already internally
            # advisory (returns None rather than raising if Ollama isn't
            # reachable or a response doesn't parse), this is just the
            # pipeline-level belt-and-suspenders on top of that.
            try:
                from dtfb.imaging.seat_vision import extract_seat_config

                result = extract_seat_config(folder, classifier=hero_opts.interior_classifier)
                if result is not None:
                    (folder / "vision-equipment.json").write_text(json.dumps(result, indent=2) + "\n")
                    log(f"    vision seat check: {result['config']} ({result['confidence']} confidence, "
                        f"from {result['source_photo']})")
            except Exception as e:
                log(f"    ! vision seat check failed: {e}")

        if hero_opts.video:
            # Same bonus-deliverable rule as the stills: a render failure
            # (ffmpeg missing, too few cutouts) must not sink the scrape.
            try:
                rendered = 0
                for fmt in hero_opts.video_formats:
                    report = render_vehicle_video(folder, hero_opts, fmt)
                    if report is None:
                        break
                    rendered += 1
                    log(f"    hero video ({fmt}): {Path(report['out_path']).name} "
                        f"({report['duration_s']}s, {report['file_size_mb']} MB, "
                        f"{report['n_shots']} shots)")
                if not rendered:
                    log("    hero video: skipped (needs 3+ exterior cutouts)")
            except Exception as e:
                log(f"    ! hero video failed: {e}")
                v.warnings.append(f"Hero video failed: {e}")

    details = dataclasses.asdict(v)
    details["_facebook_post_notes"] = explain_facebook_post(v)
    (folder / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    # facebook.txt is the actual text to post, right alongside the images
    # meant to go with it -- lives in bundle/, not the folder root.
    bundle_dir = folder / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "facebook.txt").write_text(build_facebook_post(v), encoding="utf-8")
    # Same facts, re-ordered and re-budgeted per surface -- see
    # social_post.py for why the long post can't just be truncated.
    (bundle_dir / "threads.txt").write_text(build_threads_post(v), encoding="utf-8")
    (bundle_dir / "instagram.txt").write_text(build_instagram_caption(v), encoding="utf-8")

    fetch_manifest.record_fetch(out_root, url, f"{bucket}/{folder.name}", vin=v.vin, stock_number=v.stock_number)

    if v.warnings:
        for w in v.warnings:
            log(f"    ! {w}")

    return folder
