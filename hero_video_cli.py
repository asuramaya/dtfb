#!/usr/bin/env python3
"""
Turn a scraped vehicle's cutouts into an animated hero video: a looping
flag-video background plus a 3-slot conveyor (left accent / hero / right
accent) that cycles through the vehicle's WHOLE shot library, timed to N
loops of an audio track.

    hero_video_cli.py ~/Documents/listings/2023-Ford-F-150-Raptor-PFA15687

The backdrop clip and audio loop come from assets/manifest.json, same as
backgrounds and borders -- override with --background-video/--audio by
manifest name, tag, or a one-off path.

See compose_cli.py for the equivalent still-image tool this mirrors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imaging import assets
from imaging.compose import render_hero_video
from imaging.compose.hero_video import (BARS_PER_LOOP, DEFAULT_BPM, DEFAULT_VIDEO_FORMAT,
                                          VIDEO_FORMATS)
from imaging.select import order_for_conveyor_start, pick_all_for_carousel


def resolve_asset_arg(category: str, value: str | None, default_name: str | None = None) -> Path:
    """Manifest lookup, with a plain filesystem path accepted as an escape
    hatch so a one-off clip can be tried without registering it first."""
    if value and Path(value).exists():
        return Path(value)
    try:
        return assets.resolve_arg(category, value, default_name=default_name)
    except ValueError as e:
        sys.exit(str(e))


def resolve_audio(value: str | None) -> tuple[Path, int]:
    """(path, bars_per_loop). An unregistered path can't carry its bar
    count, so it falls back to the 4-bar assumption the module documents
    -- registering it in the manifest is how you say otherwise."""
    if value and Path(value).exists():
        return Path(value), BARS_PER_LOOP
    try:
        entry = assets.entry_for("audio", value, default_name="Its Mine")
    except ValueError as e:
        sys.exit(str(e))
    return assets.resolve(entry), int(entry.get("bars", BARS_PER_LOOP))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vehicle_folder", help="A folder produced by dtfb.py (contains images/exterior/cutout/)")
    parser.add_argument("--background-video", help="Video name/tag from assets/manifest.json, or a path "
                                                    "(default: American Flag Waving)")
    parser.add_argument("--audio", help="Audio name/tag from assets/manifest.json, or a path (default: Its Mine). "
                                         "Must be a seamless loop of a whole number of bars -- the manifest entry's "
                                         "`bars` is what the beat math divides by.")
    parser.add_argument("--border", help="Border name or tag (default: first available)")
    parser.add_argument("--loops", type=int, default=None,
                         help="Force the length to N audio-loop lengths. Default is one full pass of the "
                              "vehicle's shot library, however long that takes.")
    parser.add_argument("--format", default=DEFAULT_VIDEO_FORMAT,
                         choices=[*VIDEO_FORMATS, "all"],
                         help="Frame shape. " + "; ".join(
                             f"{k} {v['canvas'][0]}x{v['canvas'][1]} ({v['note']})"
                             for k, v in VIDEO_FORMATS.items()) +
                              ". 'all' renders each in turn. The edit is identical in every "
                              "format -- only the blocking changes.")
    parser.add_argument("--budget-mb", type=float, default=None,
                         help="Target max file size in MB. Defaults to the format's own budget "
                              "(square 50, the others 66 -- the 50MB cap is Marketplace's and "
                              "does not apply to Reels/Shorts/TikTok).")
    parser.add_argument("--no-glow", action="store_true")
    parser.add_argument("--glow-color", default="white", choices=["white", "blue", "gold", "red"])
    parser.add_argument("--glow-radius", type=int, default=24)
    parser.add_argument("--glow-intensity", type=float, default=0.75)
    parser.add_argument("--flag-background", action="store_true",
                         help="Use the backdrop video clip instead of the default rotating vehicle-color gradient.")
    parser.add_argument("--frame", action="store_true",
                         help="Composite the dealer frame back on (frameless is the default).")
    parser.add_argument("--music", action="store_true",
                         help="Score the video. Silent is the default; timing comes from --bpm either way, so "
                              "the cut/pump cadence is identical.")
    parser.add_argument("--bpm", type=float, default=None,
                         help=f"Tempo driving the bar/beat grid when silent (default: {DEFAULT_BPM:g}, which "
                              "reproduces the scored version's exact cadence).")
    parser.add_argument("--nvenc", action="store_true",
                         help="Encode on the GPU (h264_nvenc). Only the encode moves; the frame compositing is CPU either way, so expect a modest win.")
    parser.add_argument("--out", help="Output path (default: <vehicle_folder>/bundle/hero-video.mp4)")
    args = parser.parse_args()

    vehicle_folder = Path(args.vehicle_folder)
    cutout_dir = vehicle_folder / "images" / "exterior" / "cutout"
    wheel_dir = vehicle_folder / "images" / "exterior" / "wheels"
    if not cutout_dir.is_dir():
        sys.exit(f"No cutouts found at {cutout_dir} -- did you run dtfb.py on this vehicle yet?")

    border_path = resolve_asset_arg("borders", args.border) if args.frame else None
    background_video = resolve_asset_arg(
        "videos", args.background_video, default_name="American Flag Waving") if args.flag_background else None

    gradient_colors = None
    if background_video is None:
        # (base_angle, start, end) -- the renderer spins base_angle over
        # the run, so only the seeded starting angle is decided here.
        import random as _random
        from imaging.palette import colors_from_details, vehicle_gradient_colors
        ext, inr = colors_from_details(vehicle_folder)
        sample = next(iter(sorted(cutout_dir.glob("*.png"))), None)
        start, end = vehicle_gradient_colors(ext, inr, sample)
        gradient_colors = (_random.Random(vehicle_folder.name).uniform(0, 360), start, end)

    if args.music or args.audio:
        audio_path, bars_per_loop = resolve_audio(args.audio)
    else:
        audio_path, bars_per_loop = None, BARS_PER_LOOP

    carousel = pick_all_for_carousel(cutout_dir, wheel_dir=wheel_dir)
    if len(carousel) < 3:
        sys.exit(f"Need at least 3 usable cutouts for the conveyor, found {len(carousel)} in {cutout_dir}")
    carousel = order_for_conveyor_start(carousel, cutout_dir)
    carousel_paths = [p for p, _label in carousel]
    carousel_labels = [label for _p, label in carousel]

    formats = list(VIDEO_FORMATS) if args.format == "all" else [args.format]
    for fmt in formats:
        render_one(fmt, args, vehicle_folder, border_path, background_video, gradient_colors,
                    audio_path, bars_per_loop, carousel_paths, carousel_labels)


def render_one(fmt, args, vehicle_folder, border_path, background_video, gradient_colors,
                audio_path, bars_per_loop, carousel_paths, carousel_labels):
    from vehicle_pipeline import video_output_path

    spec = VIDEO_FORMATS[fmt]
    out_path = Path(args.out) if args.out else video_output_path(vehicle_folder, fmt)

    report = render_hero_video(
        background_video=background_video,
        border_path=border_path,
        carousel_paths=carousel_paths,
        carousel_labels=carousel_labels,
        audio_path=audio_path,
        bars_per_loop=bars_per_loop,
        gradient_colors=gradient_colors,
        bpm=args.bpm or DEFAULT_BPM,
        out_path=out_path,
        canvas_size=spec["canvas"],
        loops=args.loops,
        budget_mb=args.budget_mb if args.budget_mb is not None else spec["budget_mb"],
        glow=not args.no_glow,
        glow_color=args.glow_color,
        glow_radius=args.glow_radius,
        glow_intensity=args.glow_intensity,
        encoder="h264_nvenc" if args.nvenc else "libx264",
    )

    print(f"Saved [{fmt} {spec['canvas'][0]}x{spec['canvas'][1]}]: "
          f"{report['out_path']} ({report['file_size_mb']} MB)")
    print(f"  duration: {report['duration_s']}s @ {report['fps']}fps ({report['total_frames']} frames)")
    print(f"  conveyor: {report['n_shots']} shots over {report['carousel_period_s']}s/pass, "
          f"bar={report['bar_dwell_s']}s (audio loop {report['audio_loop_s']}s / {report['bars_per_loop']} bars), "
          f"beat={report['beat_s']}s, transition={report['transition_s']}s")
    print(f"  backdrop: {report['backdrop']}, frame: {'yes' if report['framed'] else 'none'}, "
          f"clock: {report['clock']}")
    print(f"  shot order: {', '.join(report['shot_order'])}")
    for name, how in report["pan_shots"].items():
        print(f"  pan: {name} -- {how}")
    if report["carousel_incomplete"]:
        print(f"  NOTE: conveyor period ({report['carousel_period_s']}s) exceeds the video length "
              f"({report['duration_s']}s) -- not every shot gets shown; raise --loops.")
    audio_note = f" + {128}kbps audio" if report["clock"] == "audio" else " (silent)"
    print(f"  bitrate: {report['bitrate_kbps']}kbps video{audio_note}")


if __name__ == "__main__":
    main()
