#!/usr/bin/env python3
"""
Turn a scraped vehicle's cutouts into a hero image. Sensible defaults for
everything -- just point it at a vehicle folder:

    compose_cli.py ~/Documents/listings/2026-Ford-Bronco-Sport-BIG-Bend-RE76118

Only override what you actually want to change:

    compose_cli.py <vehicle-folder> --background american-flag --layout corners --no-glow

See what's available before picking:

    compose_cli.py --list-assets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imaging import assets
from imaging.compose import LAYOUTS, compose_hero
from imaging.select import pick_for_layout


def list_assets():
    print("Backgrounds:")
    for e in assets.list_backgrounds():
        print(f"  {e['name']}  (tags: {', '.join(e.get('tags', []))})")
    print("\nBorders:")
    for e in assets.list_borders():
        print(f"  {e['name']}  (tags: {', '.join(e.get('tags', []))})")


def resolve_asset_arg(category: str, value: str | None) -> Path:
    """CLI wrapper around assets.resolve_arg() that exits with a clear
    message instead of raising -- defaults to "American Flag" for
    backgrounds so this matches dtfb.py's own default when unset."""
    default_name = "American Flag" if category == "backgrounds" else None
    try:
        return assets.resolve_arg(category, value, default_name=default_name)
    except ValueError as e:
        sys.exit(str(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vehicle_folder", nargs="?",
                         help="A folder produced by dtfb.py (contains images/exterior/cutout/)")
    parser.add_argument("--background", help="Background name or tag (default: American Flag)")
    parser.add_argument("--border", help="Border name or tag (default: first available)")
    parser.add_argument("--layout", default="quad", choices=list(LAYOUTS),
                         help="single: just the hero. corners: hero + up to 2 corner accents. "
                              "quad: hero + front/back/side accents (default -- matches dtfb.py's "
                              "automatic pipeline).")
    parser.add_argument("--no-glow", action="store_true", help="Disable the glow behind each car cutout "
                                                                 "(on by default, matches dtfb.py)")
    parser.add_argument("--glow-color", default="white", choices=["white", "blue", "gold", "red"])
    parser.add_argument("--glow-radius", type=int, default=24, help="Glow blur radius in pixels (default: 24)")
    parser.add_argument("--glow-intensity", type=float, default=0.75, help="Glow opacity, 0-1 (default: 0.75)")
    parser.add_argument("--out", help="Output path (default: <vehicle_folder>/bundle/hero-<layout>.png)")
    parser.add_argument("--list-assets", action="store_true", help="List available backgrounds/borders and exit")
    args = parser.parse_args()

    if args.list_assets:
        list_assets()
        return

    if not args.vehicle_folder:
        parser.error("vehicle_folder is required (or pass --list-assets)")

    vehicle_folder = Path(args.vehicle_folder)
    cutout_dir = vehicle_folder / "images" / "exterior" / "cutout"
    if not cutout_dir.is_dir():
        sys.exit(f"No cutouts found at {cutout_dir} -- did you run dtfb.py on this vehicle yet?")

    background_path = resolve_asset_arg("backgrounds", args.background)
    border_path = resolve_asset_arg("borders", args.border)

    n_accents = {"single": 0, "corners": 2, "quad": 3}[args.layout]
    car_paths = pick_for_layout(cutout_dir, layout=args.layout, n_accents=n_accents)
    if not car_paths:
        sys.exit(f"No usable cutouts in {cutout_dir}")

    result = compose_hero(
        background_path=background_path,
        border_path=border_path,
        car_paths=car_paths,
        layout=args.layout,
        glow=not args.no_glow,
        glow_color=args.glow_color,
        glow_radius=args.glow_radius,
        glow_intensity=args.glow_intensity,
    )

    out_path = Path(args.out) if args.out else vehicle_folder / "bundle" / f"hero-{args.layout}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Saved: {out_path}")
    print(f"  background: {background_path.name}")
    print(f"  border: {border_path.name}")
    print(f"  cars: {', '.join(p.name for p in car_paths)}")


if __name__ == "__main__":
    main()
