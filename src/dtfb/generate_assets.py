#!/usr/bin/env python3
"""
Generate composition assets (backgrounds, later borders) via Recraft and
register them in assets/manifest.json.

Usage:
    generate_assets.py background "a sleek futuristic car showroom, empty, wide reflective floor" \\
        --name "Futuristic Showroom" --tags showroom sci-fi

Each run is exactly one paid Recraft API call ($0.035 at the default model).
Nothing here runs as a side effect of dtfb.py -- generation is always an
explicit, separate step.
"""
from __future__ import annotations

import argparse

from dtfb.imaging.generate import DEFAULT_MODEL, DEFAULT_SIZE, generate_background


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)

    bg = sub.add_parser("background", help="Generate and register a background image")
    bg.add_argument("prompt", help="Image generation prompt")
    bg.add_argument("--name", required=True, help="Display name (also used to derive the filename)")
    bg.add_argument("--tags", nargs="*", default=[], help="Tags for lookup, e.g. showroom sci-fi")
    bg.add_argument("--model", default=DEFAULT_MODEL)
    bg.add_argument("--size", default=DEFAULT_SIZE)

    args = parser.parse_args()

    if args.kind == "background":
        path = generate_background(args.prompt, args.name, args.tags, model=args.model, size=args.size)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
