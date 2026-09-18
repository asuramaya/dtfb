#!/usr/bin/env python3
"""
Rebuild the post text for vehicles already on disk, without re-scraping.

    posts_cli.py ~/Documents/listings                 # every vehicle
    posts_cli.py ~/Documents/listings/used/2021-Vol   # just one

Post copy is derived entirely from details.json, so a wording change, a new
surface, or a fixed hashtag rule shouldn't cost a Cloudflare-bypass browser
session and a full CV pipeline per vehicle.

Deliberately NOT folded into recompose_cli.py: that tool finds vehicles by
looking for images/exterior/cutout/, because composition is meaningless
without cutouts. Post text only needs details.json, and there are real
folders with one and not the other -- a vehicle whose photos were all
filtered, or one scraped before the imaging pipeline ran. Keying off the
wrong marker would silently skip exactly those.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lotstretcher.facebook_post import build_facebook_post
from lotstretcher.scrape import Vehicle
from lotstretcher.social_post import build_instagram_caption, build_threads_post

WRITERS = {
    "facebook.txt": build_facebook_post,
    "threads.txt": build_threads_post,
    "instagram.txt": build_instagram_caption,
}


def find_vehicle_folders(root: Path) -> list[Path]:
    if (root / "details.json").is_file():
        return [root]
    return sorted(p.parent for p in root.glob("*/**/details.json"))


def load_vehicle(folder: Path) -> Vehicle | None:
    """The scraped record back into a Vehicle. Unknown keys are dropped
    rather than raising: details.json also carries derived audit fields
    (_facebook_post_notes) and may predate a field being added."""
    try:
        raw = json.loads((folder / "details.json").read_text())
    except (OSError, ValueError):
        return None
    raw = raw.get("vehicle", raw)
    fields = set(Vehicle.__dataclass_fields__)
    return Vehicle(**{k: val for k, val in raw.items() if k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="A listings root, or a single vehicle folder")
    parser.add_argument("--only", action="append", choices=list(WRITERS),
                         help="Rebuild just this file; repeatable. Default: all three.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the first vehicle's output and exit, without writing")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    folders = find_vehicle_folders(root)
    if not folders:
        sys.exit(f"No vehicle folders (with details.json) found under {root}")

    wanted = args.only or list(WRITERS)
    print(f"{len(folders)} vehicle(s), rebuilding: {', '.join(wanted)}\n")

    written = skipped = 0
    for folder in folders:
        v = load_vehicle(folder)
        if v is None:
            print(f"  ! {folder.name}: unreadable details.json, skipped")
            skipped += 1
            continue
        if args.dry_run:
            for name in wanted:
                print(f"===== {folder.name} / {name}\n")
                print(WRITERS[name](v))
            return
        bundle = folder / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            (bundle / name).write_text(WRITERS[name](v), encoding="utf-8")
        written += 1

    print(f"Rebuilt {len(wanted)} file(s) for {written} vehicle(s)"
          + (f", skipped {skipped}" if skipped else "") + ".")


if __name__ == "__main__":
    main()
