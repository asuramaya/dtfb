#!/usr/bin/env python3
"""
Backfill front-seat-config vision extraction (imaging/seat_vision.py) onto
vehicles already scraped, without re-scraping anything.

    seat_vision_cli.py ~/Documents/listings                 # every vehicle
    seat_vision_cli.py ~/Documents/listings/used/2021-Vol    # just one

Runs on EVERY vehicle with an interior photo, not just sticker-silent
ones -- the same extraction that fills a sticker's silence also gives
equipment_search.py its only coverage for used vehicles that never had a
sticker at all. Requires `ollama serve` running locally with gemma4:e2b
pulled; if Ollama isn't reachable this says so once and exits rather than
retrying per vehicle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dtfb.imaging.interior import InteriorSubjectClassifier
from dtfb.imaging.seat_vision import extract_seat_config, ollama_available


def find_vehicle_folders(root: Path) -> list[Path]:
    if (root / "details.json").is_file():
        return [root]
    return sorted(p.parent for p in root.glob("*/**/details.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="A listings root, or a single vehicle folder")
    parser.add_argument("--force", action="store_true",
                         help="Re-run vehicles that already have a vision-equipment.json")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    if not ollama_available():
        sys.exit("Ollama isn't reachable at http://127.0.0.1:11434 -- start it with `ollama serve` "
                  "(and make sure gemma4:e2b is pulled) before running this.")

    folders = find_vehicle_folders(root)
    if not folders:
        sys.exit(f"No vehicle folders (with details.json) found under {root}")

    classifier = InteriorSubjectClassifier()

    written = skipped = no_photos = failed = 0
    for folder in folders:
        out_path = folder / "vision-equipment.json"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        if not (folder / "images" / "interior").is_dir():
            no_photos += 1
            continue

        result = extract_seat_config(folder, classifier=classifier)
        if result is None:
            failed += 1
            print(f"  ! {folder.name}: no result (no readable photo, or the model's answer didn't parse)")
            continue

        out_path.write_text(json.dumps(result, indent=2) + "\n")
        written += 1
        print(f"  {folder.name}: {result['config']} ({result['confidence']}, from {result['source_photo']})")

    print(f"\n{written} written, {skipped} already done, {no_photos} with no interior photos, "
          f"{failed} failed")


if __name__ == "__main__":
    main()
