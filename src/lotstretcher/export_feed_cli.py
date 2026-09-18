#!/usr/bin/env python3
"""
Build a vehicle inventory feed (CSV/TSV) from vehicles already on disk,
in the shape listing platforms expect -- see export_feed.py's docstring
for the field-mapping caveat and why this exists (no open DMS/inventory
feed standard, but exporting needs nobody's permission).

    export-feed ~/Documents/listings --out feed.csv
    export-feed ~/Documents/listings/used --out used-feed.csv --condition used
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lotstretcher.export_feed import build_rows, write_csv
from lotstretcher.recompose_cli import find_vehicle_folders, vehicle_record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="A listings root, or a single vehicle folder")
    parser.add_argument("--out", required=True, help="Output feed file path")
    parser.add_argument("--tsv", action="store_true", help="Tab-delimited instead of comma-delimited")
    parser.add_argument("--condition", choices=["new", "used", "all"], default="all",
                         help="Only include vehicles of this condition (default: all)")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    folders = find_vehicle_folders(root)
    if not folders:
        sys.exit(f"No vehicle folders (with images/exterior/cutout/) found under {root}")

    if args.condition != "all":
        from lotstretcher.export_feed import CONDITION_TO_STATE
        folders = [f for f in folders
                   if CONDITION_TO_STATE.get((vehicle_record(f).get("condition") or "").lower()) == args.condition]

    rows = build_rows(folders)
    skipped = len(folders) - len(rows)
    write_csv(rows, Path(args.out), delimiter="\t" if args.tsv else ",")

    print(f"Wrote {len(rows)} vehicle(s) to {args.out}"
          + (f" ({skipped} skipped -- delisted, or missing price/photos)" if skipped else "") + ".")
    print("Validate against your destination platform's own feed validator before going live "
          "(see export_feed.py's field-mapping caveat).")


if __name__ == "__main__":
    main()
