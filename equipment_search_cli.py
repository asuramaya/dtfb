#!/usr/bin/env python3
"""
Natural-language search over already-scraped inventory's window-sticker
equipment data. See equipment_search.py's module docstring for how the
query gets compiled and what it can/can't see.

    equipment_search_cli.py "a center console and no folding bench"
    equipment_search_cli.py "leather seats and a moonroof" --out ~/Documents/listings
"""
from __future__ import annotations

import argparse
from pathlib import Path

from equipment_search import compile_query, load_searchable_fleet, search


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Natural-language equipment query, in quotes")
    parser.add_argument("--out", default=str(Path.home() / "Documents" / "listings"),
                         help="Listings root to search (default: ~/Documents/listings)")
    parser.add_argument("--explain", action="store_true",
                         help="Show how the query was compiled before the results")
    args = parser.parse_args()

    root = Path(args.out)
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    fleet = load_searchable_fleet(root)
    filt = compile_query(args.query)

    print(f"searching {len(fleet)} vehicle(s) with a window sticker and/or a vision seat-config check")
    if args.explain:
        parts = []
        if filt.model_terms:
            parts.append(f"model/trim: {', '.join(filt.model_terms)}")
        if filt.include:
            parts.append("has: " + " AND ".join(f"({' or '.join(c)})" for c in filt.include))
        if filt.exclude:
            parts.append("without: " + " AND ".join(f"({' or '.join(c)})" for c in filt.exclude))
        print("interpreted as:", "; ".join(parts) if parts else "(no constraints recognized)")
    print()

    results = search(root, filt)
    if not results:
        print("No matches.")
        return

    for r in results:
        price = f"${r['price']}" if r["price"] else "price n/a"
        print(f"{r['title']} ({r['trim'] or 'n/a'}) -- {price}")
        print(f"  {r['folder']}")
        if r["url"]:
            print(f"  {r['url']}")
        print()


if __name__ == "__main__":
    main()
