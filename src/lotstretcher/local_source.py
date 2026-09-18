"""
The non-scrape entry point: turn a local folder of photos into a Vehicle
record, so a listing with no VDP at all (an auction photo dump, a trade-in
walked around with a phone, a dealer's own DAM export) can still go through
the same CLIP/cutout/compose/copy pipeline as a scraped one.

A "local vehicle" is a directory:

    my-2023-f150/
        01.jpg
        02.jpg
        ...
        vehicle.json      # optional -- see LOCAL_VEHICLE_META_FILENAME

`vehicle.json` is optional because the CV pipeline itself needs nothing
from it -- photo classification, cutouts, and hero/video composition run
off the images alone. Without it you still get a full bundle/, just with
thin post copy (no year/make/model/price to write sentences about) and a
folder name derived from the directory name instead of vehicle specs. Any
subset of Vehicle's fields (scrape.py) can be set; unknown keys are
rejected loudly rather than silently ignored, since a typo'd key
(`"marke"` instead of `"make"`) silently producing an empty field is a
much worse failure mode than an error naming the bad key.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from lotstretcher.scrape import Vehicle

LOCAL_VEHICLE_META_FILENAME = "vehicle.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}

# url/photo_urls/warnings are derived by this module, not user-settable --
# excluded here so a vehicle.json setting "url" fails with the field-list
# error below instead of a raw TypeError from Vehicle(url=..., **meta)
# colliding on the same keyword.
_DERIVED_FIELDS = {"url", "photo_urls", "warnings"}
_VEHICLE_FIELDS = {f.name for f in dataclasses.fields(Vehicle)} - _DERIVED_FIELDS


def is_local_source(path: str) -> bool:
    """True if `path` names an existing directory -- the signal cli.py
    uses to route an argument to this module instead of treating it as a
    VDP/listing URL. Deliberately narrow: a bare relative/absolute path
    that doesn't exist on disk is far more likely a typo'd URL than a
    local source, and should fail as one rather than silently no-op here."""
    return Path(path).is_dir()


def load_local_vehicle(folder: Path) -> Vehicle:
    """Build a Vehicle from a local photo folder. photo_urls ends up full
    of real filesystem paths (as strings) rather than HTTP URLs --
    photos.py's fetch step already knows to read those straight off disk
    instead of making a request, so nothing downstream needs to special-
    case a local source at all."""
    folder = Path(folder)
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(
            f"No photos found in {folder} (looked for {', '.join(sorted(IMAGE_EXTENSIONS))}). "
            "Nothing to process."
        )

    meta = {}
    meta_path = folder / LOCAL_VEHICLE_META_FILENAME
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise RuntimeError(f"{meta_path}: invalid JSON ({e})") from e
        if not isinstance(meta, dict):
            raise RuntimeError(f"{meta_path}: expected a JSON object, got {type(meta).__name__}")
        unknown = set(meta) - _VEHICLE_FIELDS
        if unknown:
            raise RuntimeError(
                f"{meta_path}: unknown field(s) {sorted(unknown)} -- valid fields are "
                f"{sorted(_VEHICLE_FIELDS)}"
            )

    v = Vehicle(url=f"local://{folder.resolve().name}", **meta)
    v.photo_urls = [str(p.resolve()) for p in images]
    # vehicle_folder_name() (scrape.py) names the output folder off
    # stock_number/vin, falling back to the literal string "unknown" when
    # both are absent -- fine for one vehicle, a silent collision for two
    # different unnamed local folders in the same --out. Falling back to
    # the source folder's own name keeps every local vehicle's output
    # folder distinct without requiring a vehicle.json at all.
    if not v.stock_number and not v.vin:
        v.stock_number = folder.resolve().name
    return v


def local_vehicle_key(folder: Path) -> str:
    """The manifest/dedup key for a local source, standing in for the URL
    a scraped vehicle would use -- see manifest.py's already_fetched()/
    record_fetch(), which key purely on this string. Uses the resolved
    folder path (not just its name) so two same-named folders in different
    parent directories are never conflated as the same "already fetched"
    vehicle."""
    return f"local://{Path(folder).resolve()}"
