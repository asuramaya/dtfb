"""
Perceptual-hash filtering for known junk/marketing images.

Dealer photo galleries sometimes have a fixed promotional graphic (a
"check our reviews!" card, etc.) mixed in among the actual vehicle photos --
the same exact image, reused across every vehicle in inventory. Since it's
a specific known image rather than a general category, perceptual hashing
(pHash) catches it far more cheaply and reliably than OCR or a trained
classifier would, with no risk of false-positives from e.g. text visible on
an infotainment screen in a real interior shot.

Add more known-junk images to templates/ as they turn up; no code changes
needed -- BUT read the next paragraph before adding one that looks like a
studio photo, because the threshold has far less room than it appears.

MEASURED HEADROOM, and the trap. Against the one template currently in
templates/ (a full-bleed "reviews" card) the filter is comfortable: over
1,074 real gallery photos the CLOSEST sits 20 bits away, against a
threshold of 8, with a median of 32 -- i.e. uncorrelated. Nothing real is
anywhere near being dropped.

That safety comes from the template being a flat graphic, not from the
threshold being conservative. On this dealer's actual vehicle photography
-- studio renders of one vehicle on a uniform grey floor, framed almost
identically shot to shot -- pHash is dominated by the background and
framing rather than by the car. Measured over 420 exterior photos, 466
pairs belonging to DIFFERENT vehicles fall within the 8-bit threshold of
each other, and spot-checking those at pixel level shows most are not
duplicates at all (a Tesla Model Y and a Toyota Camry match at 8 bits with
a mean pixel difference of 14/255).

So a junk template that resembles a studio shot would silently delete real
photos across most of the inventory, and the deletion happens before any
classifier sees them. Keep templates to flat, full-bleed graphics; if one
ever needs to be a photo, verify it against the fleet first rather than
trusting the 8-bit threshold.

A genuine cross-vehicle duplicate does exist and is worth knowing about
for a different reason: the two Transit vans share manufacturer renders
outright (0 bits, 0.00 mean pixel difference on some frames), because the
dealer merchandises them with factory imagery rather than photographs of
the actual vans. That is not junk and must not be filtered -- it is the
only photography those listings have.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import imagehash
from PIL import Image

DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"
DEFAULT_THRESHOLD = 8  # max Hamming distance (out of 64 bits) to count as a match


class JunkFilter:
    def __init__(self, templates_dir: Path = DEFAULT_TEMPLATES_DIR, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.templates: list[tuple[str, imagehash.ImageHash]] = []
        if templates_dir.is_dir():
            for path in sorted(templates_dir.glob("*")):
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                try:
                    self.templates.append((path.name, _hash_image(Image.open(path))))
                except Exception:
                    continue

    def is_junk(self, content: bytes) -> tuple[bool, str | None]:
        """Returns (is_junk, matched_template_name)."""
        if not self.templates:
            return False, None
        try:
            h = _hash_image(Image.open(io.BytesIO(content)))
        except Exception:
            return False, None
        for name, template_hash in self.templates:
            if h - template_hash <= self.threshold:
                return True, name
        return False, None


def _hash_image(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img.convert("RGB"))


# --- Stock-render detection -------------------------------------------
#
# Separate concern from the junk templates above: this doesn't drop
# anything, it just notices when a vehicle's gallery is the SAME imagery
# another vehicle already had. Real case: the two Transit vans are
# merchandised with manufacturer renders rather than photographs of the
# actual vans, sharing frames at 0 bits and 0.00 mean pixel difference.
#
# Worth surfacing because Marketplace prohibits stock photos outright --
# "images must be actual photos of the item you're selling" -- and
# stock/stolen-photo detection is a documented removal trigger. It is the
# dealer's imagery choice, not something this pipeline introduces, so the
# right response is a warning on the listing, never a deletion: for those
# two vans it is the only photography they have.
#
# EXACT hash match, then a pixel confirmation. Nothing looser is safe on
# this content -- 466 pairs from different vehicles fall within the 8-bit
# junk threshold of each other, and a Model Y matches a Camry there. See
# the module docstring for that measurement.
STOCK_RENDER_MAX_BITS = 0
STOCK_RENDER_MAX_PIXEL_DIFF = 6.0
PHOTO_HASH_FILENAME = "photo-hashes.json"


def gallery_hashes(exterior_dir: Path) -> dict[str, str]:
    """{filename: phash} for one vehicle's exterior photos."""
    out = {}
    for path in sorted(Path(exterior_dir).glob("*.jpg")):
        try:
            out[path.name] = str(_hash_image(Image.open(path)))
        except Exception:
            continue
    return out


def _mean_pixel_diff(a: Path, b: Path) -> float:
    """Mean absolute per-pixel difference, or inf if either can't be read.
    Only ever called on an exact-hash candidate, so this is cheap."""
    import numpy as np

    try:
        ia = Image.open(a).convert("RGB")
        ib = Image.open(b).convert("RGB")
    except Exception:
        return float("inf")
    if ia.size != ib.size:
        ib = ib.resize(ia.size)
    return float(np.abs(np.asarray(ia, dtype=np.float64) - np.asarray(ib, dtype=np.float64)).mean())


def find_shared_gallery(out_root: Path, folder_name: str, hashes: dict[str, str],
                         exterior_dir: Path) -> dict[str, int]:
    """{other_vehicle_folder: how many photos it shares with this one}.

    Reads the index written by record_gallery_hashes(); returns {} on a
    first run, an unreadable index, or no matches -- this is advisory, so
    it must never be the reason a scrape fails.
    """
    index_path = Path(out_root) / PHOTO_HASH_FILENAME
    try:
        index = json.loads(index_path.read_text())
    except (OSError, ValueError):
        return {}

    lookup: dict[str, list[tuple[str, str]]] = {}
    for other, entries in index.items():
        if other == folder_name:
            continue
        for name, h in (entries or {}).items():
            lookup.setdefault(h, []).append((other, name))

    shared: dict[str, int] = {}
    for name, h in hashes.items():
        for other, other_name in lookup.get(h, []):
            other_path = Path(out_root) / other / "images" / "exterior" / other_name
            if not other_path.is_file():
                continue
            if _mean_pixel_diff(Path(exterior_dir) / name, other_path) <= STOCK_RENDER_MAX_PIXEL_DIFF:
                shared[other] = shared.get(other, 0) + 1
                break
    return shared


def record_gallery_hashes(out_root: Path, folder_name: str, hashes: dict[str, str]) -> None:
    """Add this vehicle to the index so later scrapes can be compared
    against it. Best-effort: a corrupt index is rewritten, never fatal."""
    index_path = Path(out_root) / PHOTO_HASH_FILENAME
    try:
        index = json.loads(index_path.read_text())
        if not isinstance(index, dict):
            index = {}
    except (OSError, ValueError):
        index = {}
    index[folder_name] = hashes
    try:
        index_path.write_text(json.dumps(index, indent=2))
    except OSError:
        pass
