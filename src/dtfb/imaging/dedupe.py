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
import sqlite3
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


# --- Cross-vehicle photo cache ------------------------------------------
#
# Separate concern from the junk templates above: this isn't about
# dropping anything, it's recognizing when THIS EXACT photo (same bytes)
# has already been downloaded and fully processed for a DIFFERENT vehicle
# -- overwhelmingly manufacturer stock renders, since the dealer's CDN
# mints a fresh unique URL per vehicle even for byte-identical content
# (confirmed: dozens of same-trim new vehicles, e.g. Bronco Sport BIG
# Bend, share 13-14 of 14 exterior photos outright). The download itself
# can't be skipped -- there is no URL-level signal, only the bytes prove
# identity -- but the expensive part (CLIP classification, rembg cutout)
# can be, by reusing what an earlier vehicle already computed for the
# identical content instead of redoing it. Nothing is stripped from any
# vehicle's output: every photo is still saved to that vehicle's own
# gallery exactly as before, just without redundant GPU work behind it.
#
# EXACT hash match, then a pixel confirmation, same conservative pairing
# the fleet's own photography needs: 466 pairs from genuinely DIFFERENT
# vehicles fall within the 8-bit junk-filter threshold of each other (a
# Model Y matches a Camry there), so nothing looser than exact-then-pixel
# is safe here. See the module docstring for that measurement.
STOCK_RENDER_MAX_BITS = 0
STOCK_RENDER_MAX_PIXEL_DIFF = 6.0
PHOTO_HASH_FILENAME = "photo-hashes.json"  # legacy, migrated into the DB below
PHOTO_CACHE_DB_FILENAME = "photo-cache.db"
_BLOB_DIRNAME = ".photo-blobs"


def _hash_str(img: Image.Image) -> str:
    return str(_hash_image(img))


def _db_path(out_root: Path) -> Path:
    return Path(out_root) / PHOTO_CACHE_DB_FILENAME


def _connect(out_root: Path) -> sqlite3.Connection:
    """Opens (creating if needed) the shared photo cache DB, one-time
    migrating the legacy per-vehicle JSON index if it's still there and
    the DB is empty -- so a mid-fleet upgrade doesn't forget every hash
    accumulated so far and start re-processing photos it already knows
    about. The old file is left in place (renamed .migrated) as a backup,
    never deleted."""
    conn = sqlite3.connect(_db_path(out_root))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photo_cache (
            phash TEXT PRIMARY KEY,
            folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            category TEXT
        )
    """)
    conn.commit()

    legacy_path = Path(out_root) / PHOTO_HASH_FILENAME
    if legacy_path.exists():
        row_count = conn.execute("SELECT COUNT(*) FROM photo_cache").fetchone()[0]
        if row_count == 0:
            try:
                legacy = json.loads(legacy_path.read_text())
            except (OSError, ValueError):
                legacy = {}
            rows = [(h, folder, name, "exterior")
                    for folder, entries in legacy.items()
                    for name, h in (entries or {}).items()]
            if rows:
                # INSERT OR IGNORE: the legacy index could (rarely) have
                # recorded the same hash under two folders if a photo was
                # shared 3+ ways -- first writer wins, same as the DB's own
                # PRIMARY KEY semantics for any future collision.
                conn.executemany(
                    "INSERT OR IGNORE INTO photo_cache (phash, folder, filename, category) VALUES (?, ?, ?, ?)",
                    rows)
                conn.commit()
            try:
                legacy_path.rename(legacy_path.with_suffix(".json.migrated"))
            except OSError:
                pass
    return conn


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


def blob_path(out_root: Path, phash: str) -> Path:
    """Where a content-addressed cutout copy for this hash lives -- shared
    across every vehicle whose photo hashes to it, so N vehicles sharing
    one stock render store the cutout ONCE instead of N times."""
    return Path(out_root) / _BLOB_DIRNAME / f"{phash}.png"


def photo_hash(content: bytes) -> str | None:
    """The one hash computed per downloaded photo -- callers pass it to
    both lookup_cached_photo() and record_photo() so it's never computed
    twice for the same bytes. None if the bytes aren't a readable image."""
    try:
        return _hash_str(Image.open(io.BytesIO(content)))
    except Exception:
        return None


def lookup_cached_photo(out_root: Path, phash: str, content: bytes, own_folder: str) -> dict | None:
    """If `content` (raw downloaded photo bytes, already hashed to `phash`
    via photo_hash()) is confirmed identical to a photo already recorded
    from a DIFFERENT vehicle, returns {"folder", "filename", "category"}
    for the match -- the caller can reuse that category (skip CLIP) and,
    if a cutout blob exists at blob_path(), reuse it too (skip rembg).
    Returns None on a cache miss, an unreadable DB, the source vehicle's
    file having since moved/gone, or any error -- this is advisory, same
    as everything else in this module, and must never be the reason a
    scrape fails or a photo goes unprocessed.
    """
    try:
        conn = _connect(out_root)
        row = conn.execute(
            "SELECT folder, filename, category FROM photo_cache WHERE phash = ? AND folder != ?",
            (phash, own_folder)).fetchone()
        conn.close()
        if row is None:
            return None
        folder, filename, category = row
        # category names the subfolder directly ("exterior"/"interior") --
        # nearly every real hit is exterior stock imagery, but this stays
        # correct for the rare interior/other match too.
        source_path = Path(out_root) / folder / "images" / category / filename
        if not source_path.is_file():
            return None
        import numpy as np
        this_arr = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"), dtype=np.float64)
        other_arr = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float64)
        if this_arr.shape != other_arr.shape:
            return None
        if float(np.abs(this_arr - other_arr).mean()) > STOCK_RENDER_MAX_PIXEL_DIFF:
            return None
        return {"folder": folder, "filename": filename, "category": category, "phash": phash}
    except Exception:
        return None


def record_photo(out_root: Path, folder: str, filename: str, phash: str, category: str,
                  cutout_bytes: bytes | None = None) -> None:
    """Records this photo's hash so later vehicles can find and reuse it.
    Best-effort: never raises.

    If `cutout_bytes` is given, also stores a shared content-addressed
    copy of the cutout at blob_path() -- UNLESS one already exists there,
    since a matching phash means byte-identical source content, so the
    first real cutout is as good as any later one.

    Confirmed real bug this upgrade-path avoids: a hash migrated from the
    legacy JSON index (see _connect()) has no blob behind it at all --
    that index only ever stored hashes, never images. Plain INSERT OR
    IGNORE would let that blob-less row permanently squat on its phash's
    primary-key slot, so no vehicle processed AFTER the migration could
    ever backfill a real, reusable blob for a hash the fleet has been
    sharing since before the DB existed -- every future encounter would
    keep re-paying full cutout cost forever, not just once. Upgrading in
    place the first time a real cutout shows up for that hash fixes it
    going forward, at the cost of one paid "warm-up" cutout per
    previously-blob-less hash, which is unavoidable -- nothing can
    conjure a cutout image the old index never saved."""
    try:
        conn = _connect(out_root)
        if cutout_bytes is not None and not blob_path(out_root, phash).is_file():
            # We have a real cutout and no blob-backed entry claims this
            # hash yet -- REPLACE takes ownership (upgrade), whether the
            # row is new or a blob-less legacy migration.
            conn.execute(
                "INSERT OR REPLACE INTO photo_cache (phash, folder, filename, category) VALUES (?, ?, ?, ?)",
                (phash, folder, filename, category))
        else:
            conn.execute(
                "INSERT OR IGNORE INTO photo_cache (phash, folder, filename, category) VALUES (?, ?, ?, ?)",
                (phash, folder, filename, category))
        conn.commit()
        conn.close()
    except Exception:
        pass
    if cutout_bytes is not None:
        try:
            path = blob_path(out_root, phash)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(cutout_bytes)
        except OSError:
            pass


def gallery_hashes(exterior_dir: Path) -> dict[str, str]:
    """{filename: phash} for one vehicle's exterior photos. Kept for any
    external/debugging use; download_photos() no longer needs this --
    it hashes and records each photo as it's downloaded, via
    lookup_cached_photo()/record_photo() above."""
    out = {}
    for path in sorted(Path(exterior_dir).glob("*.jpg")):
        try:
            out[path.name] = _hash_str(Image.open(path))
        except Exception:
            continue
    return out
