"""
Download, parse, and render a vehicle's window sticker.

Distinct from imaging/sticker.py (which only parses PDF *content* --
equipment/pricing/warranties, and finds where a two-panel page's gutter
sits): this module owns the download, the placeholder-PDF check, the
pdftoppm render, and turning that render into two clean portrait panel
images. imaging/sticker.py is a pure PDF-text library; this is the
pipeline that drives it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import requests
from PIL import Image

from dtfb.imaging.sticker import find_panel_split_x, parse_sticker
from dtfb.scrape import Vehicle, download_file

PLACEHOLDER_STICKER_MARKERS = ("check back later", "has not yet been", "not available")


def is_placeholder_sticker(pdf_path: Path) -> bool:
    """forddirect.com returns a real, well-formed 1-page PDF -- HTTP 200, valid
    PDF -- that just says "Please check back later" for vehicles it has no
    sticker for (seen on a non-Ford trade-in, but likely also happens for
    Fords before the sticker is published). Detect it by text content rather
    than trusting a 200 status."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=20,
        )
        text = result.stdout.lower()
        return any(marker in text for marker in PLACEHOLDER_STICKER_MARKERS)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def parse_window_sticker(v: Vehicle, pdf_path: Path, out_dir: Path) -> None:
    """Extract structured feature/warranty/pricing data from a real (non-
    placeholder) window sticker and stash it on v.sticker for
    build_facebook_post() to prefer over the site's own (less detailed)
    feature blob. A real sticker's layout is a fixed Ford template, but PDFs
    are still an external input -- never let a parse hiccup here fail the
    whole vehicle scrape, just fall back to the site data with a warning."""
    try:
        data = parse_sticker(pdf_path)
    except Exception as e:
        print(f"    ! window sticker parse failed: {e}", file=sys.stderr)
        v.warnings.append("Window sticker downloaded but structured parsing failed; "
                           "using site-listed features as fallback.")
        return
    if not data.get("equipment"):
        v.warnings.append("Window sticker parsed but no equipment data found; "
                           "using site-listed features as fallback.")
        return
    v.sticker = data
    (out_dir / "window-sticker.json").write_text(json.dumps(data, indent=2))
    v.warnings.append("Extracted structured feature data from window sticker (window-sticker.json).")


def split_sticker_panels(bundle_dir: Path, pdf_path: Path, dpi: int) -> None:
    """A Ford Monroney sticker commonly prints as two side-by-side panels
    on one landscape page (equipment/pricing on the left, fuel economy/
    safety ratings on the right) meant to be read/posted as two separate
    portrait images, not one wide strip -- a wide render is a "crammed"
    shot on mobile and holds up worse under upload recompression than two
    normally-proportioned images would. Replaces each rendered page with
    left/right crops (window-sticker-1a.png / -1b.png, ...) when
    imaging/sticker.py::find_panel_split_x() finds a real gutter (leaves
    the page untouched if it doesn't -- a genuinely single-flow layout).
    That text-based x is only an approximate anchor; _refine_split_bounds()
    pinpoints the exact pixel gap against the rendered image itself before
    cropping, so each panel ends flush against its own border rule rather
    than carrying a dangling strip of dead white space."""
    split_x_pt = find_panel_split_x(pdf_path)
    if split_x_pt is None:
        return
    approx_split_x_px = round(split_x_pt * dpi / 72)

    for page_path in sorted(bundle_dir.glob("window-sticker-*.png")):
        img = Image.open(page_path)
        w, h = img.size
        gap_start, gap_end = _refine_split_bounds(img, approx_split_x_px)
        img.crop((0, 0, gap_start, h)).save(page_path.with_name(page_path.stem + "a.png"))
        img.crop((gap_end, 0, w, h)).save(page_path.with_name(page_path.stem + "b.png"))
        page_path.unlink()


def _refine_split_bounds(img: Image.Image, approx_x_px: int, search_radius_px: int = 80,
                          white_threshold: int = 245, min_white_frac: float = 0.9) -> tuple[int, int]:
    """find_panel_split_x()'s text-bbox gutter is only an approximate
    anchor -- the real sticker prints an actual cut-guide at the panel
    boundary (confirmed on a real render: a ~4px colored rule, a clean
    ~14px white gap, then a ~8px black rule), and text coordinates alone
    don't say exactly where that gap's white buffer sits in pixel space.

    Scans a window of columns around the approximate split for the widest
    run of near-pure-white columns (checked across the full image height,
    same idea as find_panel_split_x() but on pixels instead of word boxes)
    and returns its (start, end) -- the caller crops panel A up to `start`
    (right after its own rule, e.g. the blue box border) and panel B from
    `end` (right before its own rule, e.g. the black box border), so the
    dead white buffer between the two rules is dropped entirely rather
    than being split in half and left dangling on both panels (confirmed
    on a real render: exactly this looked like ~7px of stray nothing
    trailing each panel's edge). Falls back to (approx_x_px, approx_x_px)
    -- a zero-width gap, i.e. no dead space to drop -- if no clean run is
    found nearby (e.g. a differently-drawn template with no such guide)."""
    import numpy as np

    arr = np.array(img.convert("RGB"))
    lo = max(0, approx_x_px - search_radius_px)
    hi = min(arr.shape[1], approx_x_px + search_radius_px)
    frac_white = np.all(arr[:, lo:hi, :] > white_threshold, axis=2).mean(axis=0)

    runs, start = [], None
    for i, fw in enumerate(frac_white):
        if fw > min_white_frac:
            start = i if start is None else start
        elif start is not None:
            runs.append((i - start, start, i))
            start = None
    if start is not None:
        runs.append((len(frac_white) - start, start, len(frac_white)))
    if not runs:
        return approx_x_px, approx_x_px

    _, best_start, best_end = max(runs)
    return lo + best_start, lo + best_end


def download_window_sticker(session: requests.Session, v: Vehicle, out_dir: Path, dpi: int) -> list[str]:
    """PDF + parsed JSON are source/reference data and stay at the vehicle
    folder's top level; the rendered page image is a postable photo like
    hero.png/framed/*, so it lands in bundle/ alongside them (see
    vehicle_pipeline.py::process_vehicle() and imaging/compose/pipeline.py
    for the rest of bundle/'s contents -- this is the one producer of it
    that isn't compose_vehicle() itself)."""
    if not v.window_sticker_url:
        return []
    pdf_path = out_dir / "window-sticker.pdf"
    if not download_file(session, v.window_sticker_url, pdf_path):
        return []
    if is_placeholder_sticker(pdf_path):
        pdf_path.unlink(missing_ok=True)
        v.warnings.append("Window sticker not yet published for this vehicle (placeholder page returned).")
        return []
    bundle_dir = out_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Clear any sticker images from a previous run first -- on a --force
    # re-fetch, leftover split panels (window-sticker-1a.png etc.) would
    # otherwise sit alongside the freshly rendered pages and both match
    # split_sticker_panels()'s glob, getting re-split a second time against
    # coordinates sized for the full page (confirmed failure mode: crashed
    # with "right < left" cropping an already-narrow panel).
    for stale in bundle_dir.glob("window-sticker-*.png"):
        stale.unlink()
    try:
        result = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(bundle_dir / "window-sticker")],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"    ! pdftoppm failed: {result.stderr.strip()}", file=sys.stderr)
            v.warnings.append("Window sticker PDF downloaded but PNG conversion failed.")
            return []
        else:
            parse_window_sticker(v, pdf_path, out_dir)
            try:
                split_sticker_panels(bundle_dir, pdf_path, dpi)
            except Exception as e:
                print(f"    ! sticker panel split failed: {e}", file=sys.stderr)
                v.warnings.append(f"Window sticker page split failed, left as one wide image: {e}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"    ! pdftoppm error: {e}", file=sys.stderr)
        v.warnings.append("Window sticker PDF downloaded but PNG conversion failed "
                           "(is poppler-utils / pdftoppm installed?).")
        return []
    return sorted(p.name for p in bundle_dir.glob("window-sticker*.png"))
