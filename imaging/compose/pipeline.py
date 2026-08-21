"""
Per-vehicle image-composition pipeline: one hero collage (quad: hero + 3
distinct accents when the gallery actually has that many angles, else
corners: hero + 1-2 bigger accents -- see select.py::pick_adaptive() for
why quad's fixed 4 slots look sparse/wasteful on a vehicle with fewer real
angles) PLUS a solo "framed" image for every exterior cutout, including the
ones already in the collage, since they only show up small there. Each
framed image is scaled to fill the border's window as big as the
collision-avoidance logic allows, with the same background/vignette/glow
treatment as the hero. Interior cutouts aren't touched here at all.

Both hero and framed are just compose_hero() calls with different
layouts/car lists; this module's only job is deciding which cutout goes
where and writing the files. See hero.py for the actual composition
(background fit, spotlight, glow, border collision-avoidance).
"""
from __future__ import annotations

from pathlib import Path

from .hero import compose_hero
from ..select import pick_adaptive, pick_for_conveyor

# The conveyor layout is sized assuming a tighter inset than the other
# layouts' 0.06 -- see conveyor_layout(). hero_video.py splits this into
# separate hero/accent values; a single value is close enough here (they
# differ by ~7px on this border) and keeps compose_hero()'s signature
# from growing a per-slot margin list for one caller.
CONVEYOR_MARGIN_FRAC = 0.03

# Extra shapes for the hero still, mirroring hero_video.py's VIDEO_FORMATS.
# Only the HERO gets them: framed/ is one solo composition per cutout and
# rendering every one of those four times over would quadruple the folder
# (213 images fleet-wide becomes 850) for gallery filler, where the hero is
# the image that actually leads a post.
#
# 4:5 rather than another 9:16 for the feed shape -- 1080x1350 is the
# tallest Instagram and Facebook render in-feed without cropping, so it is
# the most screen a still can occupy there.
HERO_STILL_FORMATS = {
    "square": (1254, 1254),      # Marketplace, and the existing hero.png
    "portrait": (1080, 1350),    # 4:5, Instagram/Facebook feed
    "vertical": (1080, 1920),    # 9:16, Stories
}
DEFAULT_HERO_STILL_FORMAT = "square"


def _backdrop(background_path, out_dir: Path, image_key: str, gradient: bool,
               exterior: str | None, interior: str | None, sample_path: Path | None,
               canvas: tuple[int, int] | None = None):
    """Either the shared background asset, or a per-image gradient built
    from this vehicle's own colors.

    Seeded per (vehicle, image) rather than globally: a single shared
    backdrop across every post is what got flagged, so every image needs
    its own, while a rerun must still reproduce the same file rather than
    churning the whole bundle."""
    if not gradient:
        return background_path
    from .background import vehicle_gradient_background
    from .hero import DEFAULT_CANVAS_SIZE
    return vehicle_gradient_background(canvas or DEFAULT_CANVAS_SIZE,
                                        f"{out_dir.parent.name}/{image_key}",
                                        exterior, interior, sample_path)


def hero_still_name(fmt: str) -> str:
    """hero.png stays the square one so nothing pointing at it breaks."""
    return "hero.png" if fmt == DEFAULT_HERO_STILL_FORMAT else f"hero-{fmt}.png"


def compose_vehicle(cutout_dir: Path, out_dir: Path, background_path, border_path: Path | None,
                     glow: bool = True, glow_color: str = "white",
                     glow_radius: int = 24, glow_intensity: float = 0.75,
                     gradient: bool = False, exterior_color: str | None = None,
                     interior_color: str | None = None,
                     hero_formats: tuple[str, ...] = (DEFAULT_HERO_STILL_FORMAT,)) -> dict:
    """Returns {"hero": Path|None, "framed": [Path, ...]}. Produces nothing
    (empty result, no error) if cutout_dir has no usable cutouts -- e.g. a
    used vehicle with no clean exterior shots to cut out at all; callers
    should treat that as "nothing to compose", not a failure."""
    cutout_dir = Path(cutout_dir)
    out_dir = Path(out_dir)
    cutout_files = sorted(cutout_dir.glob("*.png")) if cutout_dir.is_dir() else []
    result = {"hero": None, "heroes": {}, "framed": []}
    if not cutout_files:
        return result

    # Same 2-up/1-down proportions as the animated hero (hero_video.py),
    # so the still and the video read as the same piece rather than two
    # different templates -- and with the accents ROLE-assigned (front
    # left, tail right) instead of just "two more angles". Needs 3
    # distinct angles to fill; below that pick_adaptive()'s corners/single
    # fallback still looks better than a conveyor with a hole in it.
    hero_cutouts = pick_for_conveyor(cutout_dir)
    layout, margin_frac = "conveyor", CONVEYOR_MARGIN_FRAC
    if len(hero_cutouts) < 3:
        layout, hero_cutouts = pick_adaptive(cutout_dir)
        margin_frac = 0.06

    if hero_cutouts:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in hero_formats:
            canvas = HERO_STILL_FORMATS[fmt]
            # Seeded per format as well as per vehicle: two shapes of the
            # same hero sharing one gradient would be the same near-
            # duplicate backdrop the per-image seeding exists to avoid.
            hero_bg = _backdrop(background_path, out_dir, f"hero/{fmt}", gradient,
                                 exterior_color, interior_color, hero_cutouts[0], canvas)
            hero_img = compose_hero(hero_bg, border_path, hero_cutouts, layout=layout,
                                     glow=glow, glow_color=glow_color, margin_frac=margin_frac,
                                     glow_radius=glow_radius, glow_intensity=glow_intensity,
                                     canvas_size=canvas)
            hero_path = out_dir / hero_still_name(fmt)
            hero_img.save(hero_path)
            result["heroes"][fmt] = hero_path
        result["hero"] = result["heroes"].get(DEFAULT_HERO_STILL_FORMAT) or next(
            iter(result["heroes"].values()))

    # Every exterior cutout gets its own solo framed image, including the
    # 4 used in the collage above -- they only appear small as accents
    # there, and each one is a distinct angle worth showing off full-size.
    # Clear framed/ rather than overwriting into it: it holds one file per
    # cutout, so a cutout that has since been renumbered or dropped (a
    # re-scrape, or a demotion by imaging/gallery.py) would otherwise leave
    # an orphan behind, and this folder is consumed by globbing.
    # Deliberately BEFORE the loop and deliberately compose_vehicle's job:
    # compose_wheel_shots() writes into this same directory and is always
    # called AFTER this function, so its output survives.
    framed_dir = out_dir / "framed"
    if framed_dir.is_dir():
        for stale in framed_dir.iterdir():
            if stale.is_file() and stale.stem.isdigit():
                stale.unlink()
    framed_dir.mkdir(parents=True, exist_ok=True)
    for cutout in cutout_files:
        bg = _backdrop(background_path, out_dir, f"framed/{cutout.name}", gradient,
                        exterior_color, interior_color, cutout)
        img = compose_hero(bg, border_path, [cutout], layout="single",
                            glow=glow, glow_color=glow_color,
                            glow_radius=glow_radius, glow_intensity=glow_intensity)
        framed_path = framed_dir / cutout.name
        img.save(framed_path)
        result["framed"].append(framed_path)

    return result


def compose_interiors(interior_dir: Path, out_dir: Path, vehicle: dict,
                       classifier=None, captions: bool = False) -> list[dict]:
    """Interior photos, white-balanced and exposure-corrected, into
    bundle/interior/. Unlike the exterior deliverables these are whole
    photos at original resolution and nothing is drawn on them -- see
    imaging/interior.py for why a cutout/backdrop treatment cannot work on
    a cabin, and why feature captions are opt-in rather than default."""
    from ..interior import process_gallery
    return process_gallery(interior_dir, out_dir / "interior", vehicle, classifier, captions)


def compose_wheel_shots(wheel_cutout_dir: Path, out_dir: Path, background_path, border_path: Path | None,
                         glow: bool = True, glow_color: str = "white",
                         glow_radius: int = 24, glow_intensity: float = 0.75,
                         gradient: bool = False, exterior_color: str | None = None,
                         interior_color: str | None = None) -> list[Path]:
    """One solo composition per confirmed wheel-money-shot cutout (see
    imaging/wheel.py and photos.py's images/exterior/wheels/), same
    single-layout/background/glow treatment as compose_vehicle()'s framed
    shots -- lands in bundle/framed/ right alongside them (a separate
    bundle/wheel-shots/ folder made the output harder to navigate, not
    easier: two folders to check for "every angle solo-composed" instead
    of one). No filename collision risk: wheel cutouts and regular exterior
    cutouts are numbered from the same per-photo counter in
    photos.py::download_photos(), so a given number is never both. Returns
    [] if there are no wheel cutouts for this vehicle -- most won't have
    one, that's expected, not an error."""
    wheel_cutout_dir = Path(wheel_cutout_dir)
    out_dir = Path(out_dir)
    wheel_files = sorted(wheel_cutout_dir.glob("*.png")) if wheel_cutout_dir.is_dir() else []
    if not wheel_files:
        return []

    framed_dir = out_dir / "framed"
    framed_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for cutout in wheel_files:
        bg = _backdrop(background_path, out_dir, f"framed/{cutout.name}", gradient,
                        exterior_color, interior_color, cutout)
        img = compose_hero(bg, border_path, [cutout], layout="single",
                            glow=glow, glow_color=glow_color,
                            glow_radius=glow_radius, glow_intensity=glow_intensity)
        out_path = framed_dir / cutout.name
        img.save(out_path)
        results.append(out_path)

    return results
