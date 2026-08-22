"""
Auto-select hero + accent photos for imaging/compose from a vehicle's
images/exterior/cutout/angles.json (see classify.py's AngleClassifier and
dtfb.py's download_photos, which writes that file).

pick_for_layout() is the one entrypoint most callers want -- routes to the
right picker for a given imaging/compose/layout.py layout name so a CLI
just needs to know the layout name, not which picker function goes with it.
"""
from __future__ import annotations

import json
from pathlib import Path

HERO_PRIORITY = ["front_3q", "rear_3q", "side", "front", "rear"]
ACCENT_PRIORITY = ["side", "front", "rear_3q", "rear", "front_3q"]

# Walkaround order for hero_video.py's animated carousel -- front to rear,
# 3q angles before straight-on since they're the more flattering of a
# front/front_3q or rear/rear_3q pair when only one should lead.
CAROUSEL_ANGLE_ORDER = ["front_3q", "front", "side", "rear_3q", "rear"]


def load_angles(cutout_dir: Path) -> dict[str, dict]:
    path = Path(cutout_dir) / "angles.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _best_by_angle(angles: dict) -> dict[str, tuple[str, float]]:
    """Best (filename, confidence) per angle label."""
    best: dict[str, tuple[str, float]] = {}
    for filename, info in angles.items():
        label, conf = info["angle"], info["confidence"]
        if label not in best or conf > best[label][1]:
            best[label] = (filename, conf)
    return best


def pick_hero_shots(cutout_dir: Path, n_accents: int = 2) -> list[Path]:
    """For layout.py's "single"/"corners" layouts: [hero, accent, ...],
    1 + up to n_accents paths. Picks a front-three-quarter hero when
    available (the standard "money angle"), then fills accent slots with
    the highest-confidence shot of each angle category not already used,
    so e.g. a side/rear-3q combo reads as genuinely different views rather
    than near-identical crops. Falls back to filename order if angles.json
    is missing or empty (e.g. --no-photo-sort was used)."""
    cutout_dir = Path(cutout_dir)
    angles = load_angles(cutout_dir)

    if not angles:
        files = sorted(cutout_dir.glob("*.png"))
        return files[: 1 + n_accents]

    best_by_angle = _best_by_angle(angles)

    hero_label = next((a for a in HERO_PRIORITY if a in best_by_angle), next(iter(best_by_angle)))
    hero_file = best_by_angle[hero_label][0]
    selected = [hero_file]

    for label in ACCENT_PRIORITY:
        if len(selected) > n_accents:
            break
        if label in best_by_angle and best_by_angle[label][0] not in selected:
            selected.append(best_by_angle[label][0])

    # still short (e.g. very few distinct angles detected)? fill with
    # whatever's left, highest confidence first.
    if len(selected) <= n_accents:
        remaining = sorted(
            ((filename, info) for filename, info in angles.items() if filename not in selected),
            key=lambda item: -item[1]["confidence"],
        )
        for filename, _info in remaining:
            if len(selected) > n_accents:
                break
            selected.append(filename)

    return [cutout_dir / f for f in selected[: 1 + n_accents]]


def pick_for_quad_layout(cutout_dir: Path) -> list[Path]:
    """For layout.py's "quad" layout: [hero, front, back, side]. Hero
    prefers front_3q (the standard dynamic hero angle); front/back/side
    each prefer their exact matching angle, with rear_3q as a fallback for
    "back" if a straight rear shot isn't in the gallery. Any role with no
    match at all is just omitted -- compose_hero() zips car_paths against
    the layout's boxes, so a shorter list simply leaves those slots empty
    rather than erroring."""
    cutout_dir = Path(cutout_dir)
    angles = load_angles(cutout_dir)
    if not angles:
        files = sorted(cutout_dir.glob("*.png"))
        return files[:4]

    best = _best_by_angle(angles)

    def pick(*labels):
        for label in labels:
            if label in best:
                return best[label][0]
        return None

    roles = [pick("front_3q", "front", "side"), pick("front", "front_3q"),
             pick("rear", "rear_3q"), pick("side")]

    seen = set()
    ordered = []
    for f in roles:
        if f and f not in seen:
            seen.add(f)
            ordered.append(f)
    return [cutout_dir / f for f in ordered]


def pick_for_layout(cutout_dir: Path, layout: str = "corners", n_accents: int = 2) -> list[Path]:
    """Single entrypoint: route to the right picker for a given
    imaging/compose layout name ("single", "corners", "quad")."""
    if layout == "quad":
        return pick_for_quad_layout(cutout_dir)
    if layout == "single":
        return pick_hero_shots(cutout_dir, n_accents=0)
    return pick_hero_shots(cutout_dir, n_accents=n_accents)


def pick_for_carousel(cutout_dir: Path, wheel_dir: Path | None = None) -> list[tuple[Path, str]]:
    """Every distinct angle available (best-confidence shot per angle, in
    CAROUSEL_ANGLE_ORDER's front-to-rear walkaround order) plus any wheel
    money shot(s) tacked on at the end as the "reveal" -- for
    hero_video.py's animated bottom slot, which (unlike the static
    single/corners/quad layouts) isn't limited to a handful of fixed roles
    and wants to show off everything there is, one at a time.

    Returns (path, angle_label) pairs, not bare paths -- hero_video.py's
    dynamic accent-rotation needs each shot's angle to avoid putting two
    redundant angles (e.g. two front-ish shots) on screen at once, and
    that label is cheap to hand over here rather than have the caller
    re-derive it from angles.json a second time. Wheel money shots get the
    synthetic label "wheel" (there's no angles.json entry for that folder).

    Any angle label the classifier produces that isn't in
    CAROUSEL_ANGLE_ORDER (a future label, or a low-confidence miscategorized
    shot) still gets included, just tacked on after the known angles rather
    than silently dropped -- this list drives what a viewer sees of the
    vehicle, so quietly omitting a real photo would be a worse failure than
    showing it in a slightly wrong position.
    """
    cutout_dir = Path(cutout_dir)
    angles = load_angles(cutout_dir)

    if angles:
        best = _best_by_angle(angles)
        ordered = [(label, best[label][0]) for label in CAROUSEL_ANGLE_ORDER if label in best]
        ordered += [(label, f) for label, (f, _c) in best.items() if label not in CAROUSEL_ANGLE_ORDER]
        selected = [(cutout_dir / f, label) for label, f in ordered]
    else:
        selected = [(f, "unknown") for f in sorted(cutout_dir.glob("*.png"))]

    if wheel_dir is not None:
        wheel_dir = Path(wheel_dir)
        if wheel_dir.is_dir():
            selected += [(f, "wheel") for f in sorted(wheel_dir.glob("*.png"))]

    return selected


FRONT_ACCENT_PRIORITY = ["front", "front_3q"]
REAR_ACCENT_PRIORITY = ["rear", "rear_3q"]


def pick_for_conveyor(cutout_dir: Path) -> list[Path]:
    """[hero, left_accent, right_accent] for compose/layout.py's
    "conveyor" layout -- the still hero image mirroring hero_video.py's
    proportions (2 accents up top, 1 big hero below).

    The two accents are ROLE-assigned, not just "two more angles": front
    on the left, tail on the right, so the pair reads as bracketing the
    vehicle rather than as an arbitrary sample. That ordering only makes
    sense because the layout's slots are fixed left/right -- contrast
    pick_hero_shots(), whose accents are interchangeable and so are
    picked purely by "most distinct angle available".

    Each slot falls back to whatever distinct angle is left (highest
    confidence first) when its preferred bucket is empty or already
    spent -- e.g. a hero that took front_3q leaves the left slot to a
    straight-on "front", and a vehicle photographed without any rear
    shot still gets a filled right slot rather than a hole. Returns
    fewer than 3 paths only when the gallery genuinely has fewer distinct
    angles; compose_hero() leaves unfilled slots empty, and
    compose_vehicle() prefers a different layout in that case.
    """
    cutout_dir = Path(cutout_dir)
    angles = load_angles(cutout_dir)
    if not angles:
        return sorted(cutout_dir.glob("*.png"))[:3]

    best = _best_by_angle(angles)
    used: set[str] = set()

    def take(preferred: list[str]) -> str | None:
        for label in preferred:
            if label in best and best[label][0] not in used:
                used.add(best[label][0])
                return best[label][0]
        # Nothing in the preferred bucket -- any distinct angle beats an
        # empty slot, best-confidence first.
        for label, (filename, _conf) in sorted(best.items(), key=lambda kv: -kv[1][1]):
            if filename not in used:
                used.add(filename)
                return filename
        return None

    picks = [take(HERO_PRIORITY), take(FRONT_ACCENT_PRIORITY), take(REAR_ACCENT_PRIORITY)]
    return [cutout_dir / f for f in picks if f]


def pick_all_for_carousel(cutout_dir: Path, wheel_dir: Path | None = None) -> list[tuple[Path, str]]:
    """EVERY cutout, not deduped to one-best-per-angle like
    pick_for_carousel() -- for hero_video.py's 3-slot conveyor mode, which
    wants to show off the complete shot library (confirmed real case: 9
    total shots -- 8 exterior cutouts plus a wheel money shot -- for a
    vehicle whose distinct angles alone would dedupe down to 6). Grouped
    by CAROUSEL_ANGLE_ORDER (all front_3q's together highest-confidence
    first, then front's, etc.) so the walkaround still reads front-to-rear
    even with duplicates, rather than dumping them in raw filename order.
    """
    cutout_dir = Path(cutout_dir)
    angles = load_angles(cutout_dir)

    if angles:
        by_angle: dict[str, list[tuple[str, float]]] = {}
        for filename, info in angles.items():
            by_angle.setdefault(info["angle"], []).append((filename, info["confidence"]))

        selected = []
        for label in CAROUSEL_ANGLE_ORDER:
            for filename, _conf in sorted(by_angle.get(label, []), key=lambda x: -x[1]):
                selected.append((cutout_dir / filename, label))
        for label, files in by_angle.items():
            if label in CAROUSEL_ANGLE_ORDER:
                continue
            for filename, _conf in sorted(files, key=lambda x: -x[1]):
                selected.append((cutout_dir / filename, label))
    else:
        selected = [(f, "unknown") for f in sorted(cutout_dir.glob("*.png"))]

    if wheel_dir is not None:
        wheel_dir = Path(wheel_dir)
        if wheel_dir.is_dir():
            selected += [(f, "wheel") for f in sorted(wheel_dir.glob("*.png"))]

    return selected


def pick_adaptive(cutout_dir: Path) -> tuple[str, list[Path]]:
    """Chooses between quad (hero + 3 distinct accents) and corners (hero +
    1-2) based on how many DISTINCT angles are actually available, instead
    of always forcing quad's 4 fixed slots.

    Confirmed failure mode on a real vehicle (a Maverick photographed from
    only 3 distinct angles -- front_3q, rear_3q, side, no straight front or
    rear shot in the gallery): pick_for_quad_layout()'s "front" role has no
    real match, falls back to the same file already used for the hero, and
    gets deduped away -- leaving quad's middle strip completely empty, a
    big dead gap of background between two accent boxes that are already
    sized small (0.32w x 0.26h) to make room for a strip that isn't there.

    corners_layout's boxes are already sized bigger (0.40w x 0.30h) since
    they don't have to coexist with that middle strip, so switching to it
    when there are only 1-2 real accents fills more of the actually
    available space instead of leaving a hole. Falls through to "single"
    if there's only ever one usable angle at all."""
    quad_cutouts = pick_for_quad_layout(cutout_dir)
    n_accents = len(quad_cutouts) - 1
    if n_accents >= 3:
        return "quad", quad_cutouts
    if n_accents >= 1:
        return "corners", pick_hero_shots(cutout_dir, n_accents=n_accents)
    return "single", pick_hero_shots(cutout_dir, n_accents=0)


def order_for_conveyor_start(pairs: list[tuple[Path, str]], cutout_dir: Path) -> list[tuple[Path, str]]:
    """Rotate/reseat `pairs` so the conveyor's FIRST frame shows exactly
    the hero still's arrangement: front on the left, angle in the hero
    slot, back on the right.

    The conveyor draws left=shots[-1], hero=shots[0], right=shots[1], so
    that arrangement is produced by putting the still's hero at index 0,
    its right accent at index 1, and its left accent last. Because a run
    is exactly one full pass, landing there also means the last frame
    hands back to the first -- the video loops seamlessly on autoplay
    instead of snapping to an unrelated shot.

    Costs the strict front-to-rear walkaround order (the "back" shot now
    comes second rather than late), which is worth it: the opening frame
    is the one a scroller actually sees, and matching the still makes the
    post read as one piece.
    """
    hero, left, right = (pick_for_conveyor(cutout_dir) + [None, None, None])[:3]
    by_path = {p: (p, lbl) for p, lbl in pairs}
    lead = [by_path[p] for p in (hero, right) if p in by_path]
    tail = [by_path[p] for p in (left,) if p in by_path and p not in (hero, right)]
    used = {p for p, _ in lead + tail}
    middle = [pair for pair in pairs if pair[0] not in used]
    return lead + middle + tail
