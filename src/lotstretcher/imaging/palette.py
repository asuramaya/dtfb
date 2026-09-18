"""
Turning a vehicle's own colors into a backdrop palette.

The scraped record carries colors as MARKETING names ("Agate Black",
"Antimatter Blue", "Avalanche"), not values, so this does two things:
parses the name for a base color word, and -- when the name is pure
branding with no color word in it at all, which "Avalanche" and
"Carbonized" both are -- falls back to measuring the paint straight off
the vehicle's own cutout. The fallback is the reliable one; the name map
just avoids a file read when the answer is already written down.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

# Base color words, longest-first at lookup time so "dark blue" doesn't
# match on a shorter substring first. Values are deliberately muted --
# these become a BACKDROP, and a fully saturated primary behind a car
# reads as a clip-art collage.
COLOR_WORDS: dict[str, tuple[int, int, int]] = {
    "black": (26, 26, 30), "white": (232, 233, 236), "silver": (188, 190, 195),
    "platinum": (198, 200, 204), "gray": (124, 127, 132), "grey": (124, 127, 132),
    "charcoal": (62, 64, 69), "graphite": (78, 81, 86), "carbonized": (88, 91, 96),
    "gunmetal": (84, 88, 94), "slate": (98, 106, 118), "steel": (110, 122, 138),
    "navy": (26, 42, 88), "blue": (38, 84, 168), "teal": (30, 118, 128),
    "turquoise": (48, 150, 156), "aqua": (60, 150, 160), "cyan": (50, 150, 170),
    "green": (40, 112, 72), "olive": (98, 104, 60), "lime": (120, 160, 60),
    "red": (168, 34, 44), "maroon": (108, 30, 46), "burgundy": (100, 28, 48),
    "crimson": (152, 28, 48), "ruby": (150, 30, 52), "rose": (180, 90, 105),
    "pink": (198, 110, 140), "purple": (96, 60, 142), "violet": (104, 66, 150),
    "orange": (206, 108, 38), "amber": (198, 140, 46), "gold": (176, 144, 70),
    "yellow": (214, 184, 56), "bronze": (136, 98, 60), "copper": (166, 100, 58),
    "brown": (92, 64, 46), "espresso": (72, 52, 42), "chestnut": (110, 70, 50),
    "tan": (188, 160, 124), "beige": (196, 178, 148), "sand": (196, 180, 150),
    "stone": (168, 164, 154), "cream": (218, 208, 186), "ivory": (222, 214, 196),
}

# Where backdrop colors are allowed to land. Cutouts are overwhelmingly
# white/silver/grey vehicles, so a light backdrop flattens them; every
# sampled color gets compressed into this value band regardless of how
# bright the paint actually is. Hue survives, which is what carries "this
# is that vehicle's color".
BACKDROP_VALUE_MIN = 0.10
BACKDROP_VALUE_MAX = 0.52
MIN_STOP_SEPARATION = 0.14  # in HSV value, between the gradient's two stops

NEUTRAL_SATURATION_CEILING = 0.10  # below this the source reads as white/grey/black

# When a vehicle's exterior and interior are the SAME color (black-on-
# black is the single most common combination), separating the two stops
# by value alone yields a single-hue ramp -- fine, but the flattest and
# least distinctive backdrop we produce, which is a problem when the
# whole point is that no two posts share one. So matching colors also get
# the far stop shifted off-hue: a real (if subtle) two-tone.
# A saturated pair (red-on-red) just rotates hue slightly. A NEUTRAL pair
# has no hue to rotate -- black, white and silver are all ~0 saturation,
# so rotating does nothing -- and instead gets a cool tint mixed in,
# which is what a studio backdrop falloff looks like anyway.
MATCHING_HUE_SHIFT = 0.055
NEUTRAL_TINT_HUE = 0.60
NEUTRAL_TINT_SATURATION = 0.30
BACKDROP_SATURATION_RANGE = (0.28, 0.78)


def parse_color_name(name: str | None) -> tuple[int, int, int] | None:
    """RGB for the first base color word found in a marketing name, or
    None if the name is pure branding ("Avalanche", "Iconic")."""
    if not name:
        return None
    lowered = name.lower()
    for word in sorted(COLOR_WORDS, key=len, reverse=True):
        if word in lowered:
            return COLOR_WORDS[word]
    return None


def sample_cutout_color(cutout_path: Path) -> tuple[int, int, int] | None:
    """The vehicle's paint color, measured from a cutout.

    Takes the median of the BRIGHTER half of the opaque pixels rather
    than of all of them: a cutout is maybe a third glass, tire and
    shadow, and a plain median over everything drags any paint color
    toward near-black. Median (not mean) so the remaining dark trim and
    specular highlights don't pull the result either way.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(cutout_path).convert("RGBA")
    arr = np.asarray(img, dtype=np.float64)
    opaque = arr[..., 3] > 200
    if opaque.sum() < 50:
        return None
    rgb = arr[..., :3][opaque]
    value = rgb.max(axis=1)
    bright = rgb[value >= np.percentile(value, 50)]
    if len(bright) < 20:
        bright = rgb
    return tuple(int(round(c)) for c in np.median(bright, axis=0))


def _to_backdrop(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """(h, s, v) compressed into the backdrop bands above."""
    h, s, v = colorsys.rgb_to_hsv(*[c / 255 for c in rgb])
    if s > NEUTRAL_SATURATION_CEILING:
        lo, hi = BACKDROP_SATURATION_RANGE
        s = min(max(s, lo), hi)
    v = BACKDROP_VALUE_MIN + v * (BACKDROP_VALUE_MAX - BACKDROP_VALUE_MIN)
    return h, s, v


def _hsv_bytes(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return round(r * 255), round(g * 255), round(b * 255)


def vehicle_gradient_colors(exterior: str | None, interior: str | None,
                             sample_path: Path | None = None) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(exterior_stop, interior_stop) as backdrop-safe RGB.

    Exterior falls back to measuring the cutout when its name carries no
    color word; interior has no equivalent fallback (there's no interior
    cutout to measure) and defaults to near-black, which is what the
    overwhelming majority of them actually are.
    """
    ext = parse_color_name(exterior)
    if ext is None and sample_path is not None:
        ext = sample_cutout_color(sample_path)
    if ext is None:
        ext = COLOR_WORDS["steel"]

    inr = parse_color_name(interior) or COLOR_WORDS["black"]

    h1, s1, v1 = _to_backdrop(ext)
    h2, s2, v2 = _to_backdrop(inr)
    # Neutrality has to be judged on the SOURCE, not on s1/s2: those are
    # already clamped up into BACKDROP_SATURATION_RANGE, so the stock
    # "black" swatch (a 13%-saturated blue-black) came out at 0.28 and
    # was mistaken for a real color -- which sent black-on-black down the
    # rotate-hue path, where the shift is invisible at near-black value.
    src_neutral = max(colorsys.rgb_to_hsv(*[c / 255 for c in ext])[1],
                      colorsys.rgb_to_hsv(*[c / 255 for c in inr])[1]) <= NEUTRAL_SATURATION_CEILING * 1.5

    # A grey-on-black vehicle (very common) lands both stops within a few
    # percent of each other, which renders as a flat field -- push them
    # apart around their midpoint so there's always a visible ramp.
    if abs(v1 - v2) < MIN_STOP_SEPARATION:
        mid = (v1 + v2) / 2
        half = MIN_STOP_SEPARATION / 2
        v1, v2 = (mid + half, mid - half) if v1 >= v2 else (mid - half, mid + half)
        v1 = min(max(v1, BACKDROP_VALUE_MIN), BACKDROP_VALUE_MAX)
        v2 = min(max(v2, BACKDROP_VALUE_MIN), BACKDROP_VALUE_MAX)

    if abs((h1 - h2 + 0.5) % 1.0 - 0.5) < 0.02 and abs(s1 - s2) < 0.05:
        if src_neutral:
            # Tint the LIGHTER stop. A hue is invisible at near-black
            # values, so tinting the dark end of a black-on-black pair
            # does nothing -- measured (40,40,56)->(21,18,26), still
            # effectively monochrome.
            if v1 >= v2:
                h1, s1 = NEUTRAL_TINT_HUE, NEUTRAL_TINT_SATURATION
            else:
                h2, s2 = NEUTRAL_TINT_HUE, NEUTRAL_TINT_SATURATION
        else:
            h2 += MATCHING_HUE_SHIFT

    return _hsv_bytes(h1, s1, v1), _hsv_bytes(h2, s2, v2)


def colors_from_details(vehicle_folder) -> tuple[str | None, str | None]:
    """(exterior, interior) color names from a vehicle folder's
    details.json. Shared by the still and video composers so "where do
    the colors come from" has one answer. Missing/unreadable yields
    (None, None), which vehicle_gradient_colors() handles by measuring
    the paint off a cutout instead."""
    import json
    try:
        data = json.loads((Path(vehicle_folder) / "details.json").read_text())
    except (OSError, ValueError):
        return None, None
    v = data.get("vehicle", data)
    return (v.get("exterior_color_factory") or v.get("exterior_color")), v.get("interior_color")
