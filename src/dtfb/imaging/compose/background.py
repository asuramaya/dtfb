"""Background fitting, generated gradient backdrops, and the adaptive
spotlight/vignette dim."""
from __future__ import annotations

import colorsys
import random

from PIL import Image

# Hue bands the generated gradients sample from, as (lo, hi) in HSV hue
# units (may exceed 1.0 and wrap). Deliberately NOT the full hue circle:
# unconstrained hue lands on yellow-greens and mustards often enough to
# look like a bug rather than a choice, and every one of these has to be
# postable without a human checking it first.
GRADIENT_BUILD_MAX = 320  # see make_linear_gradient()

_SPOTLIGHT_CACHE: dict = {}
_SPOTLIGHT_CACHE_MAX = 16

GRADIENT_HUE_BANDS = [
    (0.55, 0.70),   # blue -> indigo
    (0.45, 0.55),   # teal -> cyan
    (0.92, 1.04),   # crimson -> red
    (0.74, 0.86),   # violet -> magenta
    (0.02, 0.08),   # rust -> warm orange
    (0.58, 0.62),   # steel blue (narrow, close to the dealer's own blue)
]


def _hsv_bytes(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return round(r * 255), round(g * 255), round(b * 255)


def gradient_spec(seed) -> dict:
    """A reproducible (angle, two colors) gradient description for `seed`.

    Seeded rather than free-random on purpose: the same vehicle+image
    always regenerates the same backdrop, so a rerun is a no-op and a
    reported problem can actually be reproduced. Variety across posts --
    which is the point, since a shared background is what got flagged --
    comes from the seed differing per image, not from the clock.

    Both stops stay dark-to-midtone (value <= ~0.72). These sit behind
    cutouts that are overwhelmingly white/silver/grey, so a light
    backdrop would flatten the vehicle against it; the existing spotlight
    pass then dims further around the car (see compute_dim_strength()).
    """
    rng = random.Random(seed)
    lo, hi = rng.choice(GRADIENT_HUE_BANDS)
    h1 = rng.uniform(lo, hi)
    h2 = h1 + rng.uniform(-0.06, 0.06)
    dark = _hsv_bytes(h1, rng.uniform(0.35, 0.72), rng.uniform(0.10, 0.22))
    light = _hsv_bytes(h2, rng.uniform(0.40, 0.80), rng.uniform(0.45, 0.72))
    # Which end is dark is itself randomized -- otherwise every post is
    # lit from the same corner and they still read as one template.
    if rng.random() < 0.5:
        dark, light = light, dark
    return {"angle": rng.uniform(0, 360), "start": dark, "end": light}


def make_linear_gradient(size: tuple[int, int], angle: float,
                          start: tuple[int, int, int], end: tuple[int, int, int]) -> Image.Image:
    """A linear gradient across `size` at `angle` degrees (0 = left-to-
    right, increasing counter-clockwise), interpolated start -> end.

    Computed by projecting every pixel onto the angle's direction vector
    and normalizing that projection over its own min/max, so the full
    colour range lands corner-to-corner at ANY angle -- projecting onto a
    fixed axis instead would compress the ramp into the middle of the
    image on diagonals and leave flat bands in the corners.
    """
    import numpy as np

    # Built at GRADIENT_BUILD_MAX and upscaled. A linear ramp is smooth by
    # definition, so there is no detail to lose, and the full-size version
    # cost 57.6ms/frame -- the single biggest per-frame cost after the
    # spotlight, for pixels that are exactly reconstructible by bilinear
    # interpolation.
    out_size = size
    scale = max(size) / GRADIENT_BUILD_MAX
    if scale > 1:
        size = (max(2, round(size[0] / scale)), max(2, round(size[1] / scale)))

    w, h = size
    theta = np.radians(angle)
    dx, dy = np.cos(theta), -np.sin(theta)  # -y so positive angles read counter-clockwise on screen

    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    proj = (xs * dx + ys * dy).astype(np.float32)
    span = proj.max() - proj.min()
    t = (proj - proj.min()) / (span if span else 1.0)

    a = np.asarray(start, dtype=np.float32).reshape(1, 1, 3)
    b = np.asarray(end, dtype=np.float32).reshape(1, 1, 3)
    arr = a + (b - a) * t[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")
    return img if img.size == out_size else img.resize(out_size, Image.BILINEAR)


def gradient_background(size: tuple[int, int], seed) -> Image.Image:
    spec = gradient_spec(seed)
    return make_linear_gradient(size, spec["angle"], spec["start"], spec["end"])


def vehicle_gradient_background(size: tuple[int, int], seed,
                                 exterior: str | None, interior: str | None,
                                 sample_path=None) -> Image.Image:
    """A gradient built from THIS vehicle's own exterior/interior colors
    (see imaging/palette.py) at a seeded random angle.

    Only the angle and the stop order are random here -- the colors are
    the vehicle's, so the backdrop reinforces the car instead of being
    arbitrary decoration, while still differing post to post."""
    from ..palette import vehicle_gradient_colors

    start, end = vehicle_gradient_colors(exterior, interior, sample_path)
    rng = random.Random(seed)
    angle = rng.uniform(0, 360)
    if rng.random() < 0.5:
        start, end = end, start
    return make_linear_gradient(size, angle, start, end)


def fit_background(bg: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
    """Cover-fit: scale to fill canvas_size (cropping overflow), like CSS
    background-size: cover. Stock background photos are typically much
    larger than the border canvas, so this usually scales down."""
    cw, ch = canvas_size
    bw, bh = bg.size
    scale = max(cw / bw, ch / bh)
    new_size = (round(bw * scale), round(bh * scale))
    resized = bg.resize(new_size, Image.LANCZOS, reducing_gap=2.0)
    left = (resized.width - cw) // 2
    top = (resized.height - ch) // 2
    return resized.crop((left, top, left + cw, top + ch))


def _luminance(rgb) -> "np.ndarray":
    """Perceptual (ITU-R BT.601) luminance per pixel from an HxWx3 uint8 array."""
    import numpy as np
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def compute_dim_strength(bg_region: Image.Image, car: Image.Image, margin: float = 35.0,
                          min_dim: float = 0.35, max_dim: float = 1.0) -> float:
    """
    How hard to dim the background so the car reads as brighter/more
    prominent, computed from the actual measured contrast rather than a
    fixed constant -- a silver car on an already-dark background needs
    little or no help; a gray car on a bright gray background needs a lot.

    bg_region should be the patch of background that will sit directly
    behind the car (not the whole canvas) -- that's the contrast that
    actually matters for the car to visually separate from what's around it.

    Method: compare alpha-weighted mean luminance of the car's opaque
    pixels against the mean luminance of that background patch. If the
    background is already `margin` levels darker than the car, no dimming
    is needed (returns 1.0, i.e. no change). Otherwise, compute the
    multiplicative factor that would need to be applied to the background
    to open up that same margin, clamped to [min_dim, max_dim] so we never
    blow the background out to black or fail to dim at all.
    """
    import numpy as np

    car_arr = np.asarray(car.convert("RGBA"), dtype=np.float64)
    alpha = car_arr[..., 3] / 255.0
    if alpha.sum() < 1:
        return 1.0
    car_lum = float((_luminance(car_arr[..., :3]) * alpha).sum() / alpha.sum())

    bg_lum = float(_luminance(np.asarray(bg_region.convert("RGB"))).mean())

    target_bg_lum = car_lum - margin
    if bg_lum <= target_bg_lum or bg_lum <= 0:
        return 1.0  # background is already darker than the car by enough margin

    dim = target_bg_lum / bg_lum if target_bg_lum > 0 else min_dim
    return float(np.clip(dim, min_dim, max_dim))


def apply_spotlight(canvas: Image.Image, center: tuple[float, float], dim_strength: float,
                     inner_radius_frac: float = 0.28, outer_radius_frac: float = 0.85) -> Image.Image:
    """
    Radial "spotlight" dim: full brightness within inner_radius of `center`,
    fading to `dim_strength` brightness by outer_radius, flat beyond that.
    Reads as light falling on the car from directly above/behind it rather
    than a flat brightness filter over the whole frame -- the car pops
    because the ground around it visibly darkens, not because of a uniform
    tint.

    Radius fractions are relative to the distance from `center` to the
    canvas's farthest corner (not canvas width/height/diagonal directly --
    those measure the wrong thing when center isn't at the canvas center).
    """
    import numpy as np

    if dim_strength >= 1.0:
        return canvas  # already enough natural contrast, nothing to do

    factor = _spotlight_factor(canvas.size, center, dim_strength, inner_radius_frac, outer_radius_frac)
    arr = np.asarray(canvas.convert("RGB"), dtype=np.float32) * factor[..., None]
    dimmed = Image.fromarray(arr.astype("uint8"), mode="RGB")
    return dimmed.convert(canvas.mode) if canvas.mode != "RGB" else dimmed


def _spotlight_factor(size, center, dim_strength, inner_radius_frac, outer_radius_frac):
    """The radial falloff multiplier, memoized.

    This is a pure function of (canvas size, center, dim strength) -- all
    of which are fixed for as long as one shot holds the hero slot -- but
    it was being rebuilt for every single frame, mgrid and sqrt over 1.5M
    pixels in float64, at 87.9ms a frame. A whole video needs only as
    many distinct masks as it has shots. Keyed on rounded values so
    floating-point jitter in the center can't defeat the cache; bounded
    because each entry is ~6MB.
    """
    import numpy as np

    key = (size, round(center[0]), round(center[1]), round(dim_strength, 3),
           inner_radius_frac, outer_radius_frac)
    hit = _SPOTLIGHT_CACHE.get(key)
    if hit is not None:
        return hit

    w, h = size
    cx, cy = center
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    max_dist = max(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in corners)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max_dist

    t = np.clip((dist - inner_radius_frac) / (outer_radius_frac - inner_radius_frac), 0, 1)
    smooth = t * t * (3 - 2 * t)  # smoothstep, avoids a visible hard-edged ring
    factor = (1.0 - smooth * (1.0 - dim_strength)).astype(np.float32)

    if len(_SPOTLIGHT_CACHE) >= _SPOTLIGHT_CACHE_MAX:
        _SPOTLIGHT_CACHE.pop(next(iter(_SPOTLIGHT_CACHE)))
    _SPOTLIGHT_CACHE[key] = factor
    return factor
