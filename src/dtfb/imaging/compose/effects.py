"""Visual effects applied to a car cutout before it's placed on the canvas."""
from __future__ import annotations

from PIL import Image

GLOW_COLORS = {
    "white": (255, 255, 255),
    "blue": (70, 140, 255),
    "gold": (255, 200, 80),
    "red": (255, 70, 70),
}
DEFAULT_GLOW_COLOR = "white"


def resolve_glow_color(name_or_rgb) -> tuple[int, int, int]:
    if isinstance(name_or_rgb, tuple):
        return name_or_rgb
    key = str(name_or_rgb).lower()
    if key not in GLOW_COLORS:
        raise ValueError(f"Unknown glow color {name_or_rgb!r}, pick one of {list(GLOW_COLORS)}")
    return GLOW_COLORS[key]


def make_glow_layer(car: Image.Image, color=DEFAULT_GLOW_COLOR, radius: int = 24,
                     intensity: float = 0.75) -> tuple[Image.Image, int]:
    """
    A soft colored halo behind the car's silhouette: blur the car's own
    alpha channel outward and recolor it. Returns (glow_rgba, pad) where
    pad is how many pixels bigger the glow image is than `car` on every
    side (the blur needs room to spread past the car's original edges) --
    paste it at (car_x - pad, car_y - pad) so it's centered behind the car.
    """
    from PIL import ImageFilter

    rgb = resolve_glow_color(color)
    pad = radius * 2
    padded_size = (car.width + pad * 2, car.height + pad * 2)

    alpha_padded = Image.new("L", padded_size, 0)
    alpha_padded.paste(car.split()[-1], (pad, pad))

    # Blur at quarter scale, then upsample back -- a glow halo is smooth by
    # definition (it's an alpha-channel blur, there's no detail to lose),
    # and a GaussianBlur's cost scales with pixel count, so this is ~16x
    # fewer pixels through the blur for a result indistinguishable at glow
    # radii (>=8px) where this is actually used. Radius is scaled down to
    # match, floored at 1 so a very small glow still blurs at all.
    small_size = (max(1, padded_size[0] // 4), max(1, padded_size[1] // 4))
    small = alpha_padded.resize(small_size, Image.BILINEAR)
    blurred_small = small.filter(ImageFilter.GaussianBlur(max(1, round(radius / 4))))
    blurred = blurred_small.resize(padded_size, Image.BILINEAR)

    # Intensity scaling via a LUT (point()) instead of a numpy float64
    # round-trip -- same clipped multiply, done in C over 256 entries
    # instead of allocating a full padded-size float64 array per glow.
    if intensity < 1.0:
        lut = [min(255, round(v * intensity)) for v in range(256)]
        blurred = blurred.point(lut)

    glow = Image.new("RGBA", padded_size, (*rgb, 0))
    glow.putalpha(blurred)
    return glow, pad


def paste_with_glow(canvas: Image.Image, car: Image.Image, x: int, y: int,
                     color=DEFAULT_GLOW_COLOR, radius: int = 24, intensity: float = 0.75) -> None:
    """Paste `car` onto `canvas` at (x, y) with a glow behind it. Mutates
    canvas in place. Glow is pasted first (so it sits behind), car second."""
    glow, pad = make_glow_layer(car, color=color, radius=radius, intensity=intensity)
    canvas.paste(glow, (x - pad, y - pad), glow)
    canvas.paste(car, (x, y), car)
