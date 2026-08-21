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
    import numpy as np

    rgb = resolve_glow_color(color)
    pad = radius * 2
    padded_size = (car.width + pad * 2, car.height + pad * 2)

    alpha_padded = Image.new("L", padded_size, 0)
    alpha_padded.paste(car.split()[-1], (pad, pad))
    blurred = alpha_padded.filter(ImageFilter.GaussianBlur(radius))

    arr = np.asarray(blurred, dtype=np.float64) * intensity
    blurred = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="L")

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
