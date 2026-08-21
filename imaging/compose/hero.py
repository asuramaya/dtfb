"""
Hero image composition: background + car cutout(s) + branded border frame.

Layer order (bottom to top): background (scaled/cropped to fill the
canvas, adaptively dimmed toward the edges) -> car cutout(s) (optionally
glowing), scaled and positioned per the chosen layout -> border (its own
alpha handles the window automatically, so it just sits on top without
extra masking logic).

See imaging/assets.py for how borders/backgrounds are looked up by
name/tag rather than hardcoded paths, imaging/select.py for picking which
angle-classified cutout goes in which layout slot, and layout.py/
background.py/effects.py in this package for the pieces this wires
together.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .window import detect_window, alpha_mask, resolve_collision
from .background import fit_background, compute_dim_strength, apply_spotlight
from .layout import LAYOUTS, conveyor_for_window, compute_placement
from .effects import paste_with_glow, DEFAULT_GLOW_COLOR


DEFAULT_CANVAS_SIZE = (1254, 1254)


def compose_hero(background_path, border_path: Path | None, car_paths: list[Path],
                  layout: str = "single", spotlight: bool = True,
                  glow: bool = False, glow_color=DEFAULT_GLOW_COLOR,
                  glow_radius: int = 24, glow_intensity: float = 0.75,
                  margin_frac: float = 0.06,
                  canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE) -> Image.Image:
    """
    car_paths[0] is always the hero (drives the spotlight measurement and
    center). car_paths[1:] fill whatever accent slots `layout` defines --
    see imaging/compose/layout.py for what each layout name expects (e.g.
    "quad" wants [hero, front, back, side]). Extra car_paths beyond what
    the layout has slots for are silently unused; fewer than the layout
    expects just leaves those slots empty.

    background_path may be a path OR an already-built PIL Image -- the
    generated gradients (background.py::gradient_background()) are made
    per-image and never hit disk, so there'd be nothing to pass a path to.

    border_path may be None for a FRAMELESS composition. The border
    normally does three jobs at once: it defines the window the cars are
    laid out in, it supplies the alpha mask placements are collision-
    checked against, and it's the top layer. With no border, the window
    becomes the whole canvas (hence canvas_size, which the border's own
    dimensions would otherwise have decided) and there is nothing to
    collide with, so that check is skipped rather than run against an
    empty mask.

    margin_frac is the breathing room left inside each layout box. The
    0.06 default suits layouts that must also look right half-empty; the
    "conveyor" layout is tuned around a tighter one (see its docstring),
    since all its slots are always filled.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}, pick one of {list(LAYOUTS)}")

    border = Image.open(border_path).convert("RGBA") if border_path is not None else None
    if border is not None:
        canvas_size = border.size
        window = detect_window(border)
    else:
        window = (0, 0, canvas_size[0], canvas_size[1])

    bg = background_path if isinstance(background_path, Image.Image) else Image.open(background_path)
    canvas = fit_background(bg.convert("RGB"), canvas_size)

    cars = [Image.open(p).convert("RGBA") for p in car_paths]
    # Same rule as the video: "conveyor" names an arrangement, not one
    # function, so a non-square canvas gets its slots stacked rather than
    # side by side (layout.py::conveyor_for_window).
    layout_fn = conveyor_for_window(window) if layout == "conveyor" else LAYOUTS[layout]
    boxes = layout_fn(window, len(cars) - 1)
    placements = [compute_placement(car, box, anchor=anchor, margin_frac=margin_frac)
                  for car, (box, anchor) in zip(cars, boxes)]

    # Layout boxes are geometric approximations of one rectangle
    # (detect_window()); real border art often isn't a clean rectangle (a
    # badge/logo hanging lower in the middle than at the edges, say), so
    # check placements against the border's actual alpha channel and nudge
    # down anything that collides. Dynamic on purpose -- works for any
    # border shape or car silhouette without per-vehicle/per-border tuning.
    # Frameless: nothing to collide with, so the nudge is skipped entirely
    # rather than run against an all-transparent mask (which would be a
    # no-op, but only by accident -- resolve_collision() is documented as
    # clearing border ART, and there is none).
    if border is not None:
        border_mask = alpha_mask(border)
        placements = [
            (x, resolve_collision(border_mask, resized, x, y), resized)
            for x, y, resized in placements
        ]

    if spotlight and placements:
        # Hero (index 0) drives the contrast measurement and spotlight center.
        x, y, resized = placements[0]
        car_box = (x, y, x + resized.width, y + resized.height)
        bg_region = canvas.crop(car_box)
        dim_strength = compute_dim_strength(bg_region, resized)
        center = ((car_box[0] + car_box[2]) / 2, (car_box[1] + car_box[3]) / 2)
        canvas = apply_spotlight(canvas, center, dim_strength)

    canvas = canvas.convert("RGBA")
    # Accents first, hero last (on top) -- the hero should never be visually
    # obstructed by a supporting angle shot if their boxes overlap.
    for x, y, resized in reversed(placements):
        if glow:
            paste_with_glow(canvas, resized, x, y, color=glow_color,
                             radius=glow_radius, intensity=glow_intensity)
        else:
            canvas.paste(resized, (x, y), resized)

    if border is not None:
        canvas.alpha_composite(border)
    return canvas.convert("RGB")
