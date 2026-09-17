"""
Hero image composition, split into small pieces so each concern (window
detection, background fit/dim, layout geometry, car-cutout effects,
top-level orchestration) can be read and changed independently:

  window.py     -- detect_window(): find a border's transparent car-window
  background.py -- fit_background(), compute_dim_strength(), apply_spotlight()
  layout.py     -- compute_placement(), named layouts (single/corners/quad)
  effects.py    -- paste_with_glow() and other car-cutout treatments
  hero.py        -- compose_hero(): wires the above together

Import from this package (`from imaging.compose import compose_hero`) --
the submodule split is an implementation detail, not part of the public API.
"""
from .background import apply_spotlight, compute_dim_strength, fit_background
from .effects import GLOW_COLORS, make_glow_layer, paste_with_glow, resolve_glow_color
from .hero import compose_hero
from .hero_video import compute_carousel_timing, compute_video_bitrate_kbps, render_hero_video
from .layout import LAYOUTS, compute_placement, corners_layout, quad_layout, single_layout
from .pipeline import compose_interiors, compose_vehicle, compose_wheel_shots
from .spin import order_for_spin, render_spin_video
from .window import detect_window

__all__ = [
    "compose_hero",
    "compose_vehicle",
    "compose_interiors",
    "compose_wheel_shots",
    "detect_window",
    "fit_background", "compute_dim_strength", "apply_spotlight",
    "compute_placement", "LAYOUTS", "single_layout", "corners_layout", "quad_layout",
    "make_glow_layer", "paste_with_glow", "resolve_glow_color", "GLOW_COLORS",
    "render_hero_video", "compute_carousel_timing", "compute_video_bitrate_kbps",
    "render_spin_video", "order_for_spin",
]
