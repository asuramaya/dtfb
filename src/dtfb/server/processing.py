"""
One image, start to finish: fetch -> classify -> cutout -> composite.
Shared by the sync single-segment endpoint and the async batch worker so
there is exactly one place that decides what "process an image" means.

This is the CarCutter-API-COMPATIBLE surface's actual value-add -- the
request/response shapes exist to make switching cheap, but this module is
what does the work, using the same imaging pipeline as the CLI (evaluate_
photo, remove_background, compose_hero). No separate "server" model or
weights; server mode is a different front door onto the same pipeline.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image

CUT_TYPES = ("complete", "normal", "blur", "none")
DEFAULT_CANVAS_SIZE = (1254, 1254)


@dataclass
class ProcessedImage:
    """Result of one process_image() call. `output_path` is None when
    cut_type="none" (classification only, nothing composited) or when
    cutout/composition wasn't possible (e.g. cutout_eligible=False, or
    rembg's own quality gate rejected it) -- category/warning still tell
    the caller why, same "not an error, just nothing to compose" pattern
    the CLI pipeline uses throughout."""
    category: str
    output_path: Path | None = None
    warning: str | None = None
    clip_label: str = ""
    clip_confidence: float = 0.0


def fetch_image(url: str, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def process_image(content: bytes, out_dir: Path, out_name: str,
                   cut_type: str = "complete", scene_id: str | None = None,
                   classifier=None, tiebreak_classifier=None,
                   canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE) -> ProcessedImage:
    """Classify `content`, and if cut_type calls for it, cut out the
    subject and composite it onto a background (a named asset via
    scene_id, or a color measured off the cutout itself when scene_id is
    unset -- see imaging/palette.py::vehicle_gradient_colors, the same
    fallback the CLI pipeline uses when a vehicle has no scraped color).

    cut_type: "complete"/"normal" -> cutout + composite (no behavioral
    difference between them yet -- CarCutter's own docs don't spell out
    the distinction either; accepted for compatibility, both currently
    map to the same treatment). "blur" -> accepted, not yet implemented
    (falls back to "complete"; a background-blur treatment is a real gap,
    tracked as a follow-up, not silently wrong -- see the server's own
    module docstring). "none" -> classify only, no cutout/composite.
    """
    from ..imaging.pipeline import evaluate_photo

    verdict = evaluate_photo(content, classifier, tiebreak_classifier) if classifier is not None else None
    category = verdict.category if verdict is not None else "unknown"
    clip_label = verdict.clip_label if verdict is not None else ""
    clip_confidence = verdict.clip_confidence if verdict is not None else 0.0

    if cut_type == "none":
        return ProcessedImage(category, None, None, clip_label, clip_confidence)

    if verdict is not None and not verdict.cutout_eligible:
        return ProcessedImage(category, None, f"not cutout-eligible (classified {category})",
                               clip_label, clip_confidence)

    from ..imaging.cutout import remove_background
    cutout = remove_background(content)
    if cutout.bbox is None or not cutout.quality_ok:
        return ProcessedImage(category, None, "cutout quality gate rejected this image",
                               clip_label, clip_confidence)

    cropped = cutout.cutout.crop(cutout.bbox)

    out_dir.mkdir(parents=True, exist_ok=True)
    cutout_path = out_dir / f"{out_name}.cutout.png"
    cropped.save(cutout_path)

    from ..imaging import assets
    from ..imaging.compose import compose_hero
    from ..imaging.palette import vehicle_gradient_colors

    background_path = None
    if scene_id:
        try:
            background_path = assets.resolve_arg("backgrounds", scene_id)
        except ValueError:
            pass  # unknown scene_id -- fall through to the measured gradient, don't fail the request over it
    if background_path is None:
        from ..imaging.compose.background import make_linear_gradient
        import random
        start, end = vehicle_gradient_colors(None, None, sample_path=cutout_path)
        background_path = make_linear_gradient(canvas_size, random.uniform(0, 360), start, end).convert("RGBA")

    composed = compose_hero(background_path, None, [cutout_path], layout="single", canvas_size=canvas_size)
    output_path = out_dir / f"{out_name}.composed.png"
    composed.save(output_path)

    return ProcessedImage(category, output_path, None, clip_label, clip_confidence)
