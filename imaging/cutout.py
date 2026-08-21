"""
rembg-based background removal / subject-presence detection for vehicle photos.

Two uses:
  1. Presence signal: does this image contain a large, car-shaped salient
     foreground object? This is a more reliable cross-check than CLIP's
     scene-label score for photos where dealer-branding overlays (banners,
     tiled watermarks) dominate the frame and confuse a holistic label
     score -- rembg's segmentation is spatially grounded, not just a global
     text/image similarity. See classify.py's docstring/notes for the case
     that motivated this (real exterior photos with a bold banner baked
     into the pixels scoring as "marketing" under CLIP alone).
  2. The actual cutout: composite the segmented subject onto a clean
     background for a polished "product photo" look.

Model weights download once on first use and run fully local afterward --
GPU via onnxruntime-gpu if available, else CPU. Default model is
birefnet-general rather than rembg's default u2net: side-by-side on our
test photos it produced visibly sharper edges (wheel spokes, roof rack,
mirror housings) with no soft blur fringe, and a lower ambiguous-alpha
fraction across the board (~0.003 vs ~0.01). Costs ~10s/image on this GPU
vs u2net's near-instant, which is fine for a batch job run in the
background but pass model_name="u2net" if you want the faster/lighter path.

FLEET MASK-QUALITY AUDIT (296 exterior cutouts): three proxy metrics --
boundary-ring "haloing" (near-white/gray semi-transparent fringe pixels,
the signature of background bleeding through the matte), boundary
raggedness (alpha-gradient variance, a jagged/noisy silhouette), and
top-band width irregularity (a proxy for a chopped mirror/antenna) --
flagged their worst ~20 cutouts each for a pixel-level look. All were
false positives on inspection, not real defects: the haloing metric
mostly fired on white/light-colored vehicles, where the body paint itself
sits in the same near-white/bright band the metric was watching for
background bleed -- a real Maverick/Transit/Jeep edge zoomed to 3x showed
a crisp cut straight onto magenta, no fringe. The width-irregularity
metric mostly fired on Bronco Sport roof antennas, which turned out to be
correctly preserved as a single unbroken thin line, not chopped -- that's
what a genuine thin protrusion looks like to a row-width profile, not a
defect. Also spot-checked a black-on-black bumper (Challenger) for the
"dark vehicle silhouette gets lost against shadow" concern -- boundary
was crisp there too. Consistent with the angle-classification and wheel-
selection audits: the pipeline's actual output is cleaner than a first
threshold-based pass over it suggests. No action taken.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

DEFAULT_MODEL = "birefnet-general"

_sessions = {}
_cuda_libs_loaded = False


def _preload_cuda_libs():
    """onnxruntime-gpu's CUDA provider dlopens libcublasLt.so.13 /
    libcublas.so.13 / libcudart.so.13 by soname, but this machine has no
    system-wide CUDA 13 -- those libraries only exist inside torch's
    bundled runtime (site-packages/nvidia/cu13/lib/), which isn't on the
    dynamic linker's search path. Without this, provider load fails and
    onnxruntime silently falls back to CPU (~10s/image for birefnet vs
    ~1s on GPU). Loading each needed library once with RTLD_GLOBAL puts
    its soname in the process's link map, so the provider's dlopen
    resolves against the already-loaded copy. Missing files are skipped
    silently: on a machine with a real system CUDA (or none at all),
    onnxruntime's own loading behaves as it always did."""
    global _cuda_libs_loaded
    if _cuda_libs_loaded:
        return
    _cuda_libs_loaded = True

    import ctypes
    from pathlib import Path

    import nvidia  # torch's bundled CUDA runtime namespace package

    nvidia_root = Path(list(nvidia.__path__)[0])
    for rel in ("cu13/lib/libcudart.so.13",
                 "cu13/lib/libcublasLt.so.13",
                 "cu13/lib/libcublas.so.13",
                 "cu13/lib/libcufft.so.12",
                 "cu13/lib/libcurand.so.10",
                 "cudnn/lib/libcudnn.so.9"):
        lib = nvidia_root / rel
        if lib.exists():
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _get_session(model_name: str):
    if model_name not in _sessions:
        _preload_cuda_libs()
        from rembg import new_session
        _sessions[model_name] = new_session(model_name)
    return _sessions[model_name]


MAX_AMBIGUOUS_FRACTION = 0.05  # above this, treat the segmentation as unreliable

# How close the foreground bbox can sit to an edge of the ORIGINAL
# (pre-crop) frame before that side counts as "touching" it. Below, how
# many sides touching flags a tight detail crop rather than a whole-vehicle
# photo. Calibrated against a real 129-photo sample of currently-accepted
# exterior cutouts across all 17 vehicles: 121 touched 0 edges, 6 touched
# exactly 1 (legitimate -- a real vehicle shot framed a little tight on one
# side, e.g. no headroom above the roofline), and 0 touched exactly 2. The
# only 2 photos that touched 3+ (both landing at exactly 3) were the same
# real failure case first calibrated here (a badge/grille close-up that
# slipped through CLIP's "detail" classification) and a second confirmed
# instance of the same bug on a different vehicle -- both genuine partial-
# body close-ups, not whole-car shots. FRAME_FILL_MIN_EDGES=2 sits in the
# untouched gap between the legitimate 1-edge cases and the confirmed-bad
# 3-edge cases, so it catches the bug with zero observed false positives.
FRAME_FILL_MARGIN_THRESHOLD = 0.03
FRAME_FILL_MIN_EDGES = 2


@dataclass
class CutoutResult:
    coverage: float          # fraction of image pixels that are foreground (alpha > threshold)
    bbox: tuple | None       # (left, top, right, bottom) of foreground, or None if empty
    cutout: Image.Image      # RGBA image, background removed
    quality_ok: bool         # False if the mask looks unreliable OR like a tight detail crop (see below)
    ambiguous_fraction: float
    fills_frame: bool        # True if the foreground touches (near enough) all four edges


def _margins(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    w, h = size
    return (left / w, top / h, (w - right) / w, (h - bottom) / h)


def remove_background(content: bytes, alpha_threshold: int = 16, model_name: str = DEFAULT_MODEL) -> CutoutResult:
    """
    Tight close-up/detail shots (e.g. a grille filling the whole frame) give
    the segmentation model nothing to key a foreground/background split off,
    and it responds with a diffuse, low-confidence alpha map instead of
    failing loudly -- visually this looks like a faint "ghost" of the
    subject on an otherwise blank white composite. A clean segmentation is
    bimodal: most pixels are confidently foreground (alpha > 200) or
    confidently background (alpha < 20). We use the fraction of pixels stuck
    in between as a quality signal (measured on u2net: <2% on real clean
    cutouts, ~18% on the known failure case) and flag anything above
    MAX_AMBIGUOUS_FRACTION as unreliable, so callers can fall back to the
    original photo instead of shipping a blank-looking image.

    That ambiguous-fraction check only catches segmentation *failures*
    (diffuse/low-confidence masks) -- it does NOT tell you whether the
    photo is a whole-vehicle shot worth cutting out at all. A tight
    headlight/wheel/badge close-up segments just as CONFIDENTLY as a full
    car (same high contrast against the studio background either way), so a
    clean-looking segmentation can still be the wrong thing to cut out.
    "Is this a detail shot" is primarily CLIP's job (see pipeline.py /
    classify.py's "detail" label) -- but CLIP can miss a borderline crop
    (confirmed real cases: a wheel/fender close-up, and separately a
    grille/badge close-up on two different vehicles, all classified
    "exterior" and segmented as technically-clean cutouts). fills_frame is
    a second, independent check on exactly that geometry, folded into
    quality_ok as a backstop for whenever CLIP's label alone isn't enough.
    Originally required the subject to touch ALL four edges (the pure
    zoomed-in-close signature), but that missed a real variant: a crop cut
    off asymmetrically on 2-3 sides while leaving real margin on the
    other(s) -- same tight-crop problem, different framing. Counting edges
    touched (>= FRAME_FILL_MIN_EDGES) rather than requiring all 4 catches
    both without over-triggering -- see the constant's own comment for the
    calibration data.
    """
    import numpy as np
    from rembg import remove

    img = Image.open(io.BytesIO(content)).convert("RGB")
    result = remove(img, session=_get_session(model_name))  # RGBA
    alpha_arr = np.array(result.split()[-1])

    confident_fg = alpha_arr > 200
    confident_bg = alpha_arr < 20
    ambiguous_fraction = 1.0 - (confident_fg.sum() + confident_bg.sum()) / alpha_arr.size

    mask = alpha_arr > alpha_threshold
    bbox = Image.fromarray(mask).getbbox()
    coverage = float(mask.sum()) / mask.size if bbox is not None else 0.0
    fills_frame = bbox is not None and sum(
        1 for m in _margins(bbox, img.size) if m < FRAME_FILL_MARGIN_THRESHOLD
    ) >= FRAME_FILL_MIN_EDGES
    quality_ok = bbox is not None and ambiguous_fraction <= MAX_AMBIGUOUS_FRACTION and not fills_frame

    return CutoutResult(
        coverage=coverage, bbox=bbox, cutout=result,
        quality_ok=quality_ok, ambiguous_fraction=float(ambiguous_fraction),
        fills_frame=fills_frame,
    )


def composite_on_white(cutout: Image.Image) -> Image.Image:
    bg = Image.new("RGB", cutout.size, (255, 255, 255))
    bg.paste(cutout, mask=cutout.split()[-1])
    return bg
