"""
A rotating "spin" video from a vehicle's own real angle-labeled cutouts --
the used-inventory-appropriate alternative to a licensed generic 3D CAD
model (right for new inventory, wrong for used: it shows the SKU, not the
actual physical car with its actual wear).

WHAT THIS IS NOT: true 3D reconstruction (photogrammetry / NeRF / Gaussian
Splatting) of the real vehicle. That's confirmed research-stage industry-
wide as of 2025-2026 (see decision 1449fdef, lotstretcher project graph) -- even
the field's own state of the art needed a purpose-built moving-vehicle
capture rig, not reprocessing of static lot photos. Nobody ships that at
scale yet, lotstretcher included.

WHAT THIS IS: the same class of trick CarCutter's own documented "NextGen
360" actually uses -- synthesize a smooth rotation from a HANDFUL of real
photos rather than requiring a turntable shoot. CarCutter's version works
from 4 photos (front/rear/left/right) with a trained view-interpolation
model. This version is classical, not learned: cross-dissolve between the
real angle-labeled cutouts already sitting in a vehicle's cutout gallery
(lotstretcher typically has 5 distinct angles per vehicle -- front, front_3q,
side, rear_3q, rear -- MORE real anchor frames than CarCutter's own
baseline), scaled/anchored to a consistent canvas position so the car
doesn't visibly jump or resize between shots. It reads as a rotation, not
a slideshow, but it is honest about being an interpolation between real
photos rather than a synthesized new viewpoint -- there is no attempt to
paint in geometry the source photos don't show.

SCOPE: one side's half-turn (front through rear along whichever side has
the fuller angle coverage), not a full 360 -- completing the other side
needs real photos of it, which isn't guaranteed per vehicle. Extending to
a full loop (this side, then a hard cut/dissolve to the mirrored return)
is a reasonable next step once this baseline is validated against real
output.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

from .background import apply_spotlight, make_linear_gradient

# front-to-rear geometric order, NOT hero_video.py's CAROUSEL_ANGLE_ORDER
# (which front-loads front_3q for a dramatic reveal) -- a spin needs to
# read as continuous rotation, so it has to be monotonic.
SPIN_ANGLE_ORDER = ["front", "front_3q", "side", "rear_3q", "rear"]

DEFAULT_CANVAS_SIZE = (1254, 1254)
DEFAULT_FPS = 30.0
HOLD_SECONDS = 0.9      # how long each REAL anchor frame is held
TRANSITION_SECONDS = 0.6  # cross-dissolve duration between anchors
MIN_ANCHORS = 3          # fewer real angles than this isn't a "spin"


def order_for_spin(cutout_dir: Path) -> list[tuple[Path, str]]:
    """Best (highest-confidence) real cutout per angle, in geometric
    front-to-rear order. Angles the vehicle doesn't have a shot for are
    just absent -- interpolating between whatever IS there is still
    honest; inventing a missing anchor would not be."""
    from ..select import load_angles

    cutout_dir = Path(cutout_dir)
    angles = load_angles(cutout_dir)
    if not angles:
        return []

    best: dict[str, tuple[str, float]] = {}
    for filename, info in angles.items():
        label, conf = info["angle"], info["confidence"]
        if label not in best or conf > best[label][1]:
            best[label] = (filename, conf)

    return [(cutout_dir / best[label][0], label) for label in SPIN_ANGLE_ORDER if label in best]


def _fit_and_anchor(cutout: Image.Image, canvas_size: tuple[int, int],
                     max_height_frac: float = 0.62) -> Image.Image:
    """Scale `cutout` to a consistent height and anchor it to a fixed
    ground line, centered horizontally. Without this, cutting between
    angles at their own native crop sizes makes the car visibly grow/
    shrink and jump vertically frame to frame -- the single biggest thing
    that makes a naive angle-to-angle cut read as a slideshow instead of
    a rotation."""
    cw, ch = canvas_size
    target_h = round(ch * max_height_frac)
    scale = target_h / cutout.height
    resized = cutout.resize((round(cutout.width * scale), target_h), Image.LANCZOS, reducing_gap=2.0)

    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    x = (cw - resized.width) // 2
    ground_y = round(ch * 0.80)  # fixed "ground" line the car's base sits on
    y = ground_y - resized.height
    layer.alpha_composite(resized, (x, y))
    return layer


def _backdrop(canvas_size: tuple[int, int], base_angle: float, start: tuple[int, int, int],
              end: tuple[int, int, int]) -> Image.Image:
    grad = make_linear_gradient(canvas_size, base_angle, start, end).convert("RGBA")
    center = (canvas_size[0] / 2, canvas_size[1] * 0.80)
    return apply_spotlight(grad, center, dim_strength=0.55).convert("RGBA")


def render_spin_video(cutout_dir: Path, out_path: Path, gradient_colors: tuple,
                       canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
                       fps: float = DEFAULT_FPS, budget_mb: float = 50.0,
                       encoder: str = "libx264") -> dict | None:
    """Renders the spin and encodes it to `out_path`. Returns a small
    report dict, or None if there aren't enough real angles to make a
    spin worth producing (see MIN_ANCHORS) -- not an error, just not
    every vehicle's gallery has the coverage for this yet.

    Mirrors hero_video.py's render_hero_video()'s own encoder-fallback
    contract on purpose (retry once with libx264 if a hardware encoder
    fails to open under GPU pressure) but is a SEPARATE, self-contained
    implementation rather than a shared call -- duplicates ~20 lines of
    subprocess/fallback plumbing in exchange for not touching the
    already-shipped, already-tested hero_video.py mid-build. Worth
    unifying later if a third caller needs the same pattern.
    """
    anchors = order_for_spin(cutout_dir)
    if len(anchors) < MIN_ANCHORS:
        return None

    cutouts = [Image.open(p).convert("RGBA") for p, _label in anchors]
    frames_static = [_fit_and_anchor(c, canvas_size) for c in cutouts]

    base_angle, start, end = gradient_colors
    backdrop = _backdrop(canvas_size, base_angle, start, end)

    hold_frames = round(HOLD_SECONDS * fps)
    trans_frames = round(TRANSITION_SECONDS * fps)
    n = len(frames_static)
    total_frames = n * hold_frames + (n - 1) * trans_frames
    total_seconds = total_frames / fps

    bitrate_kbps = round((budget_mb * 8192) / total_seconds * 0.90)  # 10% container-overhead margin

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".ffmpeg.log")

    def _frame_at(f: int) -> Image.Image:
        cycle = hold_frames + trans_frames
        idx, within = divmod(f, cycle)
        if idx >= n - 1 or within < hold_frames:
            idx = min(idx, n - 1)
            layer = frames_static[idx]
        else:
            tau = (within - hold_frames) / trans_frames
            layer = Image.blend(frames_static[idx], frames_static[idx + 1], tau)
        canvas = backdrop.copy()
        canvas.alpha_composite(layer)
        return canvas.convert("RGB")

    def _run_encode(enc: str) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{canvas_size[0]}x{canvas_size[1]}",
            "-r", str(fps), "-i", "pipe:0",
            "-c:v", enc, "-preset", "medium", "-b:v", f"{bitrate_kbps}k",
            "-maxrate", f"{int(bitrate_kbps * 1.2)}k", "-bufsize", f"{int(bitrate_kbps * 2)}k",
            "-pix_fmt", "yuv420p", "-t", f"{total_seconds:.3f}", str(out_path),
        ]
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logf)
            try:
                for f in range(total_frames):
                    try:
                        proc.stdin.write(_frame_at(f).tobytes())
                    except BrokenPipeError:
                        break
            finally:
                proc.stdin.close()
                ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (exit {ret}) -- see {log_path}")

    enc_used = encoder
    try:
        _run_encode(encoder)
    except RuntimeError:
        if encoder == "libx264":
            raise
        print(f"    ! {encoder} failed for spin video, retrying with libx264", file=sys.stderr)
        enc_used = "libx264"
        _run_encode("libx264")

    log_path.unlink(missing_ok=True)
    return {
        "out_path": str(out_path),
        "encoder": enc_used,
        "duration_s": round(total_seconds, 2),
        "n_anchors": n,
        "angles": [label for _p, label in anchors],
        "file_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
    }
