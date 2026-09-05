"""
Animated version of compose_hero(): a looping video background (e.g. a
waving American flag) instead of a still, with a 3-slot CONVEYOR --
left accent, hero, right accent -- that continuously rotates through the
vehicle's whole shot library, timed to the beat of a looping audio track.

THE CONVEYOR: at any moment the 3 visible shots are simply 3 consecutive
entries of the (already front-to-rear ordered) shot list: left=shots[i-1],
hero=shots[i], right=shots[i+1]. Every bar, i advances by 1 and the whole
window slides. What makes this read as one continuous cycle instead of 3
independently-cutting slots is that each shot's OWN journey through the 3
slots is animated as a single throughline instead of 3 unrelated
crossfades:
  - The outgoing hero (shots[i]) doesn't just disappear -- it shrinks and
    moves into the left slot, becoming the new left accent.
  - The old left accent (shots[i-1]) fades out in place -- its journey
    ends here, it's been hero and left already.
  - The current right accent (shots[i+1]) grows and moves down into the
    hero slot, becoming the new hero.
  - A brand new shot (shots[i+2]) fades in at the right slot -- entering
    the conveyor for the first time.
Every shot's lifecycle is therefore: fade in at right -> grow into hero ->
shrink into left -> fade out. Confirmed this needed to be a real
position+scale morph, not independent alpha crossfades per slot -- an
earlier version faded new accent content in from nowhere (no visible
relationship to what was just on screen) and it read as random/jarring
rather than a cycle.

PAN, INTEGRATED WITH THE CONVEYOR: a shot whose native aspect is much
wider than the hero box (a side profile, mainly) fills the box's full
height and drives across it, entering and leaving past the frame edges.

The key is that EVERY appearance of a shot -- accent slot, static hero,
panning hero, and every intermediate frame of a morph -- is ONE rect:
where the whole, undistorted image is drawn. Nothing is ever clipped to a
sub-frame window. A pan is then just "draw wider than the frame and slide
it sideways," an accent is the same image drawn small, and the morph
between them is a plain interpolation of that single rect -- from the
exact frame steady playback left off at, to the exact frame it resumes
at, so there's no cut, no dissolve, and no pop at either boundary. The
vehicle is always whole; the frame's own edge (supplied by the border art
composited on top) is the only thing that ever hides part of it, which
reads as driving off screen.

Four earlier attempts are worth not repeating, all of them variations on
treating a pan as "an image inside a window":
  1. Interpolating the pan's window rect straight to an accent rect
     force-resized the image into shapes it doesn't have -- visible
     squish.
  2. A separate aspect-correct hero endpoint fixed the squish but landed
     the morph somewhere other than where the pan actually sits, so the
     hand-off popped.
  3. Cross-dissolving to hide that pop left the pan's edges gapped
     mid-transition.
  4. Carrying a real clip window (correct while the window WAS the frame)
     dragged a hard vertical edge across the vehicle's body the moment
     that window began shrinking toward the accent slot -- a truck cut in
     half in mid-frame.
The fix for all four is the same: there is no window. Draw the whole
thing and let it leave the frame.

Pan direction is set so the vehicle slides the way it FACES, which is
what reads as driving forward -- the cutout is a sprite over a static
flag, so it is the moving object; a right-facing vehicle sliding left
reads as reversing, however natural "reveal the tail then the hood"
sounds in the abstract. Which way any given photo faces is detected, not
assumed -- see imaging/classify.py::detect_hood_side(); confirmed real
need, since this vehicle's own two side shots face opposite ways.

MUSICAL TIMING CONTRACT: the audio file passed in MUST be a clean,
seamlessly-looping BARS_PER_LOOP-bar segment (4 by default) for the bar/
beat math below to land on the actual music -- there's no beat detection
here, just audio_loop_seconds / 4 bars / 4 beats-per-bar. A file that
isn't a clean 4-bar loop will make the conveyor's cuts and the hero's
pump drift off the actual downbeats. This is a real constraint on the
input, not an implementation detail to work around -- see
probe_duration_s().

Rendering strategy:
  - Every shot's placement (resized + collision-resolved position) in
    each of the 3 boxes is precomputed ONCE per shot -- 3 placements x N
    shots, not per frame.
  - During the STEADY portion of each bar (most of it), the 3 layers
    (left/hero/right) are simple pastes at their precomputed positions --
    cheap. Only the beat pulse (hero only) needs a live per-frame resize.
  - During the TRANSITION portion (the last `crossfade` seconds of each
    bar), all 4 moving/fading elements are recomposited live -- 2 morphs
    (resize from the shot's own full-res image at the current interpolated
    box size, not from an already-downscaled version, to avoid
    double-resampling) and 2 alpha fades (cheap: reuse the precomputed
    resting-size image, just scale its alpha channel).
  - The flag clip's own frames are decoded and cover-cropped to the canvas
    size ONCE, then reused via modulo indexing no matter how long the
    final video runs.
Frames are streamed to ffmpeg over a pipe as raw rgb24 rather than written
to disk as an intermediate PNG sequence.
"""
from __future__ import annotations

import math
import statistics
import subprocess
from pathlib import Path

from PIL import Image

from .background import apply_spotlight, compute_dim_strength, fit_background, make_linear_gradient
from .effects import DEFAULT_GLOW_COLOR, make_glow_layer
from .layout import LAYOUTS, conveyor_for_window, compute_placement
from .window import alpha_mask, detect_window, resolve_collision
from ..classify import detect_hood_side

AUDIO_BITRATE_KBPS = 128

# See the module docstring's MUSICAL TIMING CONTRACT -- these aren't tuned
# constants, they're the definition of "a bar" and "a beat" given a clean
# N-bar loop, so changing them means "the input has a different bar/beat
# count," not "make the pacing faster/slower."
BARS_PER_LOOP = 4
BEATS_PER_BAR = 4

MIN_TRANSITION_S = 0.3
MAX_TRANSITION_S = 0.9
TRANSITION_FRAC = 0.35  # fraction of a bar spent on the 4-way conveyor morph


# The beat "pump": a quick scale-up on the beat that snaps back down --
# real punch, not a slow sine wave. 0.035 (3.5%) is deliberately
# conservative: quad_layout()/corners_layout() leave a ~2%-of-window-height
# gap between the hero box and the accent row above it (their own
# `round(wh * 0.02)`), and on this border's real geometry that gap works
# out to ~4% of a typical hero rect's height -- so the pulse's peak
# growth, symmetric around the hero's own center, stays under that gap
# with margin to spare rather than reaching back into the accents, which
# is the exact bug the hero box bounds fix (layout.py) exists to prevent.
PULSE_STRENGTH = 0.035
PULSE_DECAY = 0.3

# A hero cutout whose native aspect ratio is much wider than the hero
# box's -- a side profile, mainly -- would otherwise be letterboxed (fit
# to width, lots of dead space above/below). Past this much extra width
# (at fill-to-height scale) it instead fills the box's full height and
# pans edge-to-edge across during its STEADY (non-transitioning) hold.
MIN_PAN_OVERFLOW_FRAC = 0.20

# Only this angle pans -- see _build_shot()'s note.
PAN_ANGLE_LABEL = "side"

# Fallbacks for the pieces that are now optional. DEFAULT_BPM is not an
# arbitrary pick: the original 4-bar loop measures 9.606s, i.e. 2.4015s a
# bar and 0.6s a beat -- exactly 100 BPM. So a music-less render at the
# default keeps the identical cut and pump cadence the scored one had.
DEFAULT_BPM = 100.0
DEFAULT_FPS = 25.0
DEFAULT_CANVAS_SIZE = (1254, 1254)

# One edit, three frames. The choreography is identical in all of them --
# same shot order, same bar-locked cuts, same pan behaviour, same pump on
# the kick; only the blocking changes, because the blocking is the part
# that depends on the frame's shape (see layout.py::conveyor_for_window).
#
# Budgets are the square's 50MB scaled by pixel count, so every format
# gets the same bits per pixel as the canvas the look was tuned on. The
# 50MB figure is a MARKETPLACE limit and applies only to the square;
# Reels, Shorts and TikTok all allow far more, so inheriting it there
# would have meant paying for a constraint that doesn't exist.
VIDEO_FORMATS = {
    "square": {"canvas": (1254, 1254), "budget_mb": 50.0,
                "note": "Marketplace, Facebook feed"},
    "vertical": {"canvas": (1080, 1920), "budget_mb": 66.0,
                  "note": "Reels, Stories, TikTok, Shorts"},
    "horizontal": {"canvas": (1920, 1080), "budget_mb": 66.0,
                    "note": "YouTube, landscape feed"},
}
DEFAULT_VIDEO_FORMAT = "square"
# Full rotations the generated gradient sweeps through over the whole
# video. The backdrop has to keep moving now that there's no waving flag
# doing it -- a static gradient behind a static accent row reads as a
# still image with a car twitching on it.
GRADIENT_TURNS = 1.0

# The pan runs from "nose centered in the frame" to "tail centered": the
# vehicle drives through, entering and leaving past the frame's edges.
# Most of it is off-frame at each endpoint, which is the point -- it
# reads as a vehicle driving past, not as a cropped photo.
#
# Nothing is ever CLIPPED to a sub-frame window, though, and that
# distinction is the whole ballgame. A vehicle continuing past the
# frame's own edge reads as driving off screen; a vehicle cut by a hard
# vertical edge in the MIDDLE of the frame reads as broken. An earlier
# version clipped the hero to a window rect that shrank toward the accent
# slot during the transition, which dragged exactly such an edge across
# the truck's body -- see _render(), which no longer clips at all.
PAN_BARS = 2

# How much of each layout box is left as breathing room. Much tighter
# than compute_placement()'s 0.06 still-image default: a still is looked
# at, so it can afford air, but a frame that's on screen for two seconds
# wants to read big, and the inset was being paid twice (once around the
# accent boxes, once around the hero) which measurably emptied the frame.
HERO_MARGIN_FRAC = 0.02
ACCENT_MARGIN_FRAC = 0.03


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def _lerp_rect(rect_from: tuple[float, float, float, float], rect_to: tuple[float, float, float, float],
                t: float) -> tuple[float, float, float, float]:
    return tuple(a + (b - a) * t for a, b in zip(rect_from, rect_to))


def probe_duration_s(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def probe_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    num, _, den = out.stdout.strip().partition("/")
    return float(num) / float(den or 1)


def compute_carousel_timing(audio_loop_s: float, bars_per_loop: int = BARS_PER_LOOP) -> tuple[float, float, float]:
    """(dwell, transition_s, beat_s) derived from the audio loop's own
    measured length. dwell (1 bar) is how long each shot spends as hero
    before the conveyor advances; transition_s is how much of that bar is
    spent on the 4-way morph into the next arrangement.

    bars_per_loop is a property of the specific recording, so it comes
    from that asset's manifest entry rather than being assumed -- see
    assets/manifest.json's audio `bars`."""
    dwell = audio_loop_s / bars_per_loop
    beat_s = dwell / BEATS_PER_BAR
    transition_s = min(MAX_TRANSITION_S, max(MIN_TRANSITION_S, dwell * TRANSITION_FRAC))
    transition_s = min(transition_s, dwell * 0.9)
    return dwell, transition_s, beat_s


def _pulse_scale(t: float, beat_s: float) -> float:
    """Snap-and-decay envelope, period beat_s: 1.0+PULSE_STRENGTH right on
    the beat, decaying back toward 1.0 before the next one."""
    if beat_s <= 0:
        return 1.0
    phase = (t % beat_s) / beat_s
    return 1.0 + PULSE_STRENGTH * math.exp(-phase / PULSE_DECAY)


# How far under the stated cap to actually aim. The 3% container
# allowance below is an ESTIMATE of muxing overhead, not a safety margin,
# and treating it as one left almost none: measured across the fleet's 40
# square renders, the largest was 46.50MB against Marketplace's 50MB and
# five were above 46MB. That is a 7% cushion absorbing every source of
# error at once -- a rate-control overshoot, a container that muxes fatter
# than estimated, or a platform that measures the limit differently than
# we do. 10% costs about 0.5dB of bitrate, which is invisible on this
# content, and buys a margin that is actually a margin.
SIZE_SAFETY_FRAC = 0.10


def compute_video_bitrate_kbps(total_seconds: float, budget_mb: float = 50.0,
                                audio_kbps: int = AUDIO_BITRATE_KBPS,
                                container_overhead_frac: float = 0.03,
                                safety_frac: float = SIZE_SAFETY_FRAC,
                                min_kbps: int = 4000, max_kbps: int = 20000) -> int:
    """Video bitrate that fills (but doesn't exceed) budget_mb once audio,
    a container-overhead estimate and a safety margin are accounted for.
    Clamped at the top too -- 1080-ish square H.264 doesn't visibly improve
    much past ~20Mbps, so a short video shouldn't request an absurd
    bitrate just because the budget technically allows it.

    The bottom clamp is what would actually break the budget: at min_kbps
    a video longer than about 105s exceeds 50MB no matter what this
    returns. Our longest pass is a shot per bar at 2.4s, so that needs ~44
    cutouts and cannot happen on real inventory -- but it is the failure
    mode to check first if a file ever comes out over.
    """
    budget_bits = budget_mb * 1024 * 1024 * 8 * (1 - container_overhead_frac) * (1 - safety_frac)
    audio_bits = audio_kbps * 1000 * total_seconds
    video_kbps = (budget_bits - audio_bits) / total_seconds / 1000
    return int(max(min_kbps, min(max_kbps, video_kbps)))


def _trim_dark_edges(frames: list, threshold_frac: float = 0.5) -> list:
    """Drop a leading/trailing run of anomalously dark frames -- confirmed
    real on a real stock flag loop, whose very last frame (of 176) was
    pure black (mean luminance 0.0 against ~130-140 for every neighbor),
    flashing black every ~7s on loop. Only trims from the two ends, never
    the interior."""
    import numpy as np

    brightness = [float(np.asarray(f).mean()) for f in frames]
    med = statistics.median(brightness)
    threshold = med * threshold_frac
    lo, hi = 0, len(frames)
    while lo < hi and brightness[lo] < threshold:
        lo += 1
    while hi > lo and brightness[hi - 1] < threshold:
        hi -= 1
    return frames[lo:hi]


def _read_bg_frames(video_path: Path, canvas_size: tuple[int, int]) -> list[Image.Image]:
    """Decode every frame of the background clip ONCE, drop any broken
    dark frame(s) at the loop seam, then cover-crop each to canvas_size."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    raw = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            raw.append(bgr[:, :, ::-1])
    finally:
        cap.release()
    if not raw:
        raise ValueError(f"Could not decode any frames from {video_path}")

    raw = _trim_dark_edges(raw)
    if not raw:
        raise ValueError(f"All frames in {video_path} were anomalously dark after trimming")
    return [fit_background(Image.fromarray(rgb), canvas_size) for rgb in raw]


class _Shot:
    __slots__ = ("path", "car", "hero_rect", "hero_fit_rect", "left_rect", "right_rect", "center",
                 "dim_strength", "is_pan", "pan_draw_w", "pan_draw_h", "pan_x_start", "pan_x_end",
                 "hood_side", "bars")
    # EVERY on-screen appearance of a shot is a (draw_rect, clip_rect)
    # pair -- see _hero_state()/_render(). draw_rect is where the WHOLE
    # image is drawn (always its own aspect, never distorted); clip_rect
    # is the window you see it through. For the accent slots and a
    # non-pan hero the two are identical (image exactly fills its box, so
    # clipping is a no-op). For a PAN hero they differ: the image is drawn
    # much wider than the box (pan_draw_w) and slid horizontally behind a
    # fixed window (hero_rect). That single representation is what makes
    # the morph work -- see the module docstring's PAN section.
    #
    # hero_rect: the hero window -- for a pan shot the edge-to-edge crop
    #   box (avail_w x avail_h); for a static shot the aspect fit.
    # hero_fit_rect: the plain aspect-preserving fit in the hero box.
    #   Static shots use it as their hero draw+clip; pan shots don't use
    #   it at all (their draw rect comes from the pan geometry below).
    # left_rect/right_rect: aspect-preserving fits in the accent boxes.
    # pan_draw_w/pan_draw_h: the size the whole image is drawn at while
    #   panning (fills the window's height, overflows its width).
    # pan_x_start/pan_x_end: draw-x at pan progress 0 and 1 -- the pan is
    #   a plain lerp between them, no direction flag needed at render
    #   time (the direction is already baked into which is which).
    # hood_side: 'left'/'right' from detect_hood_side(), the thing that
    #   decides those two endpoints; kept for the run report.
    # bars: how many bars this shot holds the hero slot (PAN_BARS for a
    #   pan, 1 otherwise).


def _place(car: Image.Image, box, anchor: str, border_mask,
            margin_frac: float = 0.06) -> tuple[Image.Image, float, float, float, float]:
    """(resized, x, y, w, h) -- collision-resolved resting placement."""
    x, y, resized = compute_placement(car, box, anchor=anchor, margin_frac=margin_frac)
    if border_mask is not None:
        y = resolve_collision(border_mask, resized, x, y)
    return resized, float(x), float(y), float(resized.width), float(resized.height)


def _build_shot(path: Path, hero_box, hero_anchor, left_box, left_anchor, right_box, right_anchor,
                 border_mask, representative_bg: Image.Image, margin_frac: float = HERO_MARGIN_FRAC,
                 pannable: bool = True, precomputed_hood_side: str | None = None) -> _Shot:
    shot = _Shot()
    shot.path = path
    shot.car = Image.open(path).convert("RGBA")

    _, *shot.left_rect = _place(shot.car, left_box, left_anchor, border_mask, ACCENT_MARGIN_FRAC)
    _, *shot.right_rect = _place(shot.car, right_box, right_anchor, border_mask, ACCENT_MARGIN_FRAC)

    hero_fit_resized, *hero_fit_rect = _place(shot.car, hero_box, hero_anchor, border_mask, margin_frac)
    shot.hero_fit_rect = tuple(hero_fit_rect)

    # is_pan needs the hero BOX's own available space, not a shot's
    # already-letterboxed fit result -- computing fill_scale from a
    # width-constrained shot's fit height (which is SMALLER than the box's
    # real avail_h) silently made every wide shot look non-pan-eligible.
    bl, bt, br, bb = hero_box
    avail_w = (br - bl) * (1 - 2 * margin_frac)
    avail_h = (bb - bt) * (1 - 2 * margin_frac)
    width_constrained = (avail_w / shot.car.width) < (avail_h / shot.car.height)
    fill_scale = avail_h / shot.car.height
    filled_w = shot.car.width * fill_scale
    overflow_frac = filled_w / avail_w - 1
    # Aspect alone is not enough to earn a pan: a 3/4 shot of a long
    # vehicle is wide too, and panning one looks like a mistake because
    # the vehicle is receding in perspective rather than lying flat
    # across the frame. Only a dead-on side profile reads as "drives
    # past", so the classifier's label is the gate and the aspect check
    # only decides whether there's room to move.
    shot.is_pan = pannable and width_constrained and overflow_frac >= MIN_PAN_OVERFLOW_FRAC

    if shot.is_pan:
        # The window spans the hero box's FULL width -- no side margin.
        # Every other shot/slot insets by margin_frac for breathing room,
        # but on a pan that inset just reads as the vehicle being sliced
        # off with a strip of background beside it, since the image
        # continues past the cut. Height keeps the margin (the vehicle
        # still wants headroom); only the horizontal inset is dropped.
        win_w = float(br - bl)
        x0 = float(bl)
        y0 = (bb - (bb - bt) * margin_frac - avail_h) if hero_anchor == "bottom" else (bt + ((bb - bt) - avail_h) / 2)
        shot.hero_rect = (x0, y0, win_w, avail_h)

        shot.pan_draw_w, shot.pan_draw_h = filled_w, avail_h
        hero_content_for_dim = shot.car.resize(
            (max(1, round(filled_w)), max(1, round(avail_h))), Image.LANCZOS
        ).crop((0, 0, max(1, round(win_w)), max(1, round(avail_h))))

        # Endpoints: the nose sits flush against one frame edge to start,
        # the tail flush against the other to finish (see PAN_BARS). Which
        # image edge is the nose decides both, and with it the slide
        # direction -- the cutout is a sprite over a STATIC flag, so
        # whichever way the image translates is the way the vehicle reads
        # as driving; it is the moving object, not a camera panning past a
        # parked truck. A right-facing vehicle must therefore slide RIGHT
        # to look like it's driving forward (front leading, like a truck
        # driving past you), which falls out of "start with the nose
        # against the right edge."
        if precomputed_hood_side is not None:
            # The normal path -- photos.py's download_photos() already ran
            # this CLIP call once, at cutout time, and stashed the answer in
            # angles.json. Falling back to a live call below only matters
            # for a caller that never had that chance (hero_video_cli.py
            # re-rendering a folder scraped before this field existed).
            shot.hood_side = precomputed_hood_side
        else:
            import io as _io
            buf = _io.BytesIO()
            shot.car.save(buf, format="PNG")
            shot.hood_side = detect_hood_side(buf.getvalue())
        xc = x0 + win_w / 2
        if shot.hood_side == "right":
            shot.pan_x_start, shot.pan_x_end = xc - filled_w, xc
        else:
            shot.pan_x_start, shot.pan_x_end = xc, xc - filled_w
    else:
        shot.hero_rect = shot.hero_fit_rect
        shot.pan_draw_w = shot.pan_draw_h = shot.pan_x_start = shot.pan_x_end = 0.0
        shot.hood_side = None
        hero_content_for_dim = hero_fit_resized

    shot.bars = PAN_BARS if shot.is_pan else 1

    x, y, w, h = shot.hero_rect
    car_box = (round(x), round(y), round(x + w), round(y + h))
    bg_region = representative_bg.crop(car_box)
    shot.dim_strength = compute_dim_strength(bg_region, hero_content_for_dim)
    shot.center = ((car_box[0] + car_box[2]) / 2, (car_box[1] + car_box[3]) / 2)
    return shot


def _hero_state(shot: _Shot, progress: float) -> tuple:
    """The draw rect for this shot as the hero at pan-progress `progress`
    (0..1 across its steady hold; ignored for a non-pan shot).

    For a pan shot the image is drawn oversized and slid sideways -- so a
    pan is a change in X only, at constant scale, which is what makes it
    interpolate cleanly against the accent rects (also whole-image
    draws). Whatever leaves the frame is simply covered by the border art
    composited on top, so the vehicle stays WHOLE and drives off screen
    rather than being sliced by a window edge. The slide direction always
    matches the way the vehicle faces, so it reads as driving forward --
    see _build_shot()'s direction note and detect_hood_side().
    """
    if not shot.is_pan:
        return shot.hero_fit_rect
    p = min(1.0, max(0.0, progress))
    x = shot.pan_x_start + (shot.pan_x_end - shot.pan_x_start) * p
    return (x, shot.hero_rect[1], shot.pan_draw_w, shot.pan_draw_h)


def _build_schedule(shots: list, dwell: float) -> tuple[list[tuple[float, float]], float]:
    """[(start, duration), ...] per shot plus the total period. Durations
    are whole bars (shot.bars), so a pan shot can hold the hero slot
    longer without any cut drifting off a downbeat."""
    schedule, t = [], 0.0
    for shot in shots:
        d = dwell * shot.bars
        schedule.append((t, d))
        t += d
    return schedule, t


def _slot_at(pos: float, schedule: list[tuple[float, float]]) -> tuple[int, float, float]:
    """(index, start, duration) of the slot containing `pos`."""
    for i, (start, duration) in enumerate(schedule):
        if pos < start + duration:
            return i, start, duration
    i = len(schedule) - 1
    return i, schedule[i][0], schedule[i][1]


def _scaled_about(rect: tuple, center: tuple[float, float], scale: float) -> tuple:
    """Scale a rect about an arbitrary canvas point (not its own center) --
    the beat pulse scales a pan hero's draw rect and its window about the
    SAME point, so the content zooms in place instead of the two sliding
    apart."""
    x, y, w, h = rect
    cx, cy = center
    return (cx + (x - cx) * scale, cy + (y - cy) * scale, w * scale, h * scale)


def _faded(resized: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return resized
    a = resized.split()[-1].point(lambda v: int(v * max(0.0, alpha)))
    out = resized.copy()
    out.putalpha(a)
    return out


_RENDER_CACHE: dict = {}
_RENDER_CACHE_MAX = 48

# Tier 3: GPU-accelerated resize + glow blur, swapped in under _render()'s
# EXISTING cache/composite/fade machinery -- not a parallel rendering
# pipeline. Every scheduling/geometry/transition decision (the part with
# four documented failed rewrite attempts, see module docstring) stays
# exactly the code it already was; only the pixel resampling inside a
# cache MISS changes backend. Measured on this machine: a single LANCZOS
# resize of a real cutout averages ~12ms on CPU vs ~0.6ms on GPU including
# transfer, and the glow's Gaussian blur ~0.5ms on GPU. Falls back to the
# PIL path automatically (_gpu_available() is False) on a machine with no
# CUDA -- same output contract either way, see effects.make_glow_layer for
# the CPU implementation this mirrors.
_gpu_checked = False
_gpu_ok = False
_gpu_source_cache: dict = {}  # id(PIL car) -> torch tensor, RGBA float32 CHW on GPU


def _gpu_available() -> bool:
    global _gpu_checked, _gpu_ok
    if not _gpu_checked:
        _gpu_checked = True
        try:
            import torch
            _gpu_ok = torch.cuda.is_available()
        except ImportError:
            _gpu_ok = False
    return _gpu_ok


def _gpu_source_tensor(car: Image.Image):
    """RGBA float32 CHW tensor on GPU for `car`, cached by object identity
    for the lifetime of one render_hero_video() call (module-level dict,
    cleared per-video at the end of render_hero_video)."""
    import torch

    key = id(car)
    hit = _gpu_source_cache.get(key)
    if hit is not None:
        return hit
    import numpy as np
    arr = np.asarray(car, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().cuda()
    _gpu_source_cache[key] = t
    return t


def _gpu_gauss_kernel1d(sigma: float, half_width: int, device):
    import torch
    x = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _gpu_render_entry(car: Image.Image, w: int, h: int, glow: bool, glow_color, glow_radius: int,
                       glow_intensity: float):
    """GPU equivalent of the CPU cache-fill in _render(): returns
    (img, halo, pad) with img/halo as PIL RGBA Images, exactly like the
    CPU path -- so the caller's compositing/fading code doesn't need to
    know which backend produced them."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    from .effects import resolve_glow_color

    src = _gpu_source_tensor(car).unsqueeze(0)  # 1x4xHxW
    resized = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=False,
                             antialias=True).clamp(0, 1)

    def to_pil_rgba(t):
        arr = (t.squeeze(0).permute(1, 2, 0) * 255).round().clamp(0, 255).byte().cpu().numpy()
        return Image.fromarray(arr, mode="RGBA")

    img = to_pil_rgba(resized)

    halo = None
    pad = 0
    if glow:
        # PIL's ImageFilter.GaussianBlur(radius) uses `radius` directly as
        # the kernel's standard deviation (measured empirically: a
        # radius=24 blur has sigma ~25, not radius/2 or radius/3) -- so the
        # GPU kernel has to use the same sigma to produce a halo the same
        # size as the CPU path's, not a tighter or looser one. half_width
        # at 3*sigma keeps 99.7% of the kernel's mass; pad matches it so
        # the halo has room to reach its own tail without being clipped by
        # the edge of its own canvas.
        sigma = float(glow_radius)
        # pad = radius*2 matches effects.make_glow_layer's own convention
        # exactly (unchanged since before this Tier 3 work) -- this is a
        # truncation of the "true" 3-sigma gaussian tail, not a full one,
        # but that IS the already-shipped, already-reviewed look. Using a
        # wider pad here would technically be a more accurate gaussian but
        # a VISIBLY DIFFERENT (larger) glow than every video rendered
        # before this change, which matters more than gaussian purity.
        half_width = glow_radius * 2
        pad = half_width
        alpha = resized[0, 3:4].unsqueeze(0)  # 1x1xHxW
        # Grow the canvas by `pad` on every side FIRST (zero/transparent
        # padding, matching the CPU path's Image.new("L", padded_size, 0)),
        # THEN blur within it -- same-size convs after this keep that
        # larger canvas, so the halo has room to spread past the car's own
        # silhouette instead of being clipped at it.
        alpha = F.pad(alpha, (pad, pad, pad, pad))
        k = _gpu_gauss_kernel1d(sigma, half_width, alpha.device)
        alpha = F.conv2d(alpha, k.view(1, 1, 1, -1), padding=(0, half_width))
        alpha = F.conv2d(alpha, k.view(1, 1, -1, 1), padding=(half_width, 0))
        alpha = (alpha * glow_intensity).clamp(0, 1)
        rgb = torch.tensor(resolve_glow_color(glow_color), dtype=torch.float32,
                            device=alpha.device).view(1, 3, 1, 1) / 255.0
        rgb = rgb.expand(1, 3, alpha.shape[2], alpha.shape[3])
        halo_t = torch.cat([rgb, alpha], dim=1)
        halo = to_pil_rgba(halo_t)

    return img, halo, pad


def _render(layer: Image.Image, shot: _Shot, draw_rect: tuple, alpha: float,
             glow: bool, glow_color, glow_radius: int, glow_intensity: float) -> None:
    """Draw the whole image at draw_rect. Resampled fresh from the
    full-res source each time rather than from an already-downscaled copy
    -- avoids compounding blur when the size changes every frame during a
    morph.

    Deliberately clips to NOTHING. An earlier version drew a pan through
    a window rect, which was fine while that window was the frame itself
    but sliced the vehicle in half the moment the window started shrinking
    toward the accent slot mid-transition. The frame edge is the only
    boundary the vehicle should ever be cut by, and the border art
    composited on top already provides it -- so anything leaving the frame
    just slides out of view, whole, the way a vehicle driving past should.
    """
    dx, dy, dw, dh = draw_rect
    w, h = max(1, round(dw)), max(1, round(dh))

    # Memoized on (image, size) -- NOT position. A pan slides the same
    # scaled bitmap sideways, and the beat pulse cycles through the same
    # handful of integer sizes every beat, so the LANCZOS resize and the
    # gaussian glow behind it are recomputed constantly for pixels that
    # are identical. Bounded because a hero-sized entry is a few MB.
    key = (id(shot.car), w, h, glow, glow_color, glow_radius, glow_intensity)
    entry = _RENDER_CACHE.get(key)
    if entry is None:
        if _gpu_available():
            img, halo, pad = _gpu_render_entry(shot.car, w, h, glow, glow_color, glow_radius, glow_intensity)
        else:
            # reducing_gap=2.0 lets PIL pre-shrink with a cheap box filter
            # before the final LANCZOS pass instead of resampling the
            # full-res source in one shot -- ~30-40% faster on a large
            # downscale with no visible quality difference, since
            # box-reducing first is exactly what LANCZOS's own
            # anti-aliasing needs anyway.
            img = shot.car.resize((w, h), Image.LANCZOS, reducing_gap=2.0)
            halo, pad = (make_glow_layer(img, color=glow_color, radius=glow_radius,
                                          intensity=glow_intensity) if glow else (None, 0))
        if len(_RENDER_CACHE) >= _RENDER_CACHE_MAX:
            _RENDER_CACHE.pop(next(iter(_RENDER_CACHE)))
        entry = _RENDER_CACHE[key] = (img, halo, pad)
    img, halo, pad = entry

    x, y = round(dx), round(dy)
    if halo is not None:
        faded_halo = _faded(halo, alpha)
        layer.paste(faded_halo, (x - pad, y - pad), faded_halo)
    faded = _faded(img, alpha)
    layer.paste(faded, (x, y), faded)


def _paste(layer: Image.Image, resized: Image.Image, x: int, y: int,
           glow: bool, glow_color, glow_radius: int, glow_intensity: float) -> None:
    if glow:
        from .effects import paste_with_glow
        paste_with_glow(layer, resized, x, y, color=glow_color, radius=glow_radius, intensity=glow_intensity)
    else:
        layer.paste(resized, (x, y), resized)


def render_hero_video(background_video: Path | None, border_path: Path | None, carousel_paths: list[Path],
                       audio_path: Path | None, out_path: Path, carousel_labels: list[str] | None = None,
                       gradient_colors: tuple | None = None, bpm: float = DEFAULT_BPM,
                       fps: float = DEFAULT_FPS, canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
                       loops: int | None = None, budget_mb: float = 50.0, layout: str = "conveyor", n_accents: int = 2,
                       bars_per_loop: int = BARS_PER_LOOP,
                       glow: bool = True, glow_color=DEFAULT_GLOW_COLOR,
                       glow_radius: int = 24, glow_intensity: float = 0.75,
                       encoder: str = "libx264",
                       hood_sides: dict[str, str] | None = None) -> dict:
    """
    carousel_paths: the FULL shot library, in conveyor order (see
    imaging/select.py::pick_all_for_carousel() -- deliberately not deduped
    to one-per-angle, the whole point of the conveyor is to show
    everything). carousel_labels is currently unused by the conveyor
    itself (kept for API compatibility / future use) since slot
    membership is now pure positional adjacency, not a diversity search.
    loops: how many audio loops long the video runs (audio duration is
    probed, not assumed -- see probe_duration_s()).

    Returns a small report dict (duration, bitrate, timing) for logging.
    """
    if layout != "conveyor" or n_accents != 2:
        raise ValueError("the conveyor is a 3-slot design (left/hero/right) -- layout must be "
                          f"'conveyor' with n_accents=2, got layout={layout!r} n_accents={n_accents!r}")
    if len(carousel_paths) < 3:
        raise ValueError("need at least 3 shots for a left/hero/right conveyor")

    if background_video is None and gradient_colors is None:
        raise ValueError("need either a background_video or gradient_colors to draw a backdrop")

    border = Image.open(border_path).convert("RGBA") if border_path is not None else None
    if border is not None:
        canvas_size = border.size
        border_mask = alpha_mask(border)
        window = detect_window(border)
    else:
        border_mask = None
        window = (0, 0, canvas_size[0], canvas_size[1])

    # "conveyor" means the conveyor arrangement that suits this frame, not
    # one specific function -- a vertical canvas needs its slots stacked
    # (layout.py::conveyor_stack_layout). An explicit layout name still
    # wins, so a caller can force either arrangement.
    layout_fn = conveyor_for_window(window) if layout == "conveyor" else LAYOUTS[layout]
    boxes = layout_fn(window, n_accents)
    (hero_box, hero_anchor), (left_box, left_anchor), (right_box, right_anchor) = boxes

    # The audio used to be the clock as well as the soundtrack. Without
    # it, a bar is derived from bpm instead, so every downstream timing
    # (cut, transition, pump) is unchanged in kind -- only its source
    # moves. `loops` still means "how many loop-lengths long", scored or not.
    if audio_path is not None:
        audio_loop_s = probe_duration_s(audio_path)
    else:
        audio_loop_s = bars_per_loop * BEATS_PER_BAR * 60.0 / bpm
    if background_video is not None:
        fps = probe_fps(background_video)

    bg_frames = _read_bg_frames(background_video, canvas_size) if background_video is not None else None
    # One representative backdrop frame for the spotlight's contrast
    # measurement (compute_dim_strength) -- it only needs a sample of
    # what sits behind the car, not the animation.
    representative_bg = (bg_frames[0] if bg_frames is not None
                          else make_linear_gradient(canvas_size, *gradient_colors))
    # Fresh per call: _gpu_source_cache is keyed by id(PIL Image), and those
    # ids get reused by CPython once a prior call's `shots` list (and the
    # Images it held) is garbage collected -- letting entries survive
    # across calls risks a stale-id false hit on a totally different
    # vehicle's cutout, and would otherwise grow unbounded (one GPU tensor
    # per cutout ever rendered) across a long multi-vehicle sync.
    _gpu_source_cache.clear()

    dwell, transition_s, beat_s = compute_carousel_timing(audio_loop_s, bars_per_loop)
    n = len(carousel_paths)
    shots = [
        _build_shot(p, hero_box, hero_anchor, left_box, left_anchor, right_box, right_anchor,
                    border_mask, representative_bg,
                    pannable=(carousel_labels[i] == PAN_ANGLE_LABEL) if carousel_labels else True,
                    precomputed_hood_side=(hood_sides.get(p.name) if hood_sides else None))
        for i, p in enumerate(carousel_paths)
    ]
    schedule, carousel_period = _build_schedule(shots, dwell)

    # Default length is ONE full pass of the shot library: the conveyor
    # has nothing new to show after that, and a run that stops there
    # can't strand a vehicle's last angles off the end (the old
    # loops-of-audio length was set by the soundtrack, which knows
    # nothing about how many photos this vehicle has). `loops` still
    # forces the old audio-driven length when given.
    total_seconds = carousel_period if loops is None else audio_loop_s * loops

    total_frames = round(total_seconds * fps)
    total_seconds = total_frames / fps  # snap to an exact frame count
    bitrate_kbps = compute_video_bitrate_kbps(
        total_seconds, budget_mb=budget_mb,
        audio_kbps=AUDIO_BITRATE_KBPS if audio_path is not None else 0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".ffmpeg.log")

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{canvas_size[0]}x{canvas_size[1]}",
        "-r", str(fps), "-i", "pipe:0",
    ]
    if audio_path is not None:
        cmd += ["-stream_loop", "-1", "-i", str(audio_path), "-map", "0:v", "-map", "1:a"]
    cmd += [
        "-c:v", encoder, "-preset", "medium", "-b:v", f"{bitrate_kbps}k",
        "-maxrate", f"{int(bitrate_kbps * 1.2)}k", "-bufsize", f"{int(bitrate_kbps * 2)}k",
        "-pix_fmt", "yuv420p",
    ]
    if audio_path is not None:
        cmd += ["-c:a", "aac", "-b:a", f"{AUDIO_BITRATE_KBPS}k"]
    cmd += ["-t", f"{total_seconds:.3f}", str(out_path)]

    accent_layer_cache: dict = {}

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logf)
        try:
            for f in range(total_frames):
                t = f / fps
                pos = t % carousel_period
                idx, slot_start, slot_duration = _slot_at(pos, schedule)
                within = pos - slot_start
                hold = slot_duration - transition_s
                hero = shots[idx]
                if bg_frames is not None:
                    bg_frame = bg_frames[f % len(bg_frames)]
                else:
                    # Rotating rather than looping: a generated backdrop has
                    # no seam to hide, so it can just keep turning.
                    bg_frame = make_linear_gradient(
                        canvas_size,
                        gradient_colors[0] + GRADIENT_TURNS * 360.0 * (f / max(1, total_frames)),
                        gradient_colors[1], gradient_colors[2])

                # bg_frame is already RGB; letting apply_spotlight convert
                # in and back out cost two full-canvas conversions a frame.
                canvas = apply_spotlight(bg_frame, hero.center, hero.dim_strength).convert("RGBA")

                if within < hold:
                    # Steady: left=shots[idx-1], hero=shots[idx] (panning
                    # and/or pulsing), right=shots[idx+1].
                    #
                    # The two accents are motionless for the whole bar, but
                    # were being re-resized (LANCZOS) and re-blurred (glow)
                    # every frame to produce an identical layer. Cache it
                    # per neighbour pair -- at most `n` distinct layers for
                    # a whole video. Only the hero still needs live work,
                    # since it pans and pulses.
                    left_i, right_i = (idx - 1) % n, (idx + 1) % n
                    base = accent_layer_cache.get((left_i, right_i))
                    if base is None:
                        base = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
                        lb, rb = shots[left_i], shots[right_i]
                        _render(base, lb, lb.left_rect, 1.0, glow, glow_color, glow_radius, glow_intensity)
                        _render(base, rb, rb.right_rect, 1.0, glow, glow_color, glow_radius, glow_intensity)
                        accent_layer_cache[(left_i, right_i)] = base
                    layer = base.copy()

                    draw = _hero_state(hero, within / hold if hold > 0 else 1.0)
                    pulse = _pulse_scale(t, beat_s)
                    if pulse != 1.0:
                        # Pulse about the FRAME's center, not the drawn
                        # image's own center: a panning image is mostly
                        # off-frame, so scaling about its own center would
                        # swing the visible part sideways instead of
                        # zooming what the viewer is actually looking at.
                        pivot = (hero.hero_rect[0] + hero.hero_rect[2] / 2,
                                 hero.hero_rect[1] + hero.hero_rect[3] / 2)
                        draw = _scaled_about(draw, pivot, pulse)
                    _render(layer, hero, draw, 1.0, glow, glow_color, glow_radius, glow_intensity)
                else:
                    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
                    # Transition: A(idx) hero->left, B(idx-1) left fades out,
                    # C(idx+1) right->hero, D(idx+2) fades in at right.
                    #
                    # A and C interpolate from the exact state steady
                    # playback left off at (pan progress 1) to the exact
                    # state it will resume at (pan progress 0), so a pan
                    # shot's hand-off is continuous by construction -- no
                    # cut, no dissolve, no pop. The vehicle stays whole
                    # throughout; it shrinks toward the accent slot rather
                    # than being progressively cut down to it.
                    tau = min(1.0, (within - hold) / transition_s)
                    ease = _ease_out(tau)
                    prev_i, next_i, next2_i = (idx - 1) % n, (idx + 1) % n, (idx + 2) % n
                    a, b, c, d = shots[idx], shots[prev_i], shots[next_i], shots[next2_i]

                    _render(layer, b, b.left_rect, 1.0 - tau, glow, glow_color, glow_radius, glow_intensity)
                    _render(layer, d, d.right_rect, tau, glow, glow_color, glow_radius, glow_intensity)

                    _render(layer, a, _lerp_rect(_hero_state(a, 1.0), a.left_rect, ease), 1.0,
                            glow, glow_color, glow_radius, glow_intensity)
                    _render(layer, c, _lerp_rect(c.right_rect, _hero_state(c, 0.0), ease), 1.0,
                            glow, glow_color, glow_radius, glow_intensity)

                canvas.alpha_composite(layer)
                if border is not None:
                    canvas.alpha_composite(border)
                proc.stdin.write(canvas.convert("RGB").tobytes())
        finally:
            proc.stdin.close()
            ret = proc.wait()
            _gpu_source_cache.clear()

    if ret != 0:
        raise RuntimeError(f"ffmpeg failed (exit {ret}) -- see {log_path}")
    log_path.unlink(missing_ok=True)

    return {
        "out_path": str(out_path),
        "duration_s": round(total_seconds, 2),
        "fps": fps,
        "total_frames": total_frames,
        "n_shots": n,
        "shot_order": [p.name for p in carousel_paths],
        "pan_shots": {s.path.name: f"hood {s.hood_side} -> slides "
                                    f"{'right' if s.pan_x_end > s.pan_x_start else 'left'}, {s.bars} bars"
                       for s in shots if s.is_pan},
        "carousel_period_s": round(carousel_period, 2),
        # Frame-rounding can leave total_seconds a hair under the
        # period; only flag a pass that is really cut short.
        "carousel_incomplete": carousel_period > total_seconds + 0.5 / fps,
        "audio_loop_s": round(audio_loop_s, 3),
        "bars_per_loop": bars_per_loop,
        "clock": "audio" if audio_path is not None else f"{bpm:g} bpm",
        "backdrop": "video" if background_video is not None else "gradient",
        "framed": border is not None,
        "bar_dwell_s": round(dwell, 3),
        "beat_s": round(beat_s, 3),
        "transition_s": round(transition_s, 2),
        "bitrate_kbps": bitrate_kbps,
        "file_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
    }
