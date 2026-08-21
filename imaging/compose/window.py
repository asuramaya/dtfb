"""Finding the transparent "window" a border frame leaves for the car(s)."""
from __future__ import annotations

from PIL import Image


def detect_window(border: Image.Image, min_width_frac: float = 0.5) -> tuple[int, int, int, int]:
    """Find the border's main transparent window as (left, top, right, bottom).

    Scans for the widest contiguous run of near-transparent pixels in a
    handful of rows spread through the middle of the image, then does the
    same for columns -- avoids being fooled by small transparent details
    elsewhere in the frame (rounded corners, logo cutouts, etc.), which are
    narrower than the real window. Not hardcoded to any one border image, so
    a differently-shaped future border still works without code changes.
    """
    import numpy as np

    alpha = np.array(border.split()[-1])
    h, w = alpha.shape
    transparent = alpha < 10
    min_width = int(w * min_width_frac)
    min_height = int(h * min_width_frac)

    def widest_run(mask_1d):
        idx = np.where(mask_1d)[0]
        if len(idx) == 0:
            return None
        breaks = np.where(np.diff(idx) != 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(idx) - 1]
        runs = [(idx[s], idx[e]) for s, e in zip(starts, ends)]
        return max(runs, key=lambda r: r[1] - r[0])

    left, right = None, None
    for y in range(h // 4, 3 * h // 4, max(1, h // 40)):
        run = widest_run(transparent[y])
        if run and (right is None or run[1] - run[0] > right - left):
            if run[1] - run[0] >= min_width:
                left, right = run

    top, bottom = None, None
    for x in range(w // 4, 3 * w // 4, max(1, w // 40)):
        run = widest_run(transparent[:, x])
        if run and (bottom is None or run[1] - run[0] > bottom - top):
            if run[1] - run[0] >= min_height:
                top, bottom = run

    if left is None or top is None:
        raise ValueError("Could not find a transparent window in this border image")
    return (int(left), int(top), int(right), int(bottom))


def alpha_mask(border: Image.Image, threshold: int = 10) -> "np.ndarray":
    """Boolean array, True where the border art is opaque (>= threshold)."""
    import numpy as np
    return np.array(border.split()[-1]) >= threshold


def resolve_collision(border_mask: "np.ndarray", car: Image.Image, x: int, y: int,
                       max_shift: int = 300, step: int = 3, margin_frac: float = 0.04) -> int:
    """
    Nudge a car placed at (x, y) straight down until its opaque pixels no
    longer overlap the border's opaque pixels, checked directly against the
    border's actual alpha channel -- not a precomputed rectangle. This is
    what makes it work regardless of the border's shape (a badge/logo
    hanging lower in the middle than at the edges, say) or the car's own
    silhouette/aspect ratio: detect_window() only ever finds ONE bounding
    rectangle for the whole window, which is wrong wherever the border's
    real edge dips into that rectangle (confirmed case: a side-profile
    accent shot placed at the window's nominal top collided with the Ford
    oval hanging down from the header, even though corner shots at the same
    y were clear). A hardcoded per-layout margin would only patch this one
    border/vehicle combination and silently re-break on the next one.

    Stopping at the exact first zero-overlap pixel technically clears the
    border but reads as uncomfortably tight -- confirmed on a real render:
    a side accent's roofline sat right under the Ford oval with visually no
    breathing room even though no pixel actually overlapped. So once a
    clear position is found, this also tries nudging further down by
    margin_frac of the car's OWN height (not a fixed pixel count, so it
    scales with whatever size car ends up in this slot) and keeps that
    instead if it's still clear -- pure padding, not a second collision fix.

    Only shifts down (never up or sideways): the case this fixes is content
    poking up into an overhead header, and "move down" is the natural
    correction. Returns the original y unchanged if there's no collision to
    begin with, or the least-overlapping shift tried if it can't fully
    clear within max_shift (should not normally happen with reasonable
    layout boxes; better to show a small overlap than throw).
    """
    import numpy as np

    car_mask = np.array(car.split()[-1]) >= 10
    ch, cw = car_mask.shape
    bh, bw = border_mask.shape

    def overlap_at(yy: int) -> int | None:
        """None if this y is out of bounds, else the pixel-overlap count."""
        if yy < 0 or yy + ch > bh or x < 0 or x + cw > bw:
            return None
        return int(np.count_nonzero(border_mask[yy:yy + ch, x:x + cw] & car_mask))

    best_y, best_overlap = y, None
    for shift in range(0, max_shift + 1, step):
        yy = y + shift
        overlap = overlap_at(yy)
        if overlap is None:
            break
        if overlap == 0:
            padded = yy + max(1, round(ch * margin_frac))
            return padded if overlap_at(padded) == 0 else yy
        if best_overlap is None or overlap < best_overlap:
            best_y, best_overlap = yy, overlap
    return best_y
