"""
Detect and crop flat-color letterbox bars some dealer photo vendors pad
photos with -- seen on a used BMW's interior gallery: every one of 19
photos had a solid pure-white band, ~78px top / ~79px bottom (measured to
where real photo content actually starts), consistent across the whole
batch, almost certainly reserved space for a watermark that wasn't filled
in for that batch.

Detection method: find the single sharpest brightness jump within the
search window (not a running/accumulated comparison -- see history below
for why that failed), and only trust it if everything before the jump is
itself close to flat (confirms a genuine solid bar, not just a photo with
a bright patch near the edge). This is deliberately per-image, not a fixed
pixel count -- the thickness (and whether the bar exists at all) varies by
photo vendor/batch.

Two earlier approaches both failed on real data, which is why this one
looks the way it does:
  1. Testing each row's own internal flatness (low variance within the
     row), walking inward with a tight tolerance: stopped partway through
     the ~15px JPEG-compression blur at the bar/content boundary (that
     transition is still near-white in aggregate but has enough internal
     pixel-to-pixel noise to fail a strict flatness test), leaving a
     faint residual white band in the output -- found by eye on a real
     cropped sample.
  2. Comparing each row's mean color against the bar's own baseline with a
     looser tolerance: fixed #1, but a loose "how far has this drifted
     from the baseline" test is easily fooled when real content near the
     edge is ALSO bright (e.g. white leather seats close to the top of
     frame) -- it kept walking well into real photo content. Confirmed on
     real data: several BMW interior photos over-cropped to 90-290px, and
     it also broke the "don't touch photos with no bar" property on
     Tomball Ford's own (bar-free) exterior gallery.

Confirmed against Tomball Ford's own (bar-free) galleries: no false
positives. But single-image detection alone still isn't fully reliable --
a handful of the 19 BMW photos have a *gradual* brightness fade into the
transition (real content near the edge happens to also be bright: a light
headliner, a sunlit window) rather than a sharp cliff, so there's no clean
per-row jump for MIN_JUMP to find. What single-image detection can't
resolve, batch consensus can: this vendor's bar size is fixed across a
whole photo batch (confirmed: every image where detection *did* find a
clean edge agreed to within 1px), so once most of a batch clearly shows the
boundary, that same size can be trusted for the few ambiguous photos in the
same batch. Use detect_batch_bars() for a set of photos from the same
vehicle/gallery; fall back to detect_bars()/crop_bars() only for a single
image with no batch context.
"""
from __future__ import annotations

from PIL import Image

# A SECOND kind of vendor padding, needing a different signal entirely.
# Tomball Ford bakes a saturated pink banner into the top (and a contact
# block into the bottom) of its studio shots. detect_bars() is blind to it
# -- that banner carries bold white text, so its luminance is the opposite
# of flat and MAX_PRE_JUMP_STD rejects it outright (measured (0,0) on the
# real photo below). Its CHROMA is what's flat: one saturated hue clear
# across the frame.
#
# This matters beyond cosmetics, because the banner changes what CLIP
# thinks the photo IS. Confirmed on a real production miss -- a Nissan
# Rogue's dead-on front studio shot (RC731600 photo 2, 1242x930):
#
#     with banner:     scene=interior(0.872)  tiebreak exterior_body=0.426
#     banner removed:  scene=exterior(0.548)  tiebreak exterior_body=0.996
#
# It routed to images/interior/, so the Rogue's cutout gallery has no
# "front" angle at all (only front_3q/side/rear_3q) and its hero lost the
# front accent. That is the same failure the Accord hit (SA035661 photo 1),
# but worse: the Accord's 0.744 was rescued by lowering
# INTERIOR_TIEBREAK_THRESHOLD to 0.65, and 0.426 is nowhere near it. No
# threshold reaches this one; the banner has to actually come off.
#
# Measured row profiles, "fraction of the row with saturation > 0.35" and
# "interquartile spread of those pixels' hue, in degrees":
#
#     banner rows      frac 0.65-1.00   hue IQR 0.0-1.4
#     photo content    frac 0.13        hue IQR 11-15
#
# Hence the two conditions below: a banner row is mostly saturated AND
# essentially monochromatic. Real photo content that happens to be
# colourful (a red car filling frame) fails the second test, because paint
# under real lighting spreads its hue far wider than 6 degrees.
#
# Deliberately NOT a blind fixed-percentage crop. That was tried first --
# lopping a flat 6% off every photo -- and while it fixed the shot being
# stared at (tiebreak 0.744 -> 0.981) it broke 4 previously-correct
# classifications on the full set, because on a photo with no banner it is
# just deleting real content. Measuring the band means a photo without one
# is returned untouched.
BANNER_SATURATION = 0.35  # above this counts as a "strongly coloured" pixel
BANNER_MIN_FRACTION = 0.5  # share of the row that must be strongly coloured
BANNER_MAX_HUE_IQR = 6.0  # degrees; banner rows measured 0.0-1.4, content 11+
BANNER_MIN_PX = 12  # thinner than this isn't worth cropping

MIN_BAR_PX = 8  # ignore anything thinner than this -- could just be a
                 # coincidentally flat sky/ceiling strip at a real photo edge
MAX_BAR_FRAC = 0.20  # never search past this fraction of the image height
MIN_JUMP = 60  # minimum single-row luminance jump to count as a real edge
               # (the BMW bar->content jump measured ~217; a natural photo
               # gradient near an edge is nowhere close to this)
MAX_PRE_JUMP_STD = 8  # how flat (luminance std-dev) everything before the
                       # jump must be to trust it as a genuine solid bar


def _row_luminance(arr) -> "np.ndarray":
    """Perceptual luminance per row (mean over the row's width) for an
    HxWx3 uint8 array -- one scalar per row."""
    import numpy as np
    rgb = arr.astype(np.float64)
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return lum.mean(axis=1)


def _find_edge(row_lum, max_bar: int) -> int:
    import numpy as np

    window = row_lum[: max_bar + 1]
    if len(window) < 2:
        return 0
    diffs = np.abs(np.diff(window))
    jump_idx = int(diffs.argmax())
    if diffs[jump_idx] < MIN_JUMP:
        return 0  # no sufficiently sharp transition in range -- no bar
    if window[: jump_idx + 1].std() > MAX_PRE_JUMP_STD:
        return 0  # what's before the jump isn't flat enough to be a bar
    return jump_idx + 1


def detect_bars(img: Image.Image) -> tuple[int, int]:
    """Returns (top_px, bottom_px) of detected solid-color padding, each
    0 if no bar (below MIN_BAR_PX) was found on that edge."""
    import numpy as np

    arr = np.asarray(img.convert("RGB"))
    h = arr.shape[0]
    max_bar = int(h * MAX_BAR_FRAC)

    row_lum = _row_luminance(arr)
    top = _find_edge(row_lum, max_bar)
    bottom = _find_edge(row_lum[::-1], max_bar)

    return (top if top >= MIN_BAR_PX else 0), (bottom if bottom >= MIN_BAR_PX else 0)


def _banner_rows(hue, sat, max_bar: int) -> int:
    """How many rows in from this edge are banner, walking inward until a
    row fails. Contiguous by design: the band touches the frame edge, so a
    gap means the banner ended and real content began."""
    import numpy as np

    count = 0
    for r in range(min(max_bar, len(sat))):
        strong = sat[r] > BANNER_SATURATION
        if strong.mean() < BANNER_MIN_FRACTION:
            break
        hues = hue[r][strong]
        if len(hues) < 10:
            break
        spread = float(np.percentile(hues, 75) - np.percentile(hues, 25))
        # Hue is circular, so a band sitting near the 0/360 wrap reads as a
        # huge spread when it is actually tight. Re-measure rotated half a
        # turn and keep whichever is smaller.
        rotated = (hues + 180.0) % 360.0
        spread = min(spread, float(np.percentile(rotated, 75) - np.percentile(rotated, 25)))
        if spread > BANNER_MAX_HUE_IQR:
            break
        count = r + 1
    return count


def detect_banner(img: Image.Image) -> tuple[int, int]:
    """Returns (top_px, bottom_px) of a saturated single-hue dealer banner,
    each 0 if that edge doesn't have one. See the constants above for the
    measured signal and the real miss that motivated it."""
    import numpy as np

    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    max_bar = int(hsv.shape[0] * MAX_BAR_FRAC)

    top = _banner_rows(hue, sat, max_bar)
    bottom = _banner_rows(hue[::-1], sat[::-1], max_bar)
    return (top if top >= BANNER_MIN_PX else 0), (bottom if bottom >= BANNER_MIN_PX else 0)


def strip_banner(img: Image.Image) -> Image.Image:
    """The image with any dealer banner cropped off, or `img` itself when
    there isn't one -- so this is always safe to call before classifying."""
    top, bottom = detect_banner(img)
    if not top and not bottom:
        return img
    w, h = img.size
    return img.crop((0, top, w, h - bottom))


def detect_batch_bars(images: list[Image.Image]) -> tuple[int, int]:
    """Run detect_bars() across a batch of photos from the same
    vehicle/gallery and return the most common non-zero (top, bottom) --
    more reliable than trusting any single image, since a real vendor bar
    is a fixed size across the whole batch even on photos where THIS
    image's own transition is too gradual for detect_bars() to find
    cleanly. Returns (0, 0) if no image in the batch showed a clear bar."""
    from collections import Counter

    results = [detect_bars(img) for img in images]
    tops = Counter(t for t, _ in results if t)
    bottoms = Counter(b for _, b in results if b)
    top = tops.most_common(1)[0][0] if tops else 0
    bottom = bottoms.most_common(1)[0][0] if bottoms else 0
    return top, bottom


def crop_bars(img: Image.Image, top: int | None = None, bottom: int | None = None) -> Image.Image:
    """Crop letterbox bars off. With no args, detects on this image alone.
    Pass top/bottom (e.g. from detect_batch_bars() over the whole gallery)
    to use a known batch-consensus size instead of per-image detection --
    prefer that when cropping more than one photo from the same vehicle.
    Returns `img` unchanged (same object) if there's nothing to crop, so
    this is always safe to call."""
    if top is None and bottom is None:
        top, bottom = detect_bars(img)
    top, bottom = top or 0, bottom or 0
    if not top and not bottom:
        return img
    w, h = img.size
    return img.crop((0, top, w, h - bottom))
