"""
Combined decision pipeline: CLIP interior/exterior/marketing/document
labeling, with two different tiebreakers for two different kinds of
ambiguity -- rembg segmentation when CLIP calls "marketing", a second
narrower CLIP pass when CLIP calls "interior". Exterior/document/detail
calls are trusted from CLIP alone.

Why not run rembg on everything: Tomball Ford's source photos have
dealer-branding (a bold banner + tiled watermark) baked directly into the
pixels of otherwise-real exterior shots, which makes CLIP's holistic
marketing-vs-exterior score genuinely ambiguous on that subset (a real
photo scored exterior=0.024 / marketing=0.952 in testing). rembg's
segmentation resolves that specific ambiguity well: the one true junk image
(a full-bleed marketing card) segmented to a bounding box touching all four
edges (0, 0, W, H), while every real *exterior* photo, banner or not, left
visible margin on every side.

A separate, real failure: CLIP occasionally calls a genuine straight-on
exterior shot "interior" outright (confirmed real case: a studio front shot
of a Bronco, dealer banner and all, scored interior over exterior). The
FIRST fix attempt reused the same rembg margin trick on the theory that a
real exterior photo leaves background margin on every side while a cabin
shot fills the frame -- this looked right on a handful of cases but failed
badly at full-population scale (225 real interior photos): rembg routinely
isolates SOME salient blob inside a busy cabin scene (a console, a mirror,
a seat) that also happens to leave margin on every side, and neither that
margin nor the blob's area/aspect ratio reliably told a real car body apart
from an arbitrary interior detail -- it flagged 104/225 (46%) as exterior.
What actually works: a second, narrower, sharply-contrasted 2-way CLIP
classifier (InteriorExteriorTiebreakClassifier, see classify.py) run only
on "interior" calls -- caught the 1 real bug and left the other 224 alone,
including one genuinely ambiguous edge case (a dashboard screen showing a
3D car render) that it scored with meaningfully lower confidence than the
real bug, hence INTERIOR_TIEBREAK_THRESHOLD below. This is the same lesson
WHEEL_DETAIL_LABELS/SPARE_TIRE_LABELS already learned (see imaging/wheel.py):
CLIP zero-shot generalizes far better across a narrow, sharply-contrasted
choice than a geometric heuristic does across visually inconsistent real
photos.

Both of those interior-vs-exterior failures share a cause that neither
threshold nor tiebreak can reach: this dealer bakes a saturated pink
banner into the top of its studio shots, and that banner changes what CLIP
thinks the photo is. Measured on a real miss (Nissan Rogue RC731600 photo
2, the dead-on front shot): with the banner it scores scene=interior(0.872)
/ exterior_body(0.426) and lands in images/interior/, leaving that vehicle
with no "front" cutout at all; with the banner cropped off it scores
scene=exterior(0.548) / exterior_body(0.996). 0.426 is far below any
threshold this file could sanely carry -- the Accord's 0.744 was rescuable
by lowering INTERIOR_TIEBREAK_THRESHOLD, this is not. So evaluate_photo()
now classifies a banner-stripped view of the photo (see _as_classified()
and imaging/letterbox.py::detect_banner). Only the classifier's view is
stripped; the saved photo and its cutout still come from the original
bytes. Verified over all 639 photos on disk: the banner is detected on 63
of them and changes ZERO classifications, while on the Rogue's original
gallery it flips exactly the one photo that was wrong. That distinction is
the whole point -- an earlier attempt to crop a blind flat 6% off every
photo fixed the shot it was aimed at but broke 4 correct classifications,
because on a photo with no banner it was just deleting real content.

A separate axis: CLIP's "detail" label (a headlight/taillight/wheel/badge
close-up) is a real category of exterior photo, not a rejection -- it
belongs in the exterior gallery folder same as a wide hero shot. What it
should NOT get is a background cutout: rembg segments a tight, high-
contrast detail shot just as confidently as a full car (same foreground/
background contrast either way), so cutout.py's ambiguous-alpha quality
gate alone can't tell them apart -- confirmed empirically, every detail
shot in testing segmented "successfully" and would otherwise have produced
a cutout of a lone headlight floating alone. So cutout_eligible is its own
flag here, separate from category -- this is the FIRST line of defense,
and the cheaper one (skips the rembg call entirely). It isn't the only
one: a borderline crop can still fool CLIP into an "exterior" call
(confirmed real case: a wheel/fender close-up), so cutout.py's
remove_background() also checks the segmented bbox's margins against the
original frame and rejects anything that fills it edge-to-edge -- the
geometric signature of a tight crop regardless of what CLIP called it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cutout import remove_background
from .classify import InteriorExteriorTiebreakClassifier, SceneClassifier
from .letterbox import strip_banner

MARKETING_LABEL = "marketing"
DETAIL_LABEL = "detail"
INTERIOR_LABEL = "interior"
MIN_MARGIN_FRACTION = 0.02

# Calibrated against 225 real interior photos: the one real bug (a genuine
# exterior shot mislabeled "interior") scored exterior_body(0.997); the one
# genuinely ambiguous case (a dashboard screen displaying a 3D car render --
# arguably still interior content) scored only exterior_body(0.550). 0.75
# sits comfortably in the gap, closer to the ambiguous side so a real bug
# with slightly lower confidence than 0.997 still gets caught.
# 0.75 originally, lowered to 0.65 on a real miss: a straight-on front
# shot of an Accord (SA035661 photo 01) scored scene=interior(0.54) with
# the tiebreak calling it exterior_body at 0.744 -- correct, and rejected
# by 0.006. It landed in images/interior/, which cost the hero its front
# accent, so a near-miss here silently degrades the composition too.
# The floor is set by the known-ambiguous case this threshold exists to
# protect (a dash screen displaying a car render, 0.55); 0.65 sits above
# that and below the real miss. Widening further would need new evidence
# -- the two are only 0.19 apart.
INTERIOR_TIEBREAK_THRESHOLD = 0.65


@dataclass
class PhotoVerdict:
    category: str  # "exterior" | "interior" | "no_subject" | "document"
    cutout_eligible: bool
    clip_label: str
    clip_confidence: float
    rembg_checked: bool
    margins: tuple[float, float, float, float] | None = None


def _margins(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    w, h = size
    return (left / w, top / h, (w - right) / w, (h - bottom) / h)


def should_extract_wheel(content: bytes, verdict: "PhotoVerdict | None", wheel_classifier) -> bool:
    """The wheel-vs-whole-vehicle routing decision for one exterior photo,
    shared verbatim by photos.py (production) and regression.py (the
    calibration harness) so the two can never drift apart. True when the
    photo should go to imaging/wheel.py's money-shot pipeline INSTEAD of
    the whole-vehicle cutout treatment: either the scene classifier
    already called it a detail close-up, or it's a wheel-centric partial
    body shot that reads as "exterior" but both WheelDetailClassifier and
    the single-dominant-wheel geometry check agree on -- see the routing
    comment in photos.py::download_photos() for the real cases behind
    each branch."""
    from .wheel import evaluate_wheel_detail, is_wheel_centric_shot

    if verdict is None or wheel_classifier is None:
        return False
    if verdict.clip_label == DETAIL_LABEL:
        return True
    return (verdict.cutout_eligible
            and evaluate_wheel_detail(content, wheel_classifier).is_wheel
            and is_wheel_centric_shot(content))


def _as_classified(content: bytes) -> bytes:
    """The photo as the classifier should see it, with any baked-in dealer
    banner cropped off (imaging/letterbox.py::strip_banner).

    Only the CLASSIFICATION view is stripped -- callers keep the original
    bytes for what they save and cut out, so the banner still ships on the
    gallery photo exactly as before. Returns `content` unchanged when
    there's no banner, and on any decode failure, so this can never make a
    photo worse than not calling it.
    """
    import io

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(content))
        stripped = strip_banner(img)
        if stripped is img:
            return content
        buf = io.BytesIO()
        stripped.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return content


def evaluate_photo(content: bytes, classifier: SceneClassifier,
                    interior_tiebreak_classifier: "InteriorExteriorTiebreakClassifier | None" = None) -> PhotoVerdict:
    content = _as_classified(content)
    c = classifier.classify(content)

    if c.label == DETAIL_LABEL:
        # A real exterior photo, just not a cutout candidate.
        return PhotoVerdict("exterior", False, c.label, c.confidence, rembg_checked=False)

    if c.label == INTERIOR_LABEL:
        if interior_tiebreak_classifier is not None:
            t = interior_tiebreak_classifier.classify(content)
            if t.label == "exterior_body" and t.confidence >= INTERIOR_TIEBREAK_THRESHOLD:
                return PhotoVerdict("exterior", True, c.label, c.confidence, rembg_checked=False)
        return PhotoVerdict("interior", False, c.label, c.confidence, rembg_checked=False)

    if c.label != MARKETING_LABEL:
        # CLIP alone was accurate on exterior/document calls in testing --
        # no need to pay rembg's (much slower) cost here.
        return PhotoVerdict(c.label, c.label == "exterior", c.label, c.confidence, rembg_checked=False)

    # Ambiguous: could be a real banner-laden exterior photo or true junk.
    from PIL import Image
    import io

    cutout = remove_background(content)
    if cutout.bbox is None:
        return PhotoVerdict("no_subject", False, c.label, c.confidence, rembg_checked=True)

    size = Image.open(io.BytesIO(content)).size
    margins = _margins(cutout.bbox, size)
    has_subject = min(margins) >= MIN_MARGIN_FRACTION

    if not has_subject:
        return PhotoVerdict("no_subject", False, c.label, c.confidence, rembg_checked=True, margins=margins)

    # Has a real subject despite the "marketing" label -- almost always the
    # banner-confusion case, which was exterior in every observed instance.
    # Fall back to whichever of interior/exterior/detail CLIP scored highest,
    # in case a future interior or detail shot ever trips this path.
    category = max(("interior", "exterior", "detail"), key=lambda k: c.scores[k])
    if category == "detail":
        return PhotoVerdict("exterior", False, c.label, c.confidence, rembg_checked=True, margins=margins)
    return PhotoVerdict(category, category == "exterior", c.label, c.confidence, rembg_checked=True, margins=margins)
