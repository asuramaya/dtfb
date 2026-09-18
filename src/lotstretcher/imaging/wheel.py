"""
Wheel money-shot detection and extraction: find a "detail" photo that's
genuinely a rolling wheel shown at a good 3/4-ish angle (not a mirror/
badge/light/handle/tread-texture-only/mounted-spare shot that CLIP's
broader "detail" label also catches) and cut the wheel out cleanly.

Final architecture -- four stages, each a model/check doing the one thing
it's actually built for:

  1. IS it a wheel: WheelDetailClassifier + SpareTireClassifier (CLIP
     zero-shot, see classify.py). Kept narrow and separate on purpose --
     one big multi-way classifier misfired badly (a spare-tire label
     pulled in genuine clean wheel shots); CLIP zero-shot is far more
     reliable choosing between a few sharply-contrasted labels than among
     many overlapping ones in a single pass.
  2. WHERE is it: CLIPSeg (text-conditioned segmentation, "a car wheel
     and tire") -> bounding box only. Its semantic localization is
     excellent; its mask EDGES are low-res/fuzzy, so the mask itself is
     never used as an alpha channel -- only its bounds.
  3. CUT it out: SAM2 (facebook/sam2.1-hiera-small) prompted with that
     box on the FULL photo. SAM2 is promptable object-level segmentation
     -- it returns a crisp, complete, single-object mask, so there is
     nothing downstream to patch.
  4. Is the ANGLE right: the mask bbox's aspect ratio (width/height) must
     fall in a calibrated band -- too tall/narrow is a front/back tread-on
     shot, too wide is a mis-segmentation that grabbed body alongside the
     wheel (see MIN_/MAX_ANGLE_ASPECT for the measured gaps on each side;
     a CLIP good-angle/bad-angle classifier was tried first and wrongly
     rejected genuinely good shots -- geometry is the honest signal here).
  5. Is the RESULT a wheel: the finished cutout is re-classified by
     WheelDetailClassifier as a last independent check -- a
     well-proportioned mask of the wrong object still gets caught.

How it got here (condensed; the failures were all real, on real inventory
photos across 17 vehicles):

  - Hough-circle geometry for localization: high false-positive rate
    (door handles, mirrors, badges, headlights all score like wheels --
    geometry can't know what it's a circle OF), and even after CLIP took
    over the is-it-a-wheel call, Hough kept locking onto the wheel-well
    ARCH instead of the tire, so crops bled fender.
  - rembg (saliency matting) for the cutout: beautiful edges, no concept
    of "wheel". Four distinct real failure modes on production photos:
    brake calipers visible through spoke gaps cut out as holes; dark tire
    rubber dropped wholesale against dark wheel wells (one wheel reduced
    to a bare floating rim); soft shadow kept as foreground; fender bleed
    at crop corners.
  - A growing stack of geometric repairs on top of rembg (mask union with
    CLIPSeg, scipy hole-filling, RANSAC ellipse fitting with crop-border
    exclusion, a centroid-symmetry "fabrication gate", a rim-only
    fallback) fixed each failure it targeted but kept minting new edge
    cases -- and on one pristine, fully-visible wheel photo the whole
    stack still shipped garbage. The lesson that ended it: every layer
    existed to reconcile two models that were each half-blind (CLIPSeg
    knows WHAT but not exactly WHERE; rembg draws WHERE but not WHAT).
    Swapping the cutout core for a model that does promptable
    pixel-accurate segmentation natively (SAM2) made the entire repair
    stack deletable: on the same 11-photo validation set that broke every
    previous revision, SAM2 one-shot masks were clean on all 10 good
    candidates (IoU 0.94-0.99, tire tread texture intact, no holes, no
    bleed, no fabrication) and the one genuine bad-angle photo still
    measured 0.53 aspect -- correctly rejected by stage 4.

SAM2 runs fully local via transformers (CUDA if available); weights
download once on first use (~180MB).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .classify import SpareTireClassifier, WheelDetailClassifier

CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
WHEEL_SEGMENT_PROMPT = "a car wheel and tire"
SAM2_MODEL = "facebook/sam2.1-hiera-small"

# Sigmoid probability threshold for a pixel to count as "wheel" in
# CLIPSeg's output mask (only the mask's BOUNDS are used, as SAM2's box
# prompt). 0.5 is the plain more-likely-than-not default; the box it
# produced was validated against all 11 real test photos without needing
# adjustment.
WHEEL_MASK_THRESHOLD = 0.5

# Calibrated: the lowest confidence seen on a genuine wheel photo was 0.52;
# every false-positive category (mirror/light/badge/handle/tread/other)
# scored its own label higher than "wheel" in every case checked.
WHEEL_CONFIDENCE_THRESHOLD = 0.5

# Calibrated: a real mounted spare scored 0.75-0.96; every genuine rolling-
# wheel shot tested (6 real cases) scored "rolling_wheel" instead, several
# with real margin (0.59-0.98).
SPARE_TIRE_CONFIDENCE_THRESHOLD = 0.5

# Calibrated: every confirmed-good 3/4-angle wheel photo measured a mask
# bbox aspect ratio (width/height) of 1.0-1.9; every confirmed-bad front/
# back tread-on angle measured 0.39-0.66 (re-confirmed at 0.53 under
# SAM2). 0.85 sits in the clean gap between them, closer to the bad side
# for a small safety margin.
MIN_ANGLE_ASPECT = 0.85

# Upper bound, added as defense in depth after seeing what a routing
# failure produces: when a full-car photo slipped into extraction (before
# is_wheel_centric_shot existed), SAM2 segmented the whole car body and
# the wide result sailed through the floor-only aspect gate. Calibrated
# under SAM2: all 14 real accepted money shots measure 0.98-1.12; the two
# real full-body masks measured 1.68 and 3.38. 1.4 sits mid-gap. (The old
# 1.9 upper figure came from rembg-era bboxes, which ran looser than
# SAM2's tight object masks.)
MAX_ANGLE_ASPECT = 1.4

# is_wheel_centric_shot() gates (see its docstring). Calibrated across all
# 18 photos an earlier, classifier-only reroute pulled in, plus the
# known-good detail-labeled money shots: every genuine wheel-centric photo
# measured dominance 1.00 (one wheel visible -> one blob) and largest-blob
# width 0.28-0.71 of the frame; every full-car photo measured dominance
# 0.50-0.65 (both axles visible -> 2-3 blobs) and width 0.09-0.20. Both
# thresholds sit in their respective measured gaps. The width gate also
# guarantees money-shot resolution: even a hypothetical single-visible-
# wheel photo taken from across the lot would be too small to ship.
WHEEL_CENTRIC_MIN_DOMINANCE = 0.8
WHEEL_CENTRIC_MIN_WIDTH_FRACTION = 0.24


@dataclass
class WheelVerdict:
    is_wheel: bool
    label: str
    confidence: float


def evaluate_wheel_detail(content: bytes, wheel_classifier: WheelDetailClassifier) -> WheelVerdict:
    """Call only on a photo SceneClassifier already labeled "detail" (see
    imaging/pipeline.py's DETAIL_LABEL) -- this doesn't re-check that
    itself, to avoid a second full-image CLIP pass when the caller already
    has the scene label from its own classify() call."""
    c = wheel_classifier.classify(content)
    is_wheel = c.label == "wheel" and c.confidence >= WHEEL_CONFIDENCE_THRESHOLD
    return WheelVerdict(is_wheel=is_wheel, label=c.label, confidence=c.confidence)


def is_mounted_spare(content: bytes, spare_classifier: SpareTireClassifier) -> bool:
    c = spare_classifier.classify(content)
    return c.label == "mounted_spare" and c.confidence >= SPARE_TIRE_CONFIDENCE_THRESHOLD


_clipseg_processor = None
_clipseg_model = None


def _to_device(model, device: str, name: str):
    """Move a model to `device`, degrading to CPU if the GPU is full.
    See imaging/classify.py for why: the GPU is shared with the desktop,
    and a slow wheel pass beats losing the vehicle."""
    import torch
    try:
        return model.to(device).eval()
    except torch.cuda.OutOfMemoryError:
        print(f"    ! GPU out of memory loading {name}; falling back to CPU")
        torch.cuda.empty_cache()
        return model.to("cpu").eval()


def _clipseg():
    """Lazy-loads CLIPSeg once, shared across every extract_wheel_shot()
    call in a run rather than reloaded per photo -- same pattern as
    classify.py's default_backbone()."""
    global _clipseg_processor, _clipseg_model
    if _clipseg_model is None:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clipseg_processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL)
        _clipseg_model = _to_device(CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL), device, CLIPSEG_MODEL)
    return _clipseg_processor, _clipseg_model


_sam2_processor = None
_sam2_model = None


def _sam2():
    """Lazy-loads SAM2 once per run, same pattern as _clipseg()."""
    global _sam2_processor, _sam2_model
    if _sam2_model is None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL)
        _sam2_model = _to_device(Sam2Model.from_pretrained(SAM2_MODEL), device, SAM2_MODEL)
    return _sam2_processor, _sam2_model


_mask_memo: dict = {"key": None, "mask": None}


def _wheel_mask(img: Image.Image) -> "np.ndarray | None":
    """CLIPSeg wheel segmentation: boolean mask of every pixel it's at
    least WHEEL_MASK_THRESHOLD confident matches WHEEL_SEGMENT_PROMPT,
    upsampled to the image's size, or None when there's no confident wheel
    pixel at all. Never used as an alpha channel directly -- CLIPSeg's
    mask edges are too low-res to matte with (its decoder runs at a fixed
    small resolution and upsamples); callers use its bounds and blob
    structure only.

    Single-slot memo: the same photo gets examined twice back-to-back on
    the reroute path (is_wheel_centric_shot() during routing, then
    _find_wheel_bbox() during extraction), and the pipeline is strictly
    serial, so remembering just the last result halves the CLIPSeg cost
    on those photos without building a real cache."""
    import torch

    key = (img.size, hash(img.tobytes()))
    if _mask_memo["key"] == key:
        return _mask_memo["mask"]

    processor, model = _clipseg()
    device = next(model.parameters()).device
    inputs = processor(text=[WHEEL_SEGMENT_PROMPT], images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()
    prob_img = Image.fromarray((probs * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    mask = np.array(prob_img) > int(WHEEL_MASK_THRESHOLD * 255)
    result = mask if mask.any() else None
    _mask_memo["key"], _mask_memo["mask"] = key, result
    return result


def _find_wheel_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of _wheel_mask(), or None -- SAM2's box prompt."""
    mask = _wheel_mask(img)
    if mask is None:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def is_wheel_centric_shot(content: bytes) -> bool:
    """Is this photo COMPOSED AROUND one wheel (a valid money-shot source)
    rather than a full-vehicle shot that merely contains wheels? Needed
    because WheelDetailClassifier alone over-fires as a router: it calls
    some full-car photos "wheel" too (wheels are visually prominent), and
    routing those into extraction produced either mangled full-body masks
    (CLIPSeg's box spanned both axles, so SAM2 segmented the car) or
    postage-stamp wheel crops too small to ship. The honest signal is
    CLIPSeg's blob structure: a wheel-centric photo shows exactly ONE
    wheel, so its confident wheel pixels form a single dominant blob
    spanning a substantial fraction of the frame; a full-car photo shows
    both axles, so the pixels split into 2-3 far-apart blobs, none
    dominant, none large. See WHEEL_CENTRIC_MIN_* for the measured
    separation (no overlap on either signal across every real case)."""
    import io

    import cv2

    img = Image.open(io.BytesIO(content)).convert("RGB")
    mask = _wheel_mask(img)
    if mask is None:
        return False
    _n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = labels == int(counts.argmax())
    dominance = largest.sum() / mask.sum()
    xs = np.where(largest)[1]
    width_frac = (xs.max() - xs.min() + 1) / img.size[0]
    return bool(dominance >= WHEEL_CENTRIC_MIN_DOMINANCE
                and width_frac >= WHEEL_CENTRIC_MIN_WIDTH_FRACTION)


def _segment_wheel(img: Image.Image, box: tuple[int, int, int, int]) -> "np.ndarray | None":
    """SAM2 with a box prompt on the full photo: returns a boolean mask of
    the single object the box indicates. SAM2 proposes several candidate
    masks per prompt (part vs whole ambiguity); its own predicted-IoU
    score picks the winner -- on every real photo tested the best-scoring
    candidate was the complete wheel+tire. Reduced to its largest
    connected component: SAM occasionally tacks on a small detached
    fragment (confirmed real case: a scrap of mudguard floating beside an
    otherwise perfect wheel mask, which also skewed the bbox the aspect
    gate measures). Returns None only if the mask comes back empty."""
    import cv2
    import torch

    processor, model = _sam2()
    device = next(model.parameters()).device
    inputs = processor(images=img, input_boxes=[[list(box)]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs.original_sizes)[0]
    scores = outputs.iou_scores.cpu().numpy().ravel()
    arr = masks.numpy() if hasattr(masks, "numpy") else np.array(masks)
    arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    mask = arr[int(scores.argmax())].astype(bool)
    if not mask.any():
        return None

    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n_labels > 2:  # background + more than one foreground piece
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # never pick the background
        mask = labels == int(counts.argmax())
    return mask


def extract_wheel_shot(content: bytes, wheel_classifier: WheelDetailClassifier,
                        spare_classifier: SpareTireClassifier) -> Image.Image | None:
    """Full pipeline for one "detail"-labeled photo: semantic gates (is it
    a wheel, is it a mounted spare), CLIPSeg localization, SAM2 cutout,
    geometric angle-quality gate. Returns an RGBA cutout cropped tight to
    its visible pixels, or None if this photo doesn't clear every gate --
    callers should treat None as "no wheel shot here", not an error."""
    import io

    verdict = evaluate_wheel_detail(content, wheel_classifier)
    if not verdict.is_wheel:
        return None
    if is_mounted_spare(content, spare_classifier):
        return None

    img = Image.open(io.BytesIO(content)).convert("RGB")
    box = _find_wheel_bbox(img)
    if box is None:
        return None
    mask = _segment_wheel(img, box)
    if mask is None:
        return None

    ys, xs = np.where(mask)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    aspect = (right - left) / (bottom - top)
    if not (MIN_ANGLE_ASPECT <= aspect <= MAX_ANGLE_ASPECT):
        return None

    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = Image.fromarray(np.dstack([np.array(img), alpha]), mode="RGBA")
    cutout = rgba.crop((left, top, right, bottom))

    # Final verification, independent of every gate above: the finished
    # cutout itself must read as a wheel. Guards against a localization or
    # segmentation failure producing a well-proportioned mask of the wrong
    # thing -- calibrated on real output, every genuine money shot scored
    # wheel(0.97-1.00) here while both real full-body mis-segmentations
    # scored "other". One extra CLIP pass per accepted candidate.
    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    check = wheel_classifier.classify(buf.getvalue())
    if check.label != "wheel" or check.confidence < WHEEL_CONFIDENCE_THRESHOLD:
        return None

    return cutout
