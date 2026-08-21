"""
CLIP zero-shot classification for downloaded vehicle photos.

Five classifiers share one loaded CLIP model (ClipBackbone) rather than
each loading its own copy -- distinct label sets, same underlying embeddings:

  SceneClassifier: exterior ("body") vs interior vs marketing-graphic vs
  document/other vs close-up detail shot.

  AngleClassifier: which exterior angle a photo shows (front 3/4, rear 3/4,
  side profile, straight front, straight rear) -- so a hero composition can
  pick "a front 3/4 for the hero, a side profile and a straight front for
  accents" automatically instead of a human eyeballing filenames. Tested
  against 8 real cutouts: matched manual labeling on 7/8, with the one
  disagreement being a genuinely ambiguous rear-angle shot the model itself
  flagged as a near-even split rather than a confident wrong answer.

  WheelDetailClassifier: sub-classifies a SceneClassifier "detail" shot into
  wheel vs everything else that label also catches (mirror/light/badge/
  handle/tread-only/other) -- see WHEEL_DETAIL_LABELS below and
  imaging/wheel.py for what consumes it.

  SpareTireClassifier: a second, narrower pass over anything the above
  calls "wheel" -- a mounted spare isn't the rolling-wheel money shot
  wanted, but reads as a real wheel under WHEEL_DETAIL_LABELS' broader
  prompt. See SPARE_TIRE_LABELS below for why this is its own classifier
  rather than an 8th label folded into the one above.

  InteriorExteriorTiebreakClassifier: a second pass over anything
  SceneClassifier calls "interior" -- SCENE_LABELS' 5-way prompt
  occasionally calls a genuine straight-on exterior body shot "interior"
  outright (confirmed real case: a studio front shot of a Bronco). This
  narrower 2-way choice is what actually catches it -- see
  INTERIOR_EXTERIOR_TIEBREAK_LABELS below and imaging/pipeline.py for how
  it's used.

Runs fully local (CUDA if available, else CPU) after a one-time model-
weight download; no network calls at classification time.

Note: open_clip's "openai"-pretrained weights are fetched from
openaipublic.azureedge.net, which is unreachable in some sandboxed
environments (seen returning HTTP 400 here). We use a LAION checkpoint
instead, which open_clip resolves via huggingface.co.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"  # HF-hub hosted; avoids the openai/azureedge weights path

SCENE_LABELS = {
    "exterior": "a wide photo of the entire exterior of a car, showing the whole vehicle body and paint",
    "interior": "a photo of the interior of a car, showing the dashboard or seats",
    "detail": "a close-up photo of a single small car part, such as a headlight, taillight, wheel, badge, or mirror, not the whole vehicle",
    "marketing": "a marketing or advertisement graphic with text and logos",
    "document": "a scanned document, brochure page, or window sticker",
}

# "Informative prompts" (arXiv 2402.19150) tested as a SECOND, independent
# defense against the typographic-attack effect (see imaging/letterbox.py's
# strip_banner, the deployed fix), on the 93 real dealer-banner photos on
# disk at the time: rewording exterior/interior/marketing to explicitly
# acknowledge "may have a small banner/watermark, but a real vehicle fills
# most of the frame" DOES help on raw (unstripped) photos -- 12/93 -> 7/93
# misclassified -- but the failure set doesn't shrink cleanly, it MOVES
# (zero overlap between the two failure lists, the same catch-all-sink
# pattern documented for ANGLE_LABELS above). More importantly: combined
# with strip_banner (already deployed and already at 4/93), the informative
# prompts made things WORSE, not better -- 7/93 wrong vs. strip_banner's 4/93
# alone. Once the banner pixels are actually gone, a longer/hedgier prompt
# just adds noise; it doesn't have anything left to defend against. Verdict:
# not adopted. strip_banner is sufficient and a reworded SCENE_LABELS would
# be a net regression on top of it.


# Audited across 296 real cutouts: front 37, front_3q 85, side 75,
# rear_3q 67, rear 32, and the labels are right essentially everywhere.
# The exception is VANS. A slab-sided Transit's side, rear three-quarter
# and straight-on rear all read as one long white box to these prompts,
# and three van cutouts are labelled "side" when they are a rear 3/4
# (x2) and a dead-on rear (x1) -- about 1% of the population, and every
# one of them a van.
#
# Deliberately NOT re-prompted. Two reasons:
#
#   * The pan gate does not trust this label alone. A shot only pans if it
#     ALSO overflows the hero box by MIN_PAN_OVERFLOW_FRAC, and all three
#     mislabels fail that independently (aspect 1.62, 1.62 and 0.89
#     against a 1.86:1 bar), so none of them produces a wrong pan. 71 of
#     the 75 "side" labels genuinely pan.
#   * Every attempt in this codebase to fix a label by widening or
#     rewording its prompt has created a new sink somewhere else, usually
#     invisible in the counts. The measured cost here is small and
#     contained: both Transits end up with no rear/rear_3q cutout at all,
#     so imaging/select.py fills their hero's tail accent from a fallback
#     instead of a true rear shot. The heroes still read correctly.
#
# If this is ever revisited, the population to protect is the 296 already
# correct, not the 3 that aren't.
ANGLE_LABELS = {
    "front_3q": "a front three-quarter view of a car, showing the front and one side, taken at an angle",
    "rear_3q": "a rear three-quarter view of a car, showing the back and one side, taken at an angle",
    "side": "a straight side profile view of a car, showing only the side, no front or back visible",
    "front": "a straight-on front view of a car, facing directly toward the camera",
    "rear": "a straight-on rear view of a car, from directly behind, facing away from the camera",
}

# Sub-classifies a SCENE_LABELS "detail" shot -- that label is a broad catch-
# all ("headlight, taillight, wheel, badge, or mirror"), and roughly 2/3 of
# real detail shots turn out NOT to be wheels (calibrated against 170 real
# photos across 17 vehicles: 48 detail shots, only 15 genuine "whole wheel
# visible" wheel photos). A first pass at pure geometric wheel-finding
# (OpenCV Hough circle detection) had a high false-positive rate -- door
# handles, side mirrors, grille badges, and headlights all produced
# circles with edge support just as strong as a real wheel, since geometry
# alone can't know what it's a circle *of*. This semantic pass is what
# actually separates them reliably; a "tread" label is included specifically
# because an extreme-close-up tire-tread shot (rim not visible) reads as
# "wheel and tire" under a looser prompt but isn't the "whole wheel visible"
# shot worth cutting out as a money shot -- see imaging/wheel.py.
WHEEL_DETAIL_LABELS = {
    "wheel": "a close-up photo showing an entire car wheel, the full round rim and tire together as a complete circle",
    "tread": "an extreme close-up of tire tread rubber texture, so close that the wheel rim is not visible",
    "mirror": "a close-up photo of a car side mirror",
    "light": "a close-up photo of a car headlight or taillight",
    "badge": "a close-up photo of a car logo, badge, or grille",
    "handle": "a close-up photo of a car door handle, fuel cap, or window",
    "other": "a close-up photo of some other car part or surface texture",
}

# A second, narrower pass over anything WHEEL_DETAIL_LABELS calls "wheel" --
# a mounted spare (bolted flat to a tailgate/rear door, not touching the
# ground) is a real wheel by that broader label, but isn't the rolling-wheel
# money shot wanted. Kept as its OWN small classifier rather than folded
# into WHEEL_DETAIL_LABELS as an 8th label: a first attempt at one combined
# multi-way classifier confused CLIP badly (a real spare-tire label pulled
# in genuine rolling-wheel shots as false positives, e.g. a clean 3/4 wheel
# shot scored spare_tire(1.00)) -- CLIP zero-shot works far more reliably
# on a small, sharply-contrasted choice than one large one with overlapping
# categories. See imaging/wheel.py for the angle-quality check (front-tread
# vs 3/4 money-shot angle), which turned out to be more reliable as a
# geometric measurement on the segmented cutout than as a CLIP label at all.
SPARE_TIRE_LABELS = {
    "mounted_spare": "a spare tire mounted flat against the outside of a vehicle tailgate or rear door, hanging in the air above the ground",
    "rolling_wheel": "one of a vehicle's regular wheels, mounted underneath the body in a wheel well, touching or near the ground",
}

# A second pass over anything SCENE_LABELS' 5-way choice calls "interior" --
# added after a real production miss (a straight-on studio front shot of a
# Bronco, dealer banner and all, scored interior over exterior). The first
# fix attempt used rembg's segmented-bbox margin as a geometric tiebreak
# (the same technique that already works well for the "marketing" label,
# see imaging/pipeline.py) on the theory that a real exterior photo leaves
# background margin on every side while a cabin shot fills the frame --
# tested clean on a handful of cases, but at full-population scale (225
# real interior photos) it flagged 104 of them (46%) as exterior: rembg
# routinely isolates SOME salient blob inside a busy cabin scene (a
# console, a mirror, a seat) that also happens to leave margin on every
# side, and neither that margin nor the blob's area/aspect ratio reliably
# told a real car body apart from an arbitrary interior detail -- unlike a
# studio exterior shot, cabin photos have no consistent enough framing for
# geometry to key off. Confirms the same lesson WHEEL_DETAIL_LABELS/
# SPARE_TIRE_LABELS already learned: a narrow, sharply-contrasted 2-way
# CLIP choice generalizes far better here than a geometric heuristic --
# this prompt correctly caught only the 1 real bug and 1 genuinely
# ambiguous edge case (a dashboard screen showing a 3D car render) out of
# the same 225, with the edge case scoring meaningfully lower confidence
# (0.55 vs 0.997) -- see INTERIOR_EXTERIOR_TIEBREAK_THRESHOLD in
# imaging/pipeline.py for how that gets used to leave it alone.
INTERIOR_EXTERIOR_TIEBREAK_LABELS = {
    "exterior_body": "a real photograph of the outside of a car, showing its painted exterior body panels, parked or standing on the ground with open space around it",
    "cabin_interior": "a photograph taken from inside a car, showing the dashboard, seats, steering wheel, console, or a screen mounted in the dashboard",
}

# For imaging/compose/hero_video.py's pan effect: which side of a side-
# profile cutout the hood is on, so a pan always travels tail-to-hood
# (reads as the vehicle "arriving") regardless of which way any given
# photo happens to face. Classifying the WHOLE left/right half first
# (an even 50/50 split) washed the signal out -- confirmed on a real pair
# of same-vehicle side shots (04.png/08.png, facing OPPOSITE ways): both
# halves of 08.png scored "rear" (0.999 and 0.917) even though the right
# half plainly contained the grille, because a 50%-wide crop is still
# mostly door/window content that looks identical regardless of facing.
# Narrowing to each end's outer ~22% -- where the grille/headlight or
# taillight/tailgate actually live, with the visually-ambiguous doors and
# windows excluded -- fixed it: the same two images scored front=0.732/
# rear=1.0 (04, hood left) and rear=0.999/front=0.745 (08, hood right),
# both correct and confidently separated.
FACING_EDGE_LABELS = {
    "front": "the front end of a car with headlights and a grille",
    "rear": "the rear end of a car with taillights",
}
FACING_EDGE_FRAC = 0.22


@dataclass
class Classification:
    label: str
    confidence: float
    scores: dict[str, float]


class ClipBackbone:
    """Lazy-loads the CLIP model+preprocessing once; shared by any number of
    ZeroShotClassifier instances so a scene pass and an angle pass don't
    each pay their own ~600MB GPU load."""

    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED, device: str | None = None):
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import open_clip
        import torch

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        try:
            model.eval().to(self._device)
        except torch.cuda.OutOfMemoryError:
            # The GPU is shared with whatever else is on this desktop --
            # a video player holding frames for hardware decode is enough
            # to leave no room. Classification is slower on CPU but it
            # WORKS, and losing an entire otherwise-successful scrape
            # (page, photos, window sticker, all fetched) because another
            # process is holding VRAM is a far worse outcome than a slow
            # one. Same reasoning as the composition steps being wrapped
            # in try/except in vehicle_pipeline.py.
            print(f"    ! GPU out of memory loading {self.model_name}; falling back to CPU")
            torch.cuda.empty_cache()
            self._device = "cpu"
            model.eval().to(self._device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def embed_texts(self, texts: list[str]):
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self._device)
            features = self._model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def embed_image(self, content: bytes):
        import torch
        from PIL import Image

        self._ensure_loaded()
        img = Image.open(io.BytesIO(content))
        if img.mode == "RGBA":
            # Composite onto white first -- rembg's cutouts (used for both
            # scene and angle classification) zero the RGB in fully
            # transparent regions, so a naive .convert("RGB") silently fills
            # them black instead. That's what this was actually validated
            # against (white background), so don't let input transparency
            # change the classification distribution by accident.
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        image = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features


_default_backbone: ClipBackbone | None = None


def default_backbone() -> ClipBackbone:
    global _default_backbone
    if _default_backbone is None:
        _default_backbone = ClipBackbone()
    return _default_backbone


class ZeroShotClassifier:
    """Classifies images against an arbitrary {label: text_prompt} dict,
    using a shared ClipBackbone. Text embeddings are computed once, lazily,
    on first classify() call."""

    def __init__(self, labels: dict[str, str], backbone: ClipBackbone | None = None):
        self.labels = labels
        self.backbone = backbone or default_backbone()
        self._label_names = list(labels.keys())
        self._text_features = None

    def _ensure_text_embedded(self):
        if self._text_features is None:
            self._text_features = self.backbone.embed_texts([self.labels[n] for n in self._label_names])

    def classify(self, content: bytes) -> Classification:
        self._ensure_text_embedded()
        image_features = self.backbone.embed_image(content)
        probs = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0].tolist()
        scores = dict(zip(self._label_names, probs))
        best_label = max(scores, key=scores.get)
        return Classification(label=best_label, confidence=scores[best_label], scores=scores)

    def classify_file(self, path: Path) -> Classification:
        return self.classify(Path(path).read_bytes())


class SceneClassifier(ZeroShotClassifier):
    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(SCENE_LABELS, backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


class AngleClassifier(ZeroShotClassifier):
    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(ANGLE_LABELS, backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


class WheelDetailClassifier(ZeroShotClassifier):
    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(WHEEL_DETAIL_LABELS, backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


class SpareTireClassifier(ZeroShotClassifier):
    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(SPARE_TIRE_LABELS, backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


class InteriorExteriorTiebreakClassifier(ZeroShotClassifier):
    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(INTERIOR_EXTERIOR_TIEBREAK_LABELS,
                          backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


class FacingEdgeClassifier(ZeroShotClassifier):
    """See FACING_EDGE_LABELS -- classifies a narrow EDGE crop (not the
    whole image) as showing a car's front or rear."""

    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(FACING_EDGE_LABELS, backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


def detect_hood_side(content: bytes, classifier: FacingEdgeClassifier | None = None) -> str:
    """'left' or 'right' -- which side of a side-profile (or any
    horizontally-oriented) cutout the hood/front is on. Classifies the
    outer FACING_EDGE_FRAC of each side against "front" vs "rear" and
    picks whichever edge scores "front" higher -- see FACING_EDGE_LABELS
    for why this needs a narrow edge crop, not a 50/50 half split."""
    import io
    from PIL import Image

    classifier = classifier or FacingEdgeClassifier()
    img = Image.open(io.BytesIO(content))
    w, h = img.size
    edge = max(1, round(w * FACING_EDGE_FRAC))

    def _front_score(box):
        buf = io.BytesIO()
        img.crop(box).convert("RGBA").save(buf, format="PNG")
        return classifier.classify(buf.getvalue()).scores["front"]

    left_front = _front_score((0, 0, edge, h))
    right_front = _front_score((w - edge, 0, w, h))
    return "left" if left_front > right_front else "right"
