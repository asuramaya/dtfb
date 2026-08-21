"""
Interior photo post-processing: an exposure fix, plus a feature callout
naming what the shot actually shows.

Interiors get fundamentally different treatment from exteriors, for a
reason established at scale: rembg cannot cut out a cabin. Tested across
225 real interior photos it isolated an arbitrary blob (a console, a
seat, a mirror) on 46% of them, so the "cutout floating on a backdrop"
language the exterior pipeline uses does not transfer. These stay whole
photos at their original resolution and aspect.

Two things are done to them:

  1. EXPOSURE. Cabin shots are taken into a dark interior with blown
     windows behind, and most arrive muddy. This is the only step here
     that improves the content rather than decorating it, and it runs on
     every interior photo.

  2. A FEATURE CALLOUT, matched to the subject. The value is entirely in
     the matching -- "Apple CarPlay" over the infotainment screen, not
     over a shot of the back seats. That needs to know what the photo
     shows, hence InteriorSubjectClassifier, and it needs the feature
     list to actually contain something relevant, hence the two source
     routes below. When either is missing or uncertain the photo simply
     gets the exposure fix and no text: a wrong callout is worse than
     none, since it reads as a lie about the vehicle.

FEATURE SOURCES, two routes. A Ford with a Monroney sticker gives a
dedicated `equipment.interior` list, which is the good one -- but roughly
half this dealer's used inventory is non-Ford (Honda, Tesla, Toyota, VW,
GMC) with no sticker at all, so the site's own scraped feature lists are
the fallback. They differ in quality: sticker text is machine shorthand
("2Nd Row Heated Seats", "Pedals-Pwr Adj W/Memory") and needs
normalizing, while site features arrive close to display-ready ("Apple
CarPlay", "Automatic temperature control").

Text baked into an image carries real risk on this account -- a frame
with contact info in it is what got posts shadowbanned. A short factual
feature name is a different category from contact/promo overlay, but it
is deliberately kept to one small line rather than a banner.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .classify import ClipBackbone, ZeroShotClassifier

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# What is actually DEPICTED, not what region of the cabin this is.
#
# The earlier version classified the region (screen/dashboard/console)
# and then captioned it with any feature belonging to that region. That
# conflates two different things -- what the car HAS versus what this
# photo SHOWS -- and produced exactly the failures you would expect: a
# screen displaying the reversing camera captioned "Apple CarPlay", a
# digital gauge cluster captioned "Automatic temperature control", a gear
# selector captioned "Remote keyless entry". Every one of those is a true
# statement about the car and a false statement about the picture.
#
# So the labels below name things that are VISIBLE, and a caption is only
# printed when the vehicle's own feature data independently confirms it
# (see FEATURE_CONFIRMATION). Two keys, both required.
DEPICTED_FEATURE_LABELS = {
    "backup_camera": "a car dashboard screen showing the rear reversing camera view behind the car, with yellow parking guide lines",
    "phone_mirroring": "a car touchscreen showing a smartphone interface with rows of app icons, Apple CarPlay or Android Auto",
    "navigation_map": "a car touchscreen showing a navigation map with roads and a route",
    "digital_cluster": "a fully digital instrument cluster screen behind a car's steering wheel showing a speedometer and tachometer",
    "climate_controls": "a car's climate control panel with temperature dials and air conditioning buttons",
    "heated_seat_controls": "a car seat heater button showing the icon of a seat with wavy heat lines through it",
    # Generic switchgear needs its own home for the same reason "none"
    # does: without it, every anonymous button close-up piled into
    # heated_seat_controls (8 of 19 photos on the first vehicle, one at
    # 0.97). It confirms against nothing, so it never captions.
    "misc_controls": "an assortment of unlabelled buttons, switches or a gear selector on a car's interior panel",
    "sunroof": "an open sunroof or moonroof seen from inside a car",
    "leather_seats": "leather upholstered car seats with visible stitching",
    # Added after "none" became the sink in turn: it was taking 136 of the
    # fleet's interior photos and 8 of 9 on some vehicles, which is the
    # same diagnostic that exposed door_panel and heated_seat_controls
    # earlier -- one label winning a suspiciously large share is absorbing
    # things it has no name for. Looking at what it held: cloth upholstery,
    # rear benches, wide dashboard views and bare steering wheels. Naming
    # those four dropped "none" to 70 and took the fleet from 51 captions
    # on 20 vehicles to 69 on 22.
    #
    # cloth_seats earns its place as the counterweight to leather_seats
    # rather than as a caption -- without it, cloth reads as leather (a
    # base Transit's fabric seat scored leather_seats 0.85), and only the
    # confirmation gate was stopping that becoming a false "Leather Seats"
    # on a vinyl van.
    "cloth_seats": "cloth or woven fabric car seat upholstery, matte textile with no leather sheen",
    "rear_seats": ("the back seat bench behind the front seats of a car, with the rear of the "
                    "front seatbacks visible in front of it"),
    "dashboard": "a wide view of a car's whole dashboard and front cabin seen from the passenger side",
    # Distinct from steering_controls on purpose: that one is the buttons,
    # this is the wheel itself. Splitting them fixed a real false caption
    # -- a bare Tesla wheel, which has no buttons at all, was captioned
    # "Steering wheel mounted A/C controls".
    "steering_wheel": "a car steering wheel seen whole, showing its rim and centre hub emblem",
    "cargo_area": "the open empty trunk or cargo area of a car",
    "wireless_charging": "a wireless phone charging pad in a car's centre console",
    "usb_ports": "a close-up of USB sockets in a car, showing the USB trident symbol",
    "steering_controls": "control buttons mounted on the spokes of a car's steering wheel",
    "none": "an ordinary interior trim surface, seat or panel with no notable equipment visible",
}

# Recognized thing -> the feature words that must ALSO appear in the
# vehicle's own feature list for the caption to be allowed. The printed
# text is the vehicle's own wording, not ours, so the caption can never
# claim something the listing does not.
FEATURE_CONFIRMATION: dict[str, list[str]] = {
    "backup_camera": ["back-up camera", "backup camera", "rear view camera", "rearview camera"],
    "phone_mirroring": ["apple carplay", "carplay", "android auto"],
    "navigation_map": ["navigation"],
    "digital_cluster": ["digital instrument", "instrument cluster", "driver information"],
    "climate_controls": ["automatic temperature control", "dual climate", "climate control",
                         "air conditioning"],
    "heated_seat_controls": ["heated front seat", "heated seat", "2nd row heated"],
    "sunroof": ["sunroof", "moonroof"],
    "leather_seats": ["leather seat", "leather-trimmed", "leather upholstery"],
    "cargo_area": ["cargo", "trunk"],
    "wireless_charging": ["wireless charging"],
    "usb_ports": ["usb"],
    "steering_controls": ["steering wheel mounted", "steering wheel audio"],
    "rear_seats": ["split folding rear seat", "60/40", "split-folding", "third row",
                    "rear bench", "fold-flat"],
    "steering_wheel": ["leather-wrapped steering wheel", "leather steering wheel",
                        "heated steering wheel", "tilt steering", "telescoping steering"],
    "misc_controls": [],
    "cloth_seats": [],
    "dashboard": [],
    "none": [],
}

# A gear_selector label was tried here and removed: it confirms against
# nothing (misc_controls already names a gear selector), and it stole two
# correct "Steering wheel mounted audio controls" captions off photos with
# no gear selector anywhere in frame, one at 0.87. A label that can only
# lose is worse than no label.

# Higher than the old region threshold: this call now decides whether we
# assert a specific piece of equipment is pictured, so it should abstain
# readily. Most interior photos legitimately depict nothing nameable.
DEPICTION_CONFIDENCE_THRESHOLD = 0.55

# Labels that exist purely to absorb the unnameable -- they never
# produce a caption, they just stop other labels from being a sink.
ABSTAIN_LABELS = {"none", "misc_controls", "cloth_seats", "dashboard"}

# Sticker shorthand -> display copy. The Monroney text is machine
# abbreviated and TitleCased by the parser ("2Nd Row Heated Seats",
# "Pedals-Pwr Adj W/Memory"), which reads as scraped rather than
# authored if pasted straight onto a photo.
_ABBREVIATIONS = [
    # No \bAuto\b -> Automatic rule: it corrupts "Android Auto", a proper
    # noun, and the sticker's own "Auto" abbreviations are rare enough
    # not to be worth breaking a headline feature name over.
    (r"\bPwr\b", "Power"), (r"\bStr\b", "Steering"), (r"\bAdj\b", "Adjustable"),
    (r"\bW/\s*", "with "), (r"\bA/C\b", "A/C"),
    (r"\bNd\b", "nd"), (r"\bRd\b", "rd"), (r"\bTh\b", "th"), (r"\bSt\b", "st"),
    (r"\bLed\b", "LED"), (r"\bUsb\b", "USB"), (r"\bAm/Fm\b", "AM/FM"),
]


def normalize_feature(text: str) -> str:
    """Sticker/site shorthand into something printable."""
    out = " ".join(text.split())
    for pattern, replacement in _ABBREVIATIONS:
        out = re.sub(pattern, replacement, out)
    out = re.sub(r"\s+", " ", out).strip(" -–—:;,")
    # "Radio: 160-Watt Audio System" -> the part worth reading
    if ":" in out:
        head, _, tail = out.partition(":")
        if len(tail.strip()) >= 6:
            out = tail.strip()
    return out


def collect_features(vehicle: dict) -> list[str]:
    """Candidate feature names, sticker route first (see module
    docstring). Order matters: the first keyword match wins, so the
    richer sticker list leads when it exists."""
    features: list[str] = []
    sticker = vehicle.get("sticker") or {}
    equipment = sticker.get("equipment") or {}
    for key in ("interior", "functional_tech"):
        features += list(equipment.get(key) or [])

    features += list(vehicle.get("main_features") or [])
    for values in (vehicle.get("features_structured") or {}).values():
        features += list(values or [])

    seen, unique = set(), []
    for f in features:
        norm = normalize_feature(str(f))
        key = norm.lower()
        if norm and key not in seen:
            seen.add(key)
            unique.append(norm)
    return unique


def confirm_feature(depicted: str, features: list[str]) -> str | None:
    """The vehicle's own wording for `depicted`, or None if its feature
    list does not corroborate it. Returning None is the common, correct
    outcome -- a photo of a gear selector depicts nothing the listing
    advertises, so it ships without text."""
    lowered = [(f, f.lower()) for f in features]
    for keyword in FEATURE_CONFIRMATION.get(depicted, []):
        for original, low in lowered:
            if keyword in low:
                return original
    return None


class InteriorSubjectClassifier(ZeroShotClassifier):
    """What equipment an interior photo actually DEPICTS -- see
    DEPICTED_FEATURE_LABELS. Same narrow-choice CLIP pattern as
    WheelDetailClassifier/SpareTireClassifier, including the deliberate
    "none" catch-all: without one, everything unnameable piles into the
    nearest real label (a door-panel sink swallowed a USB tray, an
    overhead console and a set of rear vents before one was added)."""

    def __init__(self, backbone: ClipBackbone | None = None, **kwargs):
        super().__init__(DEPICTED_FEATURE_LABELS,
                          backbone=backbone or (ClipBackbone(**kwargs) if kwargs else None))


# Where the exposure lift starts backing off, and where it stops entirely.
# Measured over 394 real interior photos: 252 of them (64%) already have
# more than 2% of their pixels clipped at white, median 3.9% and p95 15.7%
# -- the windows. A lift applied flat across the frame cannot help those
# pixels (they are already at 1.0) and actively harms the ones just below,
# pushing near-white into clipping and washing the glass out further. So
# the lift is faded out over the top end instead: full strength through
# the shadows and midtones where the cabin actually lives, nothing at all
# where the glass is.
HIGHLIGHT_KNEE = 0.55
HIGHLIGHT_CEILING = 0.92

# White balance is estimated from bright NEAR-NEUTRAL pixels only, not
# grey-world. Grey-world assumes the average of the scene is grey, which
# is exactly wrong for a cabin: a King Ranch's saddle leather or a tan
# interior would be read as a colour cast and neutralised into mud. Bright
# near-neutral pixels -- daylight through the glass, a light headliner,
# specular highlights on trim -- are a far safer illuminant estimate for
# this content.
WHITE_BALANCE_MIN_CAST = 0.06  # below this, leave it alone
WHITE_BALANCE_STRENGTH = 0.6   # correct most of the way, never all
WHITE_BALANCE_MAX_GAIN = 1.25  # hard cap per channel


def enhance_exposure(img: Image.Image, target_median: float = 0.46,
                      max_lift: float = 0.55) -> Image.Image:
    """Lift a dark cabin toward a readable midtone, holding the highlights.

    Gamma rather than a shadow-only curve: a cabin is dark almost
    everywhere except the windows, so a shadow-targeted lift leaves the
    mid-dark trim flat. Gamma brightens the low end hardest, which is the
    shape this content wants.

    The lift is then faded out across HIGHLIGHT_KNEE..HIGHLIGHT_CEILING
    using the ORIGINAL luminance, so a pixel's treatment is decided by how
    bright it started rather than by how bright the lift made it -- see the
    constants above for the measurement that made this necessary.

    Only ever brightens (gamma <= 1). A correctly-exposed interior comes
    through untouched; darkening a photo the dealer chose is not this
    function's business.
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    luminance = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    median = float(np.median(luminance))
    if median <= 0.001 or median >= target_median:
        return img

    gamma = max(max_lift, np.log(target_median) / np.log(median))
    lifted = np.power(arr, gamma)

    # smoothstep from 0 (apply the lift fully) to 1 (leave the pixel alone)
    t = np.clip((luminance - HIGHLIGHT_KNEE) / (HIGHLIGHT_CEILING - HIGHLIGHT_KNEE), 0.0, 1.0)
    protect = (t * t * (3.0 - 2.0 * t))[..., None]
    blended = lifted * (1.0 - protect) + arr * protect
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype("uint8"), mode="RGB")


def correct_white_balance(img: Image.Image) -> Image.Image:
    """Take a colour cast off the cabin, estimated from bright near-neutral
    pixels only. Returns `img` unchanged when there's no meaningful cast,
    so this is always safe to call.

    Measured on the fleet: 129 of 394 interior photos carry a cast above
    WHITE_BALANCE_MIN_CAST, 23 of them above 0.20 -- mixed showroom
    lighting through tinted glass. The correction is deliberately partial
    and capped: a cabin lit warm is partly a real property of the car, and
    over-correcting reads as a filter.
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    brightness = flat.max(axis=1)
    spread = flat.max(axis=1) - flat.min(axis=1)

    # Bright, but not blown, and not strongly coloured -- coloured pixels
    # are the ones that would drag a tan interior toward grey.
    bright_enough = brightness >= np.percentile(brightness, 80)
    neutralish = spread <= 0.18 * np.maximum(brightness, 1e-6) + 0.06
    sample = flat[bright_enough & neutralish & (brightness < 0.99)]
    if len(sample) < 200:
        return img

    means = sample.mean(axis=0)
    if means.min() <= 1e-6:
        return img
    cast = float((means.max() - means.min()) / means.mean())
    if cast < WHITE_BALANCE_MIN_CAST:
        return img

    gains = np.clip(means.mean() / means, 1.0 / WHITE_BALANCE_MAX_GAIN, WHITE_BALANCE_MAX_GAIN)
    gains = 1.0 + (gains - 1.0) * WHITE_BALANCE_STRENGTH
    return Image.fromarray(
        (np.clip(arr * gains, 0, 1) * 255).astype("uint8"), mode="RGB")


def enhance_interior(img: Image.Image) -> Image.Image:
    """The whole non-destructive treatment: neutralise, then lift.

    White balance first so the exposure lift measures a neutral image --
    running it the other way lets a strong cast bias the median the gamma
    is solved against. Nothing here crops, composites, or adds anything to
    the frame; the photo that comes out is the photo that went in, exposed
    and neutralised.
    """
    return enhance_exposure(correct_white_balance(img))


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_callout(img: Image.Image, text: str, scrim_frac: float = 0.22,
                  text_frac: float = 0.040) -> Image.Image:
    """One line of text along the bottom, over a gradient scrim.

    The scrim is a gradient rather than a solid bar so it reads as
    lighting rather than as a UI chrome strip pasted over the photo, and
    so it stays legible whatever happens to be behind it -- interior
    shots have no reliable dark region to place text against.
    """
    import numpy as np

    img = img.convert("RGB")
    w, h = img.size
    band = max(1, int(h * scrim_frac))

    arr = np.asarray(img, dtype=np.float32)
    ramp = np.linspace(0.0, 0.72, band, dtype=np.float32) ** 1.6
    arr[h - band:] *= (1.0 - ramp)[:, None, None]
    out = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")

    draw = ImageDraw.Draw(out)
    font = _font(max(12, int(h * text_frac)))
    pad = int(w * 0.035)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((pad, h - pad - (bbox[3] - bbox[1]) - bbox[1]), text, font=font, fill=(255, 255, 255))
    return out


def assign_callouts(subjects: list[tuple[str, str, float]], features: list[str]) -> dict[str, str]:
    """{photo_name: callout} across a WHOLE gallery, using each feature at
    most once.

    Per-photo matching alone produced "Power Windows" on 8 of 19 photos
    of one Accord -- every door/console close-up hitting the same first
    keyword. Repeating one caption across a gallery is exactly the
    duplication these images are supposed to avoid, so allocation is
    global: highest-confidence subject calls get first pick, and a photo
    whose subject's features are already spent simply ships without text.
    """
    used: set[str] = set()
    assigned: dict[str, str] = {}
    for name, depicted, confidence in sorted(subjects, key=lambda s: -s[2]):
        if depicted in ABSTAIN_LABELS or confidence < DEPICTION_CONFIDENCE_THRESHOLD:
            continue
        match = confirm_feature(depicted, features)
        if match and match not in used:
            assigned[name] = match
            used.add(match)
    return assigned


def process_interior(path: Path, features: list[str],
                      classifier: InteriorSubjectClassifier | None) -> tuple[Image.Image, dict]:
    """(image, {subject, confidence, callout}) for one interior photo.
    Always returns an image -- the exposure fix is unconditional, the
    callout is not."""
    content = Path(path).read_bytes()
    img = enhance_exposure(Image.open(path))
    info: dict = {"subject": None, "confidence": None, "callout": None}

    if classifier is None or not features:
        return img, info

    result = classifier.classify(content)
    info["subject"], info["confidence"] = result.label, round(result.confidence, 3)
    if result.label in ABSTAIN_LABELS or result.confidence < DEPICTION_CONFIDENCE_THRESHOLD:
        return img, info

    callout = confirm_feature(result.label, features)
    if not callout:
        return img, info

    info["callout"] = callout
    return draw_callout(img, callout), info


def process_gallery(interior_dir, out_dir, vehicle: dict,
                     classifier: InteriorSubjectClassifier | None = None,
                     captions: bool = False) -> list[dict]:
    """Post-process a vehicle's whole interior gallery into out_dir.

    The default output is the photograph itself, corrected and nothing
    more -- no text, no compositing, no crop, original resolution. That is
    a deliberate narrowing: the feature-callout path below works and is
    kept, but printing text into a listing photo is a merchandising
    decision rather than an image-processing one, so it is opt-in
    (captions=True) rather than something the pipeline does on its own.

    With captions on, two passes: classify everything first, then allocate
    across the gallery (see assign_callouts) -- a per-photo decision
    cannot know a caption has already been used eight times. Every photo
    is written either way.
    """
    interior_dir, out_dir = Path(interior_dir), Path(out_dir)
    photos = sorted(interior_dir.glob("*.jpg")) + sorted(interior_dir.glob("*.png"))
    if not photos:
        return []

    callouts: dict[str, str] = {}
    by_name: dict[str, tuple] = {}
    if captions and classifier is not None:
        features = collect_features(vehicle)
        subjects: list[tuple[str, str, float]] = []
        if features:
            for path in photos:
                result = classifier.classify(path.read_bytes())
                subjects.append((path.name, result.label, result.confidence))
        callouts = assign_callouts(subjects, features)
        by_name = {n: (s, c) for n, s, c in subjects}

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in photos:
        img = enhance_interior(Image.open(path))
        callout = callouts.get(path.name)
        if callout:
            img = draw_callout(img, callout)
        dest = out_dir / f"{path.stem}.jpg"
        img.save(dest, quality=92)
        subject, confidence = by_name.get(path.name, (None, None))
        written.append({"file": dest.name, "subject": subject,
                        "confidence": round(confidence, 3) if confidence else None,
                        "callout": callout})
    return written
