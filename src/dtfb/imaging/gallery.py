"""
Gallery-level check that every cutout is a cutout OF THIS VEHICLE.

Every other classifier in imaging/ asks "what is in this photo?". None of
them asks "is the thing in this photo the car we're selling?", and that
gap produced two real production failures:

  1. A Volkswagen Atlas photo shot from the driver's seat, looking out
     through the windshield at the dealer lot. rembg fused the rear-view
     mirror with a silver Hyundai parked outside into ONE connected blob
     (verified: single component at every alpha threshold), and that blob
     became the hero shot of the vehicle's still AND its video.
  2. A Ford Transit cargo-bay photo whose ghostly open rear doors were cut
     out as if they were a vehicle body.

Neither is detectable by asking a question about the photo's content,
because nothing is wrong with the content. The Atlas photo really is
dominated by car exteriors -- six of them, none an Atlas. The defect is
REFERENTIAL, not semantic: it is a fine picture of the wrong car. So the
only test that can see it is one that has a referent, which is what this
module supplies -- it never asks "what is this?", only "is this the same
thing as the vehicle's other cutouts?".

Five other approaches were measured and rejected first. Recording them
because each looked correct until it was measured against real photos:

  - Adding "cargo_bay"/"through_glass" to SCENE_LABELS' 5-way choice.
    Caught both failures, but cargo_bay is a WHITE-VAN sink: 16 of the 18
    photos it claimed were clean Transit body shots, including the best
    hero photo either van had (front 3/4, scored cargo_bay 0.898). Would
    have left both Transits with zero exterior cutouts. through_glass ate
    5 of 8 -- exterior mirror close-ups shot from outside, dragged in by
    the word "mirror" in its own prompt. Same lesson WHEEL_DETAIL_LABELS
    and SPARE_TIRE_LABELS already record: widening a broad multi-way
    choice makes a new catch-all, and the counts won't reveal it -- only
    looking at the images will.
  - CLIP on the CUTOUT with a "whole vehicle" prompt. Inverted: both bad
    blobs scored 0.71/0.88 while a dozen flawless cutouts scored
    0.05-0.21, because "the entire vehicle body visible from bumper to
    bumper" grades VIEWPOINT -- rear and side profiles can't show both
    bumpers and sank, while the Hyundai in the mirror blob is a textbook
    front 3/4 and floated. No prompt fixes this: that blob genuinely
    contains a complete car, so "is there a whole vehicle here?" is
    truthfully yes.
  - Camera-position ("inside" vs "outside") over cutout candidates.
    Nailed the Transit (0.998) but scored the Atlas 0.117 -- outside,
    which is arguably right, the frame IS mostly outdoors -- and called
    49 of 156 legitimate cutouts "inside", almost all front-3/4 studio
    shots where the cabin is visible through the windshield.
  - Self-consistency normalised PER GALLERY, to avoid a magic constant.
    Both variants are worse than the raw value: MAD z-score fails
    outright (a good Mustang cutout becomes the biggest outlier in the
    fleet at -8.79, because a tight gallery drives MAD toward zero), and
    ratio-to-median shrinks the genuine/impostor margin from 0.042 to
    0.012.
  - Largest-gap dynamic splitting, same motive. Four CLEAN galleries
    split harder than the two bad ones (Camry 5.37x, Expedition 3.94x vs
    2.76x/2.91x for the real failures).

The last three all fail for one reason, and it is the reason this module
needs TWO keys rather than one: they measure distance from the gallery,
and a legitimate dead-on rear or side shot is also distant from a gallery
otherwise made of 3/4 views. Self-consistency alone cannot separate "a
different angle of the same car" from "a different car".

Paint colour can, because it is angle-invariant -- the property the
embedding lacks. So a cutout is only demoted when BOTH agree: it doesn't
look like its siblings AND it isn't painted like them. The two keys are
independent in practice, which is what makes the rule safe to act on
unsupervised.

Calibrated by cross-gallery injection over the 21-vehicle fleet: score
every cutout against every OTHER vehicle's gallery, giving 3068 real
impostors instead of the 2 reported failures. At the thresholds below:

    genuine cutouts demoted     0 / 154   (0.0%)
    realistic impostors caught  29.9% of 3068
    the 2 reported failures     2 / 2     (0.561/188.1 and 0.588/136.3)

Zero false positives held across every operating point from 0.65/100 to
0.80/120, and no genuine cutout comes close to failing both keys: the
worst by colour is 0.847/120.2 (similarity nowhere near the line), the
worst by similarity is 0.753/101.3 (colour fine).

Two limits worth knowing. Same-MODEL twins are invisible -- two white
Transits of the same generation score 0.955 against each other -- so the
signal's real ceiling is "is this the same model", not "the same car";
that is harmless here only because a scrape pulls from one VDP, so a
different Transit can never enter this Transit's gallery. And the
impostors this misses are the ones whose paint happens to match the
target vehicle, which is also the least visually jarring failure: a
silver blob in a silver car's gallery doesn't read as broken, while a
black mirror and a white Hyundai in a white Atlas's gallery does.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

# Below this many cutouts the check abstains entirely rather than guessing.
# Both keys are leave-one-out measurements against the siblings, so on a
# 3-cutout gallery the embedding reference is a mean of 2 and the colour
# reference is a median of 2 -- neither is a consensus. The calibration
# above deliberately excluded galleries under this size.
MIN_GALLERY_FOR_CHECK = 4

# Two ways a cutout can be rejected, because one threshold could not do
# both jobs. Measured over 185 genuine cutouts and 4388 injected impostors:
#
#   sim < 0.75 alone            12/185 genuine wrongly demoted  -- unusable
#   sim < 0.75 AND colour       0/185,  29.1% of impostors caught
#   sim < 0.58 alone            0/185,  54.2%
#   either of the two above     0/185,  66.5%
#
# So DOUBTFUL is the level where the embedding is merely unsure and needs
# the colour key to corroborate; CERTAIN is the level where the gallery
# disowns a cutout so completely that no second opinion is needed. Taking
# the union of the two roughly doubles what gets caught at the same
# measured zero false-positive rate.
DOUBTFUL_SIMILARITY = 0.75
CERTAIN_SIMILARITY = 0.58

# Back-compat alias for the single-threshold name this module used before
# the union rule.
SIMILARITY_FLOOR = DOUBTFUL_SIMILARITY

# Euclidean RGB distance from the median paint colour of the other
# cutouts, via palette.sample_cutout_color(). Genuine cutouts top out at
# 120.2 across the fleet; the two real failures sit at 136.3 and 188.1.
COLOR_DISTANCE_CEILING = 110.0

# Safety valve: if more than half a gallery fails, the siblings forming
# the reference are themselves suspect and demoting the minority would be
# backwards. Abstain and let a human look instead -- this module exists to
# remove an obvious interloper from a healthy gallery, not to adjudicate a
# gallery that has gone wrong wholesale.
MAX_DEMOTE_FRACTION = 0.5

# CERTAIN_SIMILARITY above is a measurement of THIS dealer's photos, not a
# law, so calibrate() re-derives it from whatever gallery is actually on
# disk and writes the answer here for the scrape path to pick up. The
# method is the one Beta-SOD uses for noisy re-ID galleries (arXiv
# 2509.08926): fit a Beta to each population -- within-gallery pairs are
# "same vehicle", cross-gallery pairs are "different vehicle" -- and put
# the boundary where the impostor density overtakes the genuine one.
#
# Three deliberate choices on top of that:
#
#   * The boundary is weighted by CALIBRATION_PRIOR, an explicit statement
#     of how often a foreign cutout is expected, because "assume at most 1
#     cutout in 100 doesn't belong" is a claim about the world that can be
#     argued with. A bare similarity number is not.
#   * The fit is bootstrapped and a LOW percentile is taken rather than
#     the point estimate. On this fleet the point estimate lands at 0.597
#     with sd 0.014, and 1 resample in 40 put it ABOVE the worst genuine
#     cutout (0.625) -- close enough to matter. The 10th percentile lands
#     at 0.581, restoring a 0.044 margin.
#   * Same-model pairs are excluded from the impostor population. Two
#     white Transits of the same generation score 0.955 against each other
#     and would drag the boundary up over nothing: a scrape pulls one VDP,
#     so a different Transit can never enter this Transit's gallery.
CALIBRATION_FILENAME = "cutout-calibration.json"
CALIBRATION_PRIOR = 0.01
CALIBRATION_BOOTSTRAPS = 60
CALIBRATION_PERCENTILE = 10
MIN_CALIBRATION_GALLERIES = 4


@dataclass
class ForeignCutout:
    """One cutout the gallery does not vouch for."""
    name: str
    similarity: float
    color_distance: float
    reason: str = "doubtful+colour"

    def describe(self) -> str:
        if self.reason == "certain":
            return (f"{self.name}: resembles the vehicle's other shots at only "
                    f"{self.similarity:.2f}, below the level where the gallery disowns it outright")
        return (f"{self.name}: resembles the vehicle's other shots at only "
                f"{self.similarity:.2f} (below {DOUBTFUL_SIMILARITY}) and its paint is "
                f"{self.color_distance:.0f} from theirs (above {COLOR_DISTANCE_CEILING:.0f})")


def _embed_on_white(path: Path, backbone):
    """CLIP image embedding of the cutout flattened onto white.

    CLIP has never seen a transparent PNG; handing it raw RGBA lets the
    alpha channel be interpreted as arbitrary colour. White matches the
    background these cutouts are composited over downstream.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    flat = Image.new("RGB", img.size, (255, 255, 255))
    flat.paste(img, mask=img.split()[3])
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    vec = np.asarray(backbone.embed_image(buf.getvalue()).detach().cpu(), dtype=np.float64).ravel()
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _gallery_embeddings(cutout_dir: Path, backbone):
    """(paths, embeddings, colours) for one gallery, dropping any cutout
    whose paint can't be sampled -- both keys are required, so one that can
    never fail the colour key shouldn't skew the reference it feeds."""
    import numpy as np

    from .palette import sample_cutout_color

    paths, colors = [], []
    for p in sorted(Path(cutout_dir).glob("*.png")):
        try:
            c = sample_cutout_color(p)
        except Exception:
            c = None
        if c is not None:
            paths.append(p)
            colors.append(np.array(c, dtype=np.float64))
    if not paths:
        return [], None, None
    return paths, np.stack([_embed_on_white(p, backbone) for p in paths]), np.stack(colors)


def _fit_beta(values):
    """Maximum-likelihood Beta fit. Cosine similarities are already on
    [0, 1] so no rescaling is needed."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import beta as beta_dist

    x = np.clip(np.asarray(values, dtype=np.float64), 1e-4, 1 - 1e-4)
    result = minimize(lambda p: -beta_dist.logpdf(x, np.exp(p[0]), np.exp(p[1])).sum(),
                       [np.log(5.0), np.log(2.0)], method="Nelder-Mead",
                       options={"maxiter": 4000})
    return float(np.exp(result.x[0])), float(np.exp(result.x[1]))


def _density_crossing(genuine, impostor, prior: float) -> float:
    """The similarity below which a cutout is more likely foreign than
    genuine, given `prior` as the expected rate of foreign cutouts."""
    import numpy as np
    from scipy.stats import beta as beta_dist

    ag, bg = _fit_beta(genuine)
    ai, bi = _fit_beta(impostor)
    grid = np.linspace(0.002, 0.998, 6000)
    foreign_wins = grid[beta_dist.logpdf(grid, ai, bi) + np.log(prior)
                         >= beta_dist.logpdf(grid, ag, bg) + np.log(1.0 - prior)]
    return float(foreign_wins.max()) if len(foreign_wins) else 0.0


def _model_key(vehicle_folder_name: str) -> str:
    """make/model, with the year and stock number stripped -- used only to
    keep same-model pairs out of the impostor population."""
    import re

    return "-".join(re.sub(r"^\d{4}-", "", vehicle_folder_name).split("-")[:3]).lower()


def calibrate(listings_root: Path, backbone=None, prior: float = CALIBRATION_PRIOR) -> dict | None:
    """Re-derive CERTAIN_SIMILARITY from the galleries actually on disk.

    Returns the calibration record (also written to
    <listings_root>/cutout-calibration.json), or None when there aren't
    enough galleries to form both populations -- in which case callers keep
    the measured default, which is the right answer for a fresh install
    rather than a reason to fail.
    """
    import json

    import numpy as np

    if backbone is None:
        from .classify import default_backbone
        backbone = default_backbone()

    listings_root = Path(listings_root)
    galleries = {}
    for cutout_dir in sorted(listings_root.glob("*/*/images/exterior/cutout")):
        paths, embeddings, colors = _gallery_embeddings(cutout_dir, backbone)
        if len(paths) >= MIN_GALLERY_FOR_CHECK:
            galleries[cutout_dir.parents[2].name] = (embeddings, colors)
    if len(galleries) < MIN_CALIBRATION_GALLERIES:
        return None

    def unit(v):
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    genuine, impostor = [], []
    for name, (embeddings, _colors) in galleries.items():
        for i in range(len(embeddings)):
            genuine.append(float(embeddings[i] @ unit(np.delete(embeddings, i, axis=0).mean(axis=0))))
        reference = unit(embeddings.mean(axis=0))
        for other, (other_emb, _c) in galleries.items():
            if other == name or _model_key(other) == _model_key(name):
                continue
            impostor.extend(float(other_emb[j] @ reference) for j in range(len(other_emb)))
    if len(genuine) < 20 or len(impostor) < 100:
        return None

    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    rng = np.random.default_rng(0)
    boots = np.array([
        _density_crossing(genuine[rng.choice(len(genuine), len(genuine), replace=True)],
                           impostor[rng.choice(len(impostor), min(len(impostor), 1500), replace=True)],
                           prior)
        for _ in range(CALIBRATION_BOOTSTRAPS)
    ])
    floor = float(np.percentile(boots, CALIBRATION_PERCENTILE))

    record = {
        "certain_similarity": round(floor, 4),
        "prior": prior,
        "percentile": CALIBRATION_PERCENTILE,
        "bootstrap_mean": round(float(boots.mean()), 4),
        "bootstrap_sd": round(float(boots.std()), 4),
        "n_galleries": len(galleries),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "worst_genuine_similarity": round(float(genuine.min()), 4),
        "margin_to_worst_genuine": round(float(genuine.min() - floor), 4),
    }
    (listings_root / CALIBRATION_FILENAME).write_text(json.dumps(record, indent=2))
    return record


def load_calibration(start: Path) -> float | None:
    """The calibrated CERTAIN_SIMILARITY for a vehicle folder, found by
    walking up to the listings root. None when nothing has been calibrated
    yet -- callers fall back to the measured default."""
    import json

    path = Path(start).resolve()
    for candidate in [path, *path.parents][:6]:
        record_path = candidate / CALIBRATION_FILENAME
        if record_path.is_file():
            try:
                value = json.loads(record_path.read_text()).get("certain_similarity")
                return float(value) if value is not None else None
            except (OSError, ValueError, TypeError):
                return None
    return None


def find_foreign_cutouts(cutout_dir: Path, backbone=None,
                          similarity_floor: float = DOUBTFUL_SIMILARITY,
                          certain_similarity: float | None = None,
                          color_ceiling: float = COLOR_DISTANCE_CEILING) -> list[ForeignCutout]:
    """Cutouts this gallery does not vouch for, worst first.

    Returns [] -- meaning "no opinion", not "all clean" -- when the
    gallery is too small to form a consensus, when colours can't be
    sampled, or when the safety valve trips. Callers should treat an empty
    result as "change nothing".
    """
    import numpy as np

    cutout_dir = Path(cutout_dir)
    if not cutout_dir.is_dir() or len(list(cutout_dir.glob("*.png"))) < MIN_GALLERY_FOR_CHECK:
        return []

    if backbone is None:
        from .classify import default_backbone
        backbone = default_backbone()

    if certain_similarity is None:
        certain_similarity = load_calibration(cutout_dir)
        if certain_similarity is None:
            certain_similarity = CERTAIN_SIMILARITY

    paths, embeddings, palette = _gallery_embeddings(cutout_dir, backbone)
    if len(paths) < MIN_GALLERY_FOR_CHECK:
        return []

    found = []
    for i, path in enumerate(paths):
        others = np.delete(embeddings, i, axis=0).mean(axis=0)
        norm = np.linalg.norm(others)
        similarity = float(embeddings[i] @ (others / norm)) if norm else 1.0
        distance = float(np.linalg.norm(palette[i] - np.median(np.delete(palette, i, axis=0), axis=0)))
        # Union: the gallery disowns it outright, or merely doubts it AND
        # the paint disagrees too.
        certain = similarity < certain_similarity
        doubtful = similarity < similarity_floor and distance > color_ceiling
        if certain or doubtful:
            found.append(ForeignCutout(path.name, round(similarity, 4), round(distance, 1),
                                        reason="certain" if certain else "doubtful+colour"))

    if len(found) > len(paths) * MAX_DEMOTE_FRACTION:
        return []

    found.sort(key=lambda f: f.similarity)
    return found


def demote_foreign_cutouts(vehicle_folder: Path, backbone=None,
                            dry_run: bool = False) -> list[ForeignCutout]:
    """Apply find_foreign_cutouts() to a vehicle folder already on disk.

    photos.py does this inline during a scrape, where the photo bytes are
    still in hand; this is the same decision applied after the fact, so a
    folder scraped before the check existed can be fixed without paying
    for the whole pipeline again. Deletes the cutout and its stale
    bundle/framed/ composition, and MOVES the photo into images/interior/
    rather than deleting it -- it's a real photo that simply shouldn't have
    been cut out.

    One difference from the scrape path: a photo moved here misses the
    batch letterbox pass the rest of the interior gallery went through
    (imaging/letterbox.py runs once over the whole batch at download
    time). Harmless for this dealer, whose galleries are confirmed
    bar-free, but it would show as an uncropped photo among cropped ones
    on a vendor that pads them.
    """
    vehicle_folder = Path(vehicle_folder)
    cutout_dir = vehicle_folder / "images" / "exterior" / "cutout"
    exterior_dir = vehicle_folder / "images" / "exterior"
    interior_dir = vehicle_folder / "images" / "interior"

    found = find_foreign_cutouts(cutout_dir, backbone=backbone)
    if not found or dry_run:
        return found

    import json

    angles_path = cutout_dir / "angles.json"
    try:
        angles = json.loads(angles_path.read_text())
    except (OSError, ValueError):
        angles = None

    existing = [int(p.stem) for p in interior_dir.glob("*.*") if p.stem.isdigit()]
    next_interior = max(existing, default=0) + 1

    for f in found:
        stem = Path(f.name).stem
        photos = [p for p in exterior_dir.glob(f"{stem}.*") if p.is_file()]
        interior_dir.mkdir(parents=True, exist_ok=True)
        for photo in photos:
            photo.rename(interior_dir / f"{next_interior:02d}{photo.suffix}")
            next_interior += 1
        (cutout_dir / f.name).unlink(missing_ok=True)
        # The solo composition built from this cutout is now orphaned; the
        # hero/video get rebuilt wholesale by the caller, but framed/ is
        # one file per cutout and would otherwise keep the stale one.
        (vehicle_folder / "bundle" / "framed" / f.name).unlink(missing_ok=True)
        if angles is not None:
            angles.pop(f.name, None)

    if angles is not None:
        angles_path.write_text(json.dumps(angles, indent=2))

    return found


@dataclass
class ReclassifiedPhoto:
    """One images/interior/ photo the CURRENT pipeline disagrees with --
    and what happened when re-promoting it to exterior was attempted."""
    name: str                    # original interior/ filename, e.g. "07.jpg"
    outcome: str                 # see resweep_interior_gallery()'s docstring for the outcome set
    detail: str = ""
    new_name: str | None = None  # exterior/cutout/<new_name> if outcome == "promoted"

    def describe(self) -> str:
        if self.outcome == "promoted":
            return f"{self.name} -> exterior/cutout/{self.new_name}"
        if self.outcome == "rejected_by_gallery":
            return (f"{self.name}: current classifier now calls this exterior, but the gallery "
                     f"consensus check would demote it right back ({self.detail}) -- left in place")
        if self.outcome == "cutout_failed":
            return (f"{self.name}: current classifier now calls this exterior, but cutout "
                     f"generation failed ({self.detail}) -- left in place")
        if self.outcome == "wheel_eligible_skipped":
            return (f"{self.name}: current classifier now calls this exterior and looks "
                     "wheel-centric -- promoting a wheel-routed shot isn't supported yet, left in place")
        if self.outcome == "insufficient_gallery":
            return (f"{self.name}: current classifier now calls this exterior, but there aren't "
                     "enough existing cutouts for the gallery consensus check to have an opinion "
                     "-- left in place rather than promoted unverified")
        if self.outcome == "classify_failed":
            return f"{self.name}: reclassification failed ({self.detail}) -- left in place"
        return f"{self.name}: {self.outcome}"


def resweep_interior_gallery(vehicle_folder: Path, classifier, interior_tiebreak_classifier=None,
                              angle_classifier=None, wheel_classifier=None, backbone=None,
                              upscale_model: str = "swinir", dry_run: bool = False
                              ) -> list[ReclassifiedPhoto]:
    """Self-heal counterpart to demote_foreign_cutouts(): re-runs the
    CURRENT evaluate_photo() against every photo already sitting in
    images/interior/, promoting any it now calls exterior.

    Exists because already_fetched() means a vehicle is only ever
    classified ONCE, at scrape time -- a classifier fix or threshold
    change (see imaging/pipeline.py's INTERIOR_TIEBREAK_THRESHOLD history,
    and the DETAIL_LABEL gap documented right below it) never
    retroactively applies to vehicles already on disk. A folder scraped
    before the fix existed just stays wrong forever unless someone
    manually re-audits it -- which is literally how the 2026-09 fleet
    audit that motivated this function was done, by hand, for exactly 2
    vehicles out of 10 candidates. This is what makes that repeatable.

    Deliberately conservative in ways confirmed necessary by that same
    audit -- 7 of the 10 "exterior-looking" interior photos found were NOT
    genuine recoverable misfiles:
      1. A photo only promotes if it also produces a valid cutout
         (remove_background's own quality gate). This alone doesn't catch
         everything: most DETAIL_LABEL misfires on a tight interior
         close-up (a door handle, a console button) still segment
         "successfully" by rembg's standards -- see (2).
      2. A promoted cutout is checked against find_foreign_cutouts() on a
         STAGED COPY of the real gallery before anything is written -- if
         the gallery consensus check would demote it right back (common:
         an unusual close-up framing genuinely doesn't resemble the
         vehicle's other shots), it's left alone. Forcing it through would
         reintroduce exactly the foreign-cutout-fusion bug class that
         check exists to prevent. If the existing gallery is too small for
         the consensus check to form an opinion at all, that's ALSO
         treated as "don't promote" (find_foreign_cutouts' own "empty
         means no opinion, not all clean" contract), not as tacit
         approval.
      3. Wheel-eligible-looking candidates (should_extract_wheel() would
         route them to the wheel pipeline) are reported but not promoted
         -- this only implements the general whole-vehicle cutout path,
         not the wheel money-shot path. Rare in practice.

    Caller is responsible for regenerating deliverables for any vehicle
    with a promotion (recompose --interiors, then hero-video) -- this
    function only touches the raw photo/cutout/angles.json layer, same
    division of labour as demote_foreign_cutouts().

    `outcome` on each returned ReclassifiedPhoto is one of: "promoted",
    "rejected_by_gallery", "cutout_failed", "wheel_eligible_skipped",
    "insufficient_gallery", "classify_failed". A photo the current
    classifier still calls interior isn't returned at all -- that's not a
    finding, it's confirmation nothing needs to change.
    """
    import io
    import json
    import shutil
    import tempfile

    from dtfb.imaging.cutout import remove_background
    from dtfb.imaging.pipeline import evaluate_photo, should_extract_wheel
    from dtfb.imaging.upscale import upscale

    vehicle_folder = Path(vehicle_folder)
    interior_dir = vehicle_folder / "images" / "interior"
    ext_dir = vehicle_folder / "images" / "exterior"
    cutout_dir = ext_dir / "cutout"
    if not interior_dir.is_dir():
        return []

    if backbone is None:
        from dtfb.imaging.classify import default_backbone
        backbone = default_backbone()

    photos = sorted(p for p in interior_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    results: list[ReclassifiedPhoto] = []

    for photo_path in photos:
        content = photo_path.read_bytes()
        try:
            verdict = evaluate_photo(content, classifier, interior_tiebreak_classifier)
        except Exception as e:
            results.append(ReclassifiedPhoto(photo_path.name, "classify_failed", str(e)))
            continue
        if verdict.category != "exterior":
            continue  # current code still calls it interior -- correct, nothing to do

        if wheel_classifier is not None and should_extract_wheel(content, verdict, wheel_classifier):
            results.append(ReclassifiedPhoto(photo_path.name, "wheel_eligible_skipped"))
            continue

        cutout = remove_background(content)
        if cutout.bbox is None or not cutout.quality_ok:
            results.append(ReclassifiedPhoto(photo_path.name, "cutout_failed", f"bbox={cutout.bbox}"))
            continue
        cropped = cutout.cutout.crop(cutout.bbox)
        rgba = upscale(cropped, upscale_model)

        existing_gallery = list(cutout_dir.glob("*.png")) if cutout_dir.is_dir() else []
        if len(existing_gallery) + 1 < MIN_GALLERY_FOR_CHECK:
            results.append(ReclassifiedPhoto(photo_path.name, "insufficient_gallery"))
            continue

        existing_n = [int(p.stem) for p in ext_dir.glob("*.jpg") if p.stem.isdigit()]
        next_n = max(existing_n, default=0) + 1
        candidate_name = f"{next_n:02d}.png"

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for f in existing_gallery:
                shutil.copy(f, tmp / f.name)
            rgba.save(tmp / candidate_name)
            foreign = find_foreign_cutouts(tmp, backbone=backbone)
        flagged = next((f for f in foreign if f.name == candidate_name), None)
        if flagged is not None:
            results.append(ReclassifiedPhoto(photo_path.name, "rejected_by_gallery", flagged.describe()))
            continue

        if dry_run:
            results.append(ReclassifiedPhoto(photo_path.name, "promoted", new_name=candidate_name))
            continue

        ext_path = ext_dir / f"{next_n:02d}{photo_path.suffix}"
        ext_path.write_bytes(content)
        cutout_dir.mkdir(parents=True, exist_ok=True)
        cutout_path = cutout_dir / candidate_name
        rgba.save(cutout_path)

        if angle_classifier is not None:
            a = angle_classifier.classify_file(cutout_path)
            angles_path = cutout_dir / "angles.json"
            try:
                angles = json.loads(angles_path.read_text())
            except (OSError, ValueError):
                angles = {}
            entry = {"angle": a.label, "confidence": round(a.confidence, 3)}
            if a.label == "side":
                from dtfb.imaging.classify import detect_hood_side
                buf = io.BytesIO()
                rgba.save(buf, format="PNG")
                try:
                    entry["hood_side"] = detect_hood_side(buf.getvalue())
                except Exception:
                    pass
            angles[candidate_name] = entry
            angles_path.write_text(json.dumps(angles, indent=2))

        photo_path.unlink()
        (vehicle_folder / "bundle" / "interior" / photo_path.name).unlink(missing_ok=True)

        results.append(ReclassifiedPhoto(photo_path.name, "promoted", new_name=candidate_name))

    return results
