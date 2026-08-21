"""
Download a vehicle's photo gallery and run it through the imaging/ pipeline:
pHash dedupe against known junk templates (imaging/dedupe.py), CLIP
exterior/interior/detail sorting with a rembg tiebreak for banner-confused
shots (imaging/pipeline.py), transparent cutouts with a quality gate and
CLIP angle tagging for exterior photos (imaging/cutout.py,
imaging/classify.py), and batch letterbox-bar cropping for interior photos
(imaging/letterbox.py). See imaging/__init__.py for the module breakdown.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from dtfb.scrape import Vehicle


WHEEL_DUPLICATE_SIMILARITY = 0.9995


def _wheel_signature(img):
    """A tiny normalised vector for comparing one wheel crop to another.
    32x32 is plenty: these are tight crops of the same object, so a real
    duplicate matches almost exactly and two different wheels do not come
    close."""
    import numpy as np

    small = img.convert("RGB").resize((32, 32))
    vec = np.asarray(small, dtype=np.float64).ravel()
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _is_duplicate_wheel(img, signatures: list) -> bool:
    """True if this wheel crop matches one already kept for this vehicle.
    Appends its signature when it doesn't, so the caller just asks."""
    import numpy as np

    sig = _wheel_signature(img)
    if any(float(sig @ prior) > WHEEL_DUPLICATE_SIMILARITY for prior in signatures):
        return True
    signatures.append(sig)
    return False


def _clear_stale_gallery(images_dir: Path) -> int:
    """Delete a previous run's photos before this run writes its own.

    Every call renumbers from 01, so any file left over from an earlier
    scrape is stale by definition -- and because every downstream consumer
    finds photos by globbing the directory (imaging/select.py,
    compose/pipeline.py, hero_video_cli.py), an orphan doesn't sit
    harmlessly on disk, it ships. Confirmed on a real re-scrape: the Nissan
    Rogue kept cutout/06.png from a run whose routing had since changed,
    byte-identical to the new 07.png, and the hero video rendered it as a
    duplicate shot in a 9-shot conveyor.

    Safe here because the whole gallery is already downloaded into memory
    by the time this runs -- a later failure can't leave the vehicle with
    neither the old photos nor the new. Only numbered files (and the
    angles sidecar) are touched, so anything a human dropped in the folder
    survives.
    """
    removed = 0
    for sub in (".", "exterior", "interior", "exterior/cutout", "exterior/wheels"):
        directory = images_dir / sub
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and (path.stem.isdigit() or path.name == "angles.json"):
                path.unlink()
                removed += 1
    return removed


def download_photos(session: requests.Session, v: Vehicle, images_dir: Path,
                     junk_filter: "JunkFilter | None" = None,
                     classifier: "SceneClassifier | None" = None,
                     upscale_cutouts: bool = False, upscale_model: str = "swinir",
                     angle_classifier: "AngleClassifier | None" = None,
                     wheel_classifier: "WheelDetailClassifier | None" = None,
                     spare_classifier: "SpareTireClassifier | None" = None,
                     interior_tiebreak_classifier: "InteriorExteriorTiebreakClassifier | None" = None,
                     strict_cutouts: bool = True) -> list[str]:
    """
    Download the gallery, drop known junk templates (pHash), and -- if a
    classifier is supplied -- sort survivors into images/exterior/ and
    images/interior/, dropping anything CLIP+rembg agree has no real vehicle
    in it (see imaging/pipeline.py). Exterior photos also get a best-effort
    white-background cutout saved to images/exterior/cutout/ under the same
    number, skipped when imaging/cutout.py's quality gate flags the
    segmentation as unreliable (typical for tight detail shots) -- the
    original stays in images/exterior/ either way.

    A "detail" close-up (imaging/pipeline.py's DETAIL_LABEL) skips that
    cutout -- but if wheel_classifier and spare_classifier are supplied and
    imaging/wheel.py's full gate confirms it's genuinely a rolling wheel at
    a real 3/4 money-shot angle (not the mirror/badge/light/handle/tread/
    mounted-spare/front-tread-on shots that label also catches), it gets
    its own tightly-cropped cutout saved to images/exterior/wheels/
    instead, for a dedicated "wheel money shot" composition. The same
    wheel treatment also applies to a photo the scene classifier called
    "exterior" when WheelDetailClassifier confidently disagrees -- see the
    routing comment in the loop below for the real misrouted-partial-shot
    cases that made this necessary.

    With strict_cutouts (the default), the finished cutout gallery then has
    to vouch for its own members: any cutout that neither resembles nor is
    painted like its siblings is deleted and its photo is DEMOTED to
    images/interior/ rather than discarded -- see imaging/gallery.py for
    the two real failures that motivated this and the five approaches that
    didn't work. Demotion rather than deletion because those photos are
    fine photos that simply shouldn't have been cut out (a shot through
    the windshield, a van's cargo bay), and "shouldn't be cut out" is the
    definition of the interior gallery here.

    Without a classifier, falls back to the flat images/NN.ext layout.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    filtered_count = 0
    for url in v.photo_urls:
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.content
        except requests.RequestException as e:
            print(f"    ! download failed ({url}): {e}", file=sys.stderr)
            continue

        if junk_filter is not None:
            is_junk, matched = junk_filter.is_junk(content)
            if is_junk:
                filtered_count += 1
                continue

        ext = Path(urlparse(url).path).suffix or ".jpg"
        downloaded.append((content, ext))

    if filtered_count:
        v.warnings.append(
            f"Filtered {filtered_count} known marketing/junk image(s) out of the gallery."
        )

    if classifier is None:
        saved = []
        for i, (content, ext) in enumerate(downloaded, start=1):
            dest = images_dir / f"{i:02d}{ext}"
            dest.write_bytes(content)
            saved.append(dest.name)
        return saved

    _clear_stale_gallery(images_dir)

    from dtfb.imaging.pipeline import evaluate_photo, should_extract_wheel
    from dtfb.imaging.cutout import remove_background
    from dtfb.imaging.upscale import upscale
    from dtfb.imaging.wheel import extract_wheel_shot

    ext_dir = images_dir / "exterior"
    int_dir = images_dir / "interior"
    cutout_dir = ext_dir / "cutout"
    wheel_dir = ext_dir / "wheels"

    saved = []
    no_subject_count = 0
    cutout_count = 0
    wheel_count = 0
    angles: dict[str, dict] = {}
    interior_buffer: list[tuple[int, bytes, str]] = []
    # cutout filename -> the exterior photo it came from, so the
    # gallery-consensus pass below can demote one without re-reading it.
    cutout_sources: dict[str, tuple[Path, bytes, str]] = {}
    wheel_signatures: list = []
    ext_i = int_i = 0
    for content, ext in downloaded:
        try:
            verdict = evaluate_photo(content, classifier, interior_tiebreak_classifier)
        except Exception as e:
            print(f"    ! classification failed, keeping unsorted: {e}", file=sys.stderr)
            verdict = None

        if verdict is None or verdict.category == "exterior":
            ext_dir.mkdir(parents=True, exist_ok=True)
            ext_i += 1
            dest = ext_dir / f"{ext_i:02d}{ext}"
            dest.write_bytes(content)
            saved.append(f"exterior/{dest.name}")

            # Routing between the whole-vehicle cutout and the wheel
            # money-shot pipeline. The scene classifier's "detail" label
            # catches most wheel close-ups, but a wheel-centric PARTIAL
            # body shot (rear quarter panel + wheel filling much of the
            # frame, greenhouse cut off by the photo edge) reads as
            # "exterior" to it -- and the whole-vehicle treatment then
            # ships an ugly floating slab of body panel as if it were a
            # full car (confirmed on three real vehicles). Two checks
            # both have to agree before re-routing such a photo: the
            # narrow WheelDetailClassifier says "wheel" (it does, at
            # 0.95+, on the real misrouted shots -- but it ALSO fires on
            # some genuine full-car photos where wheels are prominent, so
            # it can't route alone), and imaging/wheel.py's
            # is_wheel_centric_shot() confirms the photo is composed
            # around ONE dominant wheel rather than showing both axles
            # (the signal that actually separates those two cases -- see
            # its docstring for the measured gap). A photo that passes
            # both goes to the wheel pipeline INSTEAD of the slab
            # treatment -- deliberately instead-of, not in-addition-to:
            # even if the wheel extraction then rejects (bad angle), the
            # slab was never worth shipping. The decision itself lives in
            # imaging/pipeline.py::should_extract_wheel() so regression.py
            # exercises the exact same code.
            wheel_route = (spare_classifier is not None
                            and should_extract_wheel(content, verdict, wheel_classifier))

            if wheel_route:
                try:
                    wheel_cutout = extract_wheel_shot(content, wheel_classifier, spare_classifier)
                    if wheel_cutout is not None and _is_duplicate_wheel(wheel_cutout, wheel_signatures):
                        # Two photos of the same wheel from near-identical
                        # angles survive as distinct exterior shots but
                        # converge once each is cropped tight to the rim --
                        # confirmed on a real vehicle whose two money shots
                        # were byte-identical after resize (cosine 1.0000).
                        # Rare (1 vehicle in 29) but it ships twice: once in
                        # bundle/framed/ and again as a repeated slot in the
                        # video carousel.
                        wheel_cutout = None
                    if wheel_cutout is not None:
                        # Unconditional, unlike the general cutout gallery below: measured
                        # across the fleet, wheel crops are the one place upscaling actually
                        # matters. Median wheel crop is ~520px but the money-shot composite
                        # scales it up ~2.1x on average (59% of real crops go past 2x) --
                        # squarely the range where 4x SR shows a visible gain over Lanczos
                        # (see imaging/upscale.py's grille-badge test). The general exterior
                        # gallery median scale is ~1.1x, where that gain doesn't materialize,
                        # which is why --upscale stays opt-in for it.
                        rgba = upscale(wheel_cutout, upscale_model)
                        wheel_dir.mkdir(parents=True, exist_ok=True)
                        wheel_path = wheel_dir / f"{ext_i:02d}.png"
                        rgba.save(wheel_path)
                        saved.append(f"exterior/wheels/{wheel_path.name}")
                        wheel_count += 1
                except Exception as e:
                    print(f"    ! wheel detection/cutout failed for exterior/{dest.name}: {e}", file=sys.stderr)

            elif verdict is None or verdict.cutout_eligible:
                try:
                    cutout = remove_background(content)
                    if cutout.quality_ok:
                        # Saved transparent and cropped tight to the visible pixels (not
                        # composited onto white) -- so downstream compositing (imaging/compose.py)
                        # can align/scale cutouts from different angles consistently, and so no
                        # information is thrown away that a later step might want back.
                        cropped = cutout.cutout.crop(cutout.bbox)
                        rgba = upscale(cropped, upscale_model) if upscale_cutouts else cropped
                        cutout_dir.mkdir(parents=True, exist_ok=True)
                        cutout_path = cutout_dir / f"{ext_i:02d}.png"
                        rgba.save(cutout_path)
                        saved.append(f"exterior/cutout/{cutout_path.name}")
                        cutout_sources[cutout_path.name] = (dest, content, ext)
                        cutout_count += 1

                        if angle_classifier is not None:
                            try:
                                a = angle_classifier.classify_file(cutout_path)
                                angles[cutout_path.name] = {
                                    "angle": a.label,
                                    "confidence": round(a.confidence, 3),
                                }
                            except Exception as e:
                                print(f"    ! angle classification failed for {cutout_path.name}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"    ! cutout failed for exterior/{dest.name}: {e}", file=sys.stderr)

        elif verdict.category == "interior":
            int_i += 1
            # Buffered, not saved yet -- some photo vendors pad interior
            # shots with a flat-color bar top/bottom (seen: solid white,
            # ~78px/79px, reserved space for a watermark that wasn't filled
            # in), but a handful of individual photos have too gradual a
            # transition for single-image detection to find reliably (real
            # content right at the boundary can itself be bright: a sunlit
            # window, a light-colored headliner). The vendor's bar size is
            # fixed across a whole gallery though, so detecting it from
            # whichever photos show it clearly and applying that size to
            # the whole batch is far more reliable -- needs every interior
            # photo decoded before we know the consensus, hence the buffer.
            interior_buffer.append((int_i, content, ext))

        else:  # "no_subject" or "document" -- CLIP+rembg agree there's no vehicle here
            no_subject_count += 1

    # Does the finished gallery vouch for every cutout in it? This has to
    # run here, after the loop -- both keys are leave-one-out measurements
    # against the vehicle's OTHER cutouts, so there is nothing to compare
    # against until they all exist -- and before the interior flush below,
    # so anything demoted joins that batch and gets its letterbox pass and
    # its number like any other interior photo.
    if strict_cutouts and cutout_sources:
        from dtfb.imaging.gallery import find_foreign_cutouts

        try:
            foreign = find_foreign_cutouts(cutout_dir, backbone=classifier.backbone)
        except Exception as e:
            print(f"    ! cutout consensus check failed, keeping all cutouts: {e}", file=sys.stderr)
            foreign = []

        demoted = []
        for f in foreign:
            source = cutout_sources.get(f.name)
            if source is None:
                continue
            photo_path, content, photo_ext = source
            (cutout_dir / f.name).unlink(missing_ok=True)
            photo_path.unlink(missing_ok=True)
            angles.pop(f.name, None)
            # Leaves a gap in the exterior numbering, which nothing
            # downstream depends on -- every consumer globs the directory
            # (see imaging/select.py) and the cutout numbering already has
            # gaps wherever the quality gate rejected a segmentation.
            for entry in (f"exterior/{photo_path.name}", f"exterior/cutout/{f.name}"):
                if entry in saved:
                    saved.remove(entry)
            int_i += 1
            interior_buffer.append((int_i, content, photo_ext))
            cutout_count -= 1
            demoted.append(f)
            print(f"    - demoted exterior/{photo_path.name} to interior: {f.describe()}")

        if demoted:
            v.warnings.append(
                f"Moved {len(demoted)} photo(s) from the exterior gallery to interior: their cutouts "
                "matched neither the shape nor the paint of this vehicle's other shots "
                "(usually a photo taken from inside the cabin, or one showing another car on the lot)."
            )

    if interior_buffer:
        from dtfb.imaging.letterbox import crop_bars, detect_batch_bars

        int_dir.mkdir(parents=True, exist_ok=True)
        decoded = [Image.open(io.BytesIO(content)) for _, content, _ in interior_buffer]
        try:
            bar_top, bar_bottom = detect_batch_bars(decoded)
        except Exception as e:
            print(f"    ! letterbox batch detection failed, keeping originals: {e}", file=sys.stderr)
            bar_top, bar_bottom = 0, 0

        for (i, content, ext), img in zip(interior_buffer, decoded):
            dest = int_dir / f"{i:02d}{ext}"
            try:
                cropped = crop_bars(img, top=bar_top, bottom=bar_bottom)
                if cropped is not img:
                    cropped.save(dest, quality=92)
                else:
                    dest.write_bytes(content)
            except Exception as e:
                print(f"    ! letterbox crop failed for interior/{dest.name}, keeping original: {e}", file=sys.stderr)
                dest.write_bytes(content)
            saved.append(f"interior/{dest.name}")

        if bar_top or bar_bottom:
            v.warnings.append(
                f"Cropped a {bar_top}px/{bar_bottom}px top/bottom letterbox bar off all interior photos "
                "(vendor watermark padding, detected from the batch)."
            )

    if no_subject_count:
        v.warnings.append(
            f"Filtered {no_subject_count} additional non-vehicle image(s) via CLIP/rembg "
            "(not in the known-junk template library)."
        )
    if cutout_count:
        v.warnings.append(f"Generated {cutout_count} transparent cutout(s).")
    if wheel_count:
        v.warnings.append(f"Found {wheel_count} whole-wheel close-up(s), cut out for a wheel money shot.")
    if angles:
        (cutout_dir / "angles.json").write_text(json.dumps(angles, indent=2))

    return saved
