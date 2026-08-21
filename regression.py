#!/usr/bin/env python3
"""
Regression harness for the imaging pipeline's calibrated decisions.

Re-runs the production decision paths -- scene classification
(imaging/pipeline.py::evaluate_photo, including the interior/exterior
tiebreak), wheel routing (should_extract_wheel, the same function
photos.py calls), and wheel money-shot extraction accept/reject
(imaging/wheel.py::extract_wheel_shot) -- against a checked-in snapshot of
ground-truth photos, and diffs the outcomes against
imaging/calibration/expected.json.

Also covers the three paths that gained authority to mutate files on
their own after the initial harness was built, each against its own
checked-in fixture set (imaging/calibration/*_expected.json):
  - imaging/gallery.py's cutout-consensus check (find_foreign_cutouts) --
    a synthetic whole-gallery fixture per case (a real vehicle's cutouts,
    sometimes plus one injected foreign cutout from a different vehicle),
    since this check is a leave-one-out measurement against a whole
    gallery and has nothing to compare against on a single photo.
  - imaging/letterbox.py's strip_banner -- real dealer-banner photos at
    several real banner heights, plus banner-free negative controls.
  - imaging/interior.py's enhance_interior -- real interior photos,
    pinning summary color/exposure stats before and after.
  - imaging/sticker.py's parse_sticker -- real Monroney window-sticker
    PDFs, including one that exposed a real bug (see sticker_expected.json).
  - imaging/select.py's hero/accent/carousel pickers -- a real vehicle's
    cutout gallery + angles.json (no CLIP needed, these are pure functions
    over that file).
  - facebook_post.py/social_post.py's post-copy builders -- 5 real
    vehicles' details.json spanning sticker/no-sticker, EV/gas, and
    confirmed-1-owner/not, diffed exact-text against pinned output.

Why this exists: every threshold in the pipeline (aspect gates, blob
dominance, tiebreak confidence, ...) was calibrated against real inventory
photos, and those decisions are exactly the kind of thing that silently
regresses when a CLIP prompt gets reworded, a model checkpoint updates, or
a dependency changes its post-processing. The source photos are snapshot
into imaging/calibration/photos/ (not referenced from the live listings
folder) precisely so the reference set survives listings being re-fetched
or cleaned.

DECISIONS are compared, not pixels: mask-level output can shift harmlessly
between library versions, but a money shot flipping to rejected, a
full-car photo flipping into the wheel path, or an interior photo flipping
to exterior is always worth a human look.

Some entries are marked KNOWN-IMPERFECT in their note: the recorded
outcome is not the ideal one, but it IS the current one -- the harness
pins it so an unnoticed change in either direction gets surfaced rather
than slipping by.

Usage:
    python regression.py            # run all cases, exit 1 on any mismatch
Runtime is a few minutes: every case pays real CLIP inference, and
wheel-routed cases pay CLIPSeg + SAM2 as well.

Adding cases: drop the photo into imaging/calibration/photos/ and append
an entry to expected.json with the reviewed expected outcome. When a
deliberate pipeline change shifts expected behavior, update expected.json
in the same commit -- the diff is the record of what changed and why.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CAL_DIR = Path(__file__).parent / "imaging" / "calibration"


def run_scene_and_wheel_cases() -> tuple[int, int]:
    from imaging.classify import (
        InteriorExteriorTiebreakClassifier,
        SceneClassifier,
        SpareTireClassifier,
        WheelDetailClassifier,
    )
    from imaging.pipeline import evaluate_photo, should_extract_wheel
    from imaging.wheel import extract_wheel_shot

    manifest = json.loads((CAL_DIR / "expected.json").read_text())
    scene_clf = SceneClassifier()
    tiebreak_clf = InteriorExteriorTiebreakClassifier()
    wheel_clf = WheelDetailClassifier()
    spare_clf = SpareTireClassifier()

    failures = 0
    for entry in manifest:
        photo = CAL_DIR / "photos" / entry["photo"]
        expect = entry["expect"]
        if not photo.exists():
            print(f"FAIL {entry['photo']}: snapshot photo missing")
            failures += 1
            continue
        content = photo.read_bytes()

        v = evaluate_photo(content, scene_clf, tiebreak_clf)
        routed = should_extract_wheel(content, v, wheel_clf)
        got = {"category": v.category, "wheel_routed": bool(routed)}
        if routed:
            got["wheel_accepted"] = extract_wheel_shot(content, wheel_clf, spare_clf) is not None

        if got == expect:
            print(f"ok   {entry['photo']}")
        else:
            print(f"FAIL {entry['photo']}: expected {expect}, got {got}  [{entry['note']}]")
            failures += 1

    return len(manifest), failures


def run_banner_cases() -> tuple[int, int]:
    from PIL import Image

    from imaging.letterbox import detect_banner

    manifest = json.loads((CAL_DIR / "banner_expected.json").read_text())
    failures = 0
    for entry in manifest:
        photo = CAL_DIR / "banner_photos" / entry["photo"]
        expect = tuple(entry["expect"])
        if not photo.exists():
            print(f"FAIL {entry['photo']}: snapshot photo missing")
            failures += 1
            continue

        got = detect_banner(Image.open(photo))
        if got == expect:
            print(f"ok   {entry['photo']}")
        else:
            print(f"FAIL {entry['photo']}: expected {expect}, got {got}")
            failures += 1

    return len(manifest), failures


def run_gallery_cases() -> tuple[int, int]:
    from imaging.classify import default_backbone
    from imaging.gallery import CERTAIN_SIMILARITY, find_foreign_cutouts

    manifest = json.loads((CAL_DIR / "gallery_expected.json").read_text())
    backbone = default_backbone()

    failures = 0
    for entry in manifest:
        folder = CAL_DIR / "gallery" / entry["folder"]
        expect = sorted(entry["expect"])
        if not folder.is_dir():
            print(f"FAIL {entry['folder']}: snapshot gallery missing")
            failures += 1
            continue

        # certain_similarity pinned explicitly rather than left to load
        # from a live cutout-calibration.json -- this harness locks in
        # the built-in default's behavior, not whatever a listings root
        # happens to be calibrated to at the moment it's run.
        found = find_foreign_cutouts(folder, backbone=backbone, certain_similarity=CERTAIN_SIMILARITY)
        got = sorted(f.name for f in found)
        if got == expect:
            print(f"ok   {entry['folder']}")
        else:
            print(f"FAIL {entry['folder']}: expected {expect}, got {got}  [{entry['note']}]")
            failures += 1

    return len(manifest), failures


def run_interior_cases() -> tuple[int, int]:
    import numpy as np
    from PIL import Image

    from imaging.interior import enhance_interior

    def stats(img):
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255
        return {
            "r": round(float(arr[:, :, 0].mean()), 4),
            "g": round(float(arr[:, :, 1].mean()), 4),
            "b": round(float(arr[:, :, 2].mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "bright_frac": round(float((arr.max(axis=2) > 0.97).mean()), 4),
        }

    manifest = json.loads((CAL_DIR / "interior_expected.json").read_text())
    tolerance = 0.01

    failures = 0
    for entry in manifest:
        photo = CAL_DIR / "interior_photos" / entry["photo"]
        expect = entry["expect_after"]
        if not photo.exists():
            print(f"FAIL {entry['photo']}: snapshot photo missing")
            failures += 1
            continue

        got = stats(enhance_interior(Image.open(photo)))
        diff = {k: (expect[k], got[k]) for k in expect if abs(expect[k] - got[k]) > tolerance}
        if not diff:
            print(f"ok   {entry['photo']}")
        else:
            print(f"FAIL {entry['photo']}: {diff}  [{entry['note']}]")
            failures += 1

    return len(manifest), failures


def run_sticker_cases() -> tuple[int, int]:
    from imaging.sticker import parse_sticker

    manifest = json.loads((CAL_DIR / "sticker_expected.json").read_text())
    failures = 0
    for entry in manifest:
        pdf = CAL_DIR / "sticker_pdfs" / entry["pdf"]
        expect = entry["expect"]
        if not pdf.exists():
            print(f"FAIL {entry['pdf']}: snapshot PDF missing")
            failures += 1
            continue

        data = parse_sticker(pdf)
        got = {
            "vin": data["overview"].get("vin"),
            "model_line": data["overview"].get("model_line"),
            "n_equipment_total": sum(len(v) for v in data["equipment"].values()),
            "n_optional": len(data["optional_equipment"]),
            "n_warranties": len(data["warranties"]),
            "pricing": data["pricing"],
        }
        if got == expect:
            print(f"ok   {entry['pdf']}")
        else:
            print(f"FAIL {entry['pdf']}: expected {expect}, got {got}  [{entry['note']}]")
            failures += 1

    return len(manifest), failures


def run_select_cases() -> tuple[int, int]:
    from imaging.select import (pick_adaptive, pick_all_for_carousel, pick_for_carousel,
                                 pick_for_conveyor, pick_for_quad_layout, pick_hero_shots,
                                 order_for_conveyor_start)

    expect = json.loads((CAL_DIR / "select_expected.json").read_text())
    cutout_dir = CAL_DIR / "select" / "cutout"
    wheel_dir = CAL_DIR / "select" / "wheels"

    def names(paths):
        return [p.name for p in paths]

    def pair_names(pairs):
        return [[p.name, l] for p, l in pairs]

    adaptive_layout, adaptive_shots = pick_adaptive(cutout_dir)
    all_carousel = pick_all_for_carousel(cutout_dir, wheel_dir)

    cases = [
        ("pick_hero_shots", names(pick_hero_shots(cutout_dir))),
        ("pick_for_quad_layout", names(pick_for_quad_layout(cutout_dir))),
        ("pick_for_conveyor", names(pick_for_conveyor(cutout_dir))),
        ("pick_adaptive", {"layout": adaptive_layout, "shots": names(adaptive_shots)}),
        ("pick_for_carousel", pair_names(pick_for_carousel(cutout_dir, wheel_dir))),
        ("pick_all_for_carousel", pair_names(all_carousel)),
        ("order_for_conveyor_start", pair_names(order_for_conveyor_start(all_carousel, cutout_dir))),
    ]

    failures = 0
    for name, got in cases:
        if got == expect[name]:
            print(f"ok   {name}")
        else:
            print(f"FAIL {name}: expected {expect[name]}, got {got}")
            failures += 1

    return len(cases), failures


def run_posts_cases() -> tuple[int, int]:
    from posts_cli import WRITERS, load_vehicle

    posts_dir = CAL_DIR / "posts"
    cases = sorted(p for p in posts_dir.iterdir() if p.is_dir())

    n = failures = 0
    for folder in cases:
        v = load_vehicle(folder)
        if v is None:
            print(f"FAIL {folder.name}: unreadable details.json fixture")
            failures += 1
            n += 1
            continue
        for wname, fn in WRITERS.items():
            n += 1
            expect_path = folder / wname
            expect = expect_path.read_text() if expect_path.exists() else None
            got = fn(v)
            if got == expect:
                print(f"ok   {folder.name}/{wname}")
            else:
                print(f"FAIL {folder.name}/{wname}: output no longer matches the pinned expected text")
                failures += 1

    return n, failures


def main() -> int:
    total = failures = 0
    for label, runner in [
        ("scene/wheel", run_scene_and_wheel_cases),
        ("banner", run_banner_cases),
        ("gallery", run_gallery_cases),
        ("interior", run_interior_cases),
        ("sticker", run_sticker_cases),
        ("select", run_select_cases),
        ("posts", run_posts_cases),
    ]:
        print(f"--- {label} ---")
        n, f = runner()
        total += n
        failures += f
        print()

    print(f"{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
