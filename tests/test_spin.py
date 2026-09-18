"""Tests for imaging/compose/spin.py's pure ordering logic (roadmap item
3, lotstretcher project graph decision 1449fdef). render_spin_video() itself needs
real cutouts + ffmpeg and is validated by hand (see the commit message),
not here -- this covers order_for_spin()'s selection/ordering, which is
plain data logic over angles.json."""
import json

from lotstretcher.imaging.compose.spin import SPIN_ANGLE_ORDER, order_for_spin


def _write_angles(cutout_dir, entries):
    cutout_dir.mkdir(parents=True, exist_ok=True)
    (cutout_dir / "angles.json").write_text(json.dumps(entries))


def test_orders_front_to_rear_not_carousel_order(tmp_path):
    """SPIN_ANGLE_ORDER must be geometric (front..rear), NOT
    select.py's CAROUSEL_ANGLE_ORDER which front-loads front_3q for a
    dramatic reveal -- a spin has to read as continuous rotation."""
    assert SPIN_ANGLE_ORDER == ["front", "front_3q", "side", "rear_3q", "rear"]

    cutout_dir = tmp_path / "cutout"
    _write_angles(cutout_dir, {
        "03.png": {"angle": "rear", "confidence": 0.9},
        "01.png": {"angle": "front", "confidence": 0.9},
        "02.png": {"angle": "side", "confidence": 0.9},
    })
    got = order_for_spin(cutout_dir)
    assert [label for _p, label in got] == ["front", "side", "rear"]
    assert [p.name for p, _l in got] == ["01.png", "02.png", "03.png"]


def test_missing_angles_are_just_absent(tmp_path):
    """No 'rear' shot for this vehicle -> spin stops at rear_3q, nothing
    invented to fill the gap."""
    cutout_dir = tmp_path / "cutout"
    _write_angles(cutout_dir, {
        "01.png": {"angle": "front", "confidence": 0.9},
        "02.png": {"angle": "front_3q", "confidence": 0.9},
    })
    got = order_for_spin(cutout_dir)
    assert [label for _p, label in got] == ["front", "front_3q"]


def test_picks_highest_confidence_duplicate(tmp_path):
    """Two 'side' shots (real case, see promote-silverado's own gallery)
    -- the higher-confidence one wins, matching select.py's own
    _best_by_angle() convention."""
    cutout_dir = tmp_path / "cutout"
    _write_angles(cutout_dir, {
        "01.png": {"angle": "front", "confidence": 0.9},
        "02.png": {"angle": "side", "confidence": 0.70},
        "03.png": {"angle": "side", "confidence": 0.95},
    })
    got = order_for_spin(cutout_dir)
    side_files = [p.name for p, label in got if label == "side"]
    assert side_files == ["03.png"]


def test_no_angles_json_returns_empty(tmp_path):
    cutout_dir = tmp_path / "cutout"
    cutout_dir.mkdir(parents=True)
    assert order_for_spin(cutout_dir) == []


def test_unknown_angle_labels_excluded(tmp_path):
    """A future/unrecognized angle label (or a real one this spin doesn't
    use, e.g. 'detail') is silently excluded, not appended like
    pick_for_carousel() does -- a spin's whole point is geometric order,
    so there's nowhere sensible to put an unplaceable shot."""
    cutout_dir = tmp_path / "cutout"
    _write_angles(cutout_dir, {
        "01.png": {"angle": "front", "confidence": 0.9},
        "02.png": {"angle": "some_future_label", "confidence": 0.9},
    })
    got = order_for_spin(cutout_dir)
    assert [label for _p, label in got] == ["front"]
