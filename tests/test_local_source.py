"""Tests for local_source.py's non-scrape entry point (lotstretcher project graph
decision <local-source-entry>, roadmap item 6): building a Vehicle from a
local photo folder instead of a scraped VDP. Pure logic only -- no CV
models needed, just real (tiny, checked-in-adjacent) image bytes."""
from PIL import Image

from lotstretcher import local_source


def _make_image(path, size=(4, 4)):
    Image.new("RGB", size, color=(200, 50, 50)).save(path)


def test_is_local_source_true_for_directory(tmp_path):
    assert local_source.is_local_source(str(tmp_path)) is True


def test_is_local_source_false_for_url(tmp_path):
    assert local_source.is_local_source("https://example.com/vehicle/123/") is False


def test_is_local_source_false_for_missing_path(tmp_path):
    assert local_source.is_local_source(str(tmp_path / "does-not-exist")) is False


def test_load_local_vehicle_no_metadata(tmp_path):
    _make_image(tmp_path / "01.jpg")
    _make_image(tmp_path / "02.jpg")
    v = local_source.load_local_vehicle(tmp_path)
    assert len(v.photo_urls) == 2
    assert all(str(tmp_path.resolve()) in p for p in v.photo_urls)
    assert v.stock_number == tmp_path.resolve().name
    assert v.url == f"local://{tmp_path.resolve().name}"


def test_load_local_vehicle_with_metadata(tmp_path):
    _make_image(tmp_path / "01.jpg")
    (tmp_path / "vehicle.json").write_text('{"make": "Ford", "model": "F-150", "stock_number": "ABC123"}')
    v = local_source.load_local_vehicle(tmp_path)
    assert v.make == "Ford"
    assert v.model == "F-150"
    assert v.stock_number == "ABC123"


def test_load_local_vehicle_ignores_non_image_files(tmp_path):
    _make_image(tmp_path / "01.jpg")
    (tmp_path / "notes.txt").write_text("not a photo")
    v = local_source.load_local_vehicle(tmp_path)
    assert len(v.photo_urls) == 1


def test_load_local_vehicle_empty_folder_raises(tmp_path):
    try:
        local_source.load_local_vehicle(tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "No photos found" in str(e)


def test_load_local_vehicle_unknown_field_raises(tmp_path):
    _make_image(tmp_path / "01.jpg")
    (tmp_path / "vehicle.json").write_text('{"marke": "Ford"}')
    try:
        local_source.load_local_vehicle(tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "marke" in str(e)


def test_load_local_vehicle_derived_field_rejected(tmp_path):
    """url/photo_urls/warnings are derived, not user-settable -- a
    vehicle.json naming one should fail with the same clean error as any
    other unknown field, not a raw TypeError from a keyword collision."""
    _make_image(tmp_path / "01.jpg")
    (tmp_path / "vehicle.json").write_text('{"url": "http://example.com"}')
    try:
        local_source.load_local_vehicle(tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "url" in str(e)


def test_load_local_vehicle_invalid_json_raises(tmp_path):
    _make_image(tmp_path / "01.jpg")
    (tmp_path / "vehicle.json").write_text("{not valid json")
    try:
        local_source.load_local_vehicle(tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "vehicle.json" in str(e)


def test_local_vehicle_key_is_stable_and_distinct(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert local_source.local_vehicle_key(a) == local_source.local_vehicle_key(a)
    assert local_source.local_vehicle_key(a) != local_source.local_vehicle_key(b)
