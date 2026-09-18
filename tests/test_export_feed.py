"""Tests for export_feed.py -- the inventory-syndication feed generator
(roadmap item 2, lotstretcher project graph decision 1449fdef). Pure data
transformation off details.json, no CV/CLIP needed, so this lives in the
fast pytest suite rather than regression.py."""
import csv
import json

from lotstretcher.export_feed import FEED_COLUMNS, build_rows, vehicle_to_row, write_csv


def _write_details(folder, **overrides):
    base = dict(
        url="https://example.com/vehicle/1HGCY1F24SA123456/",
        vin="1HGCY1F24SA123456",
        stock_number="ST123",
        year=2025,
        make="Honda",
        model="Accord",
        trim="LX",
        title="2025 Honda Accord LX",
        condition="Certified Pre-Owned",
        mileage=15000,
        display_price=25000,
        exterior_color_generic="Gray",
        interior_color="Gray",
        transmission="CVT",
        drivetrain="FWD",
        fuel_type="Gasoline",
        body_type="Sedan",
        dealer_name="Tomball Ford",
        dealer_address="22702 TX-249, Tomball, TX 77375",
        dealer_phone=None,
        dealer_description="A great car.",
        photo_urls=["https://cdn.example.com/1.jpg", "https://cdn.example.com/2.jpg"],
        delisted_at=None,
    )
    base.update(overrides)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "details.json").write_text(json.dumps(base))
    return folder


def test_vehicle_to_row_maps_core_fields(tmp_path):
    folder = _write_details(tmp_path / "vehicle1")
    row = vehicle_to_row(folder)
    assert row["vehicle_id"] == "1HGCY1F24SA123456"
    assert row["make"] == "Honda"
    assert row["model"] == "Accord"
    assert row["year"] == 2025
    assert row["vin"] == "1HGCY1F24SA123456"
    assert row["mileage.value"] == 15000
    assert row["mileage.unit"] == "MI"
    assert row["price"] == 25000
    assert row["currency"] == "USD"
    assert row["state_of_vehicle"] == "used"  # Certified Pre-Owned -> used
    assert row["image[0].url"] == "https://cdn.example.com/1.jpg"
    assert row["image[1].url"] == "https://cdn.example.com/2.jpg"
    assert "image[2].url" not in row


def test_new_condition_maps_to_new_state(tmp_path):
    folder = _write_details(tmp_path / "vehicle2", condition="New")
    row = vehicle_to_row(folder)
    assert row["state_of_vehicle"] == "new"


def test_delisted_vehicle_excluded(tmp_path):
    folder = _write_details(tmp_path / "vehicle3", delisted_at="2026-09-01T00:00:00Z")
    assert vehicle_to_row(folder) is None


def test_missing_price_excluded(tmp_path):
    folder = _write_details(tmp_path / "vehicle4", display_price=None)
    assert vehicle_to_row(folder) is None


def test_missing_photos_excluded(tmp_path):
    folder = _write_details(tmp_path / "vehicle5", photo_urls=[])
    assert vehicle_to_row(folder) is None


def test_formatted_price_string_cleaned(tmp_path):
    """Some dealer payloads give display_price as a formatted string
    rather than a bare number -- must still come out as a clean int."""
    folder = _write_details(tmp_path / "vehicle6", display_price="$25,382")
    row = vehicle_to_row(folder)
    assert row["price"] == 25382


def test_image_count_capped_at_max(tmp_path):
    urls = [f"https://cdn.example.com/{i}.jpg" for i in range(30)]
    folder = _write_details(tmp_path / "vehicle7", photo_urls=urls)
    row = vehicle_to_row(folder)
    assert row["image[19].url"] == urls[19]
    assert "image[20].url" not in row


def test_build_rows_skips_none(tmp_path):
    good = _write_details(tmp_path / "good")
    bad = _write_details(tmp_path / "bad", display_price=None)
    rows = build_rows([good, bad])
    assert len(rows) == 1
    assert rows[0]["vehicle_id"] == "1HGCY1F24SA123456"


def test_write_csv_round_trips(tmp_path):
    folder = _write_details(tmp_path / "vehicle8")
    rows = build_rows([folder])
    out = tmp_path / "feed.csv"
    write_csv(rows, out)

    with open(out) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FEED_COLUMNS
        got = list(reader)
    assert len(got) == 1
    assert got[0]["vehicle_id"] == "1HGCY1F24SA123456"
    assert got[0]["price"] == "25000"  # CSV round-trips everything as text


def test_write_csv_tsv_delimiter(tmp_path):
    folder = _write_details(tmp_path / "vehicle9")
    rows = build_rows([folder])
    out = tmp_path / "feed.tsv"
    write_csv(rows, out, delimiter="\t")
    first_line = out.read_text().splitlines()[0]
    assert "\t" in first_line
    assert "," not in first_line.split("\t")[0]
