"""Tests for server/store.py's SQLite-backed registry (roadmap item 5,
dtfb project graph decision 1449fdef). Pure CRUD logic, no CV/HTTP
needed -- fast pytest suite."""
from dtfb.server import store


def test_upsert_and_get_vehicle(tmp_path):
    store.upsert_vehicle(tmp_path, "VIN1", {"make": "Ford"})
    v = store.get_vehicle(tmp_path, "VIN1")
    assert v["vehicle_id"] == "VIN1"
    assert v["status"] == store.STATUS_NEW
    assert v["metadata"] == {"make": "Ford"}


def test_upsert_is_idempotent_update(tmp_path):
    store.upsert_vehicle(tmp_path, "VIN1", {"make": "Ford"})
    store.upsert_vehicle(tmp_path, "VIN1", {"make": "Ford", "model": "F-150"})
    v = store.get_vehicle(tmp_path, "VIN1")
    assert v["metadata"] == {"make": "Ford", "model": "F-150"}
    assert len(store.list_vehicles(tmp_path)) == 1


def test_get_unknown_vehicle_returns_none(tmp_path):
    assert store.get_vehicle(tmp_path, "NOPE") is None


def test_list_vehicles_filters_by_status(tmp_path):
    store.upsert_vehicle(tmp_path, "VIN1", {})
    store.upsert_vehicle(tmp_path, "VIN2", {})
    store.set_vehicle_status(tmp_path, "VIN2", store.STATUS_DONE)

    all_v = store.list_vehicles(tmp_path)
    new_v = store.list_vehicles(tmp_path, status=store.STATUS_NEW)
    assert len(all_v) == 2
    assert [v["vehicle_id"] for v in new_v] == ["VIN1"]


def test_set_vehicle_status_returns_false_for_unknown(tmp_path):
    assert store.set_vehicle_status(tmp_path, "NOPE", store.STATUS_DONE) is False


def test_delete_vehicle(tmp_path):
    store.upsert_vehicle(tmp_path, "VIN1", {})
    assert store.delete_vehicle(tmp_path, "VIN1") is True
    assert store.get_vehicle(tmp_path, "VIN1") is None
    assert store.delete_vehicle(tmp_path, "VIN1") is False


def test_create_submission_and_get(tmp_path):
    sub_id = store.create_submission(tmp_path, "VIN1", {"a": "http://x/a.jpg", "b": "http://x/b.jpg"}, None)
    sub = store.get_submission(tmp_path, sub_id)
    assert sub["vehicle_id"] == "VIN1"
    assert sub["webhook_url"] is None
    statuses = {i["image_id"]: i["status"] for i in sub["images"]}
    assert statuses == {"a": store.STATUS_PENDING, "b": store.STATUS_PENDING}


def test_get_unknown_submission_returns_none(tmp_path):
    assert store.get_submission(tmp_path, "nope") is None


def test_update_image_result(tmp_path):
    sub_id = store.create_submission(tmp_path, None, {"a": "http://x/a.jpg"}, None)
    store.update_image_result(tmp_path, sub_id, "a", store.STATUS_DONE,
                               output_path="/out/a.png", category="exterior")
    sub = store.get_submission(tmp_path, sub_id)
    img = sub["images"][0]
    assert img["status"] == store.STATUS_DONE
    assert img["output_path"] == "/out/a.png"
    assert img["category"] == "exterior"
    assert img["warning"] is None


def test_submission_webhook_url_persisted(tmp_path):
    sub_id = store.create_submission(tmp_path, "VIN1", {"a": "http://x/a.jpg"}, "http://callback/hook")
    sub = store.get_submission(tmp_path, sub_id)
    assert sub["webhook_url"] == "http://callback/hook"
