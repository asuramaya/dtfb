"""
Server mode: a CarCutter-API-shaped HTTP surface in front of lotstretcher's own
imaging pipeline, self-hosted.

WHY THIS EXISTS (roadmap item 5, strategy decision 1449fdef, lotstretcher project
graph): CarCutter's own documented API (cloud.car-cutter.com/doc/
api.html) is the shape a lot of dealer-software integrations already
speak. lotstretcher's core pipeline (classify, cutout, composite) does the same
job, self-hosted and fully open -- no data ever leaves your own server,
and every line of the model/processing code is readable, unlike a closed
SaaS API. This exposes that pipeline behind a compatible enough surface
that swapping the base URL is a realistic option, not a rewrite.

TWO MODES, ONE PIPELINE: this is a genuinely different way of running
lotstretcher than the CLI (a long-lived process instead of a one-shot script),
which is why it's a separate, OPT-IN dependency group (`pip install
-e ".[server]"`) -- installing lotstretcher to run it as a script never drags in
a web framework. See CONTRIBUTING.md's "CLI Only" note, revised to
document both modes rather than silently contradicted by this file.

SCOPE, STATED HONESTLY: this implements the CORE image-processing surface
(submission/status/result, single-segment) plus a light vehicle registry
(submission/list/status/delete/shotlist) end to end, backed by SQLite and
a thread pool -- not every field CarCutter's docs mention. guideline_id,
location_id, processing_speed, and retouching_accuracy are ACCEPTED (for
request-shape compatibility) but not yet wired to different behavior;
cut_type="blur" currently maps to the same treatment as "complete" (no
blur treatment built yet). Each such gap is a real, trackable follow-up,
not a silent no-op -- see the response's own "warning" field where one
applies. Extend FEED_COLUMNS-style: this is a small, flat request/
response schema (below), not scattered logic, specifically so a real
field difference is cheap to fix once one surfaces in practice.

Run:
    pip install -e ".[server]"
    lotstretcher-serve --data-dir ~/.lotstretcher-server --out ~/.lotstretcher-server/images
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import store
from .processing import fetch_image, process_image

MAX_IMAGES_PER_SUBMISSION = 60

app = FastAPI(title="lotstretcher server", description="Self-hosted, CarCutter-API-shaped vehicle imaging service")

# Populated by run()/configure() before the app serves traffic. Not a
# FastAPI dependency-injection setup on purpose -- this mirrors every
# other CLI entry point in this codebase (module-level state set once at
# startup), which keeps this file readable next to them rather than
# introducing a different pattern just for the server.
_state: dict = {}


def configure(data_dir: Path, out_dir: Path, max_workers: int = 4) -> None:
    from ..imaging.classify import InteriorExteriorTiebreakClassifier, SceneClassifier, default_backbone

    backbone = default_backbone()
    _state["data_dir"] = Path(data_dir)
    _state["out_dir"] = Path(out_dir)
    _state["classifier"] = SceneClassifier(backbone=backbone)
    _state["tiebreak"] = InteriorExteriorTiebreakClassifier(backbone=backbone)
    _state["executor"] = ThreadPoolExecutor(max_workers=max_workers)


# -- Request/response schemas -------------------------------------------------

class VehicleSubmission(BaseModel):
    vehicle_id: str
    metadata: dict = {}


class ImageEntry(BaseModel):
    id: str
    url: str


class ImageSubmission(BaseModel):
    vehicle_id: Optional[str] = None
    images: list[ImageEntry]
    cut_type: str = "complete"
    scene_id: Optional[str] = None
    guideline_id: Optional[str] = None       # accepted, not yet behavioral -- see module docstring
    location_id: Optional[str] = None        # accepted, not yet behavioral
    processing_speed: Optional[str] = None   # accepted, not yet behavioral
    retouching_accuracy: Optional[str] = None  # accepted, not yet behavioral
    webhook_url: Optional[str] = None


class SingleSegmentRequest(BaseModel):
    image_url: str
    cut_type: str = "complete"
    scene_id: Optional[str] = None


# -- Vehicle registry ----------------------------------------------------------

@app.post("/vehicle/submission")
def submit_vehicle(body: VehicleSubmission):
    store.upsert_vehicle(_state["data_dir"], body.vehicle_id, body.metadata)
    return {"vehicle_id": body.vehicle_id, "status": store.STATUS_NEW}


@app.post("/vehicle/list")
def list_vehicles(status: Optional[str] = None):
    return {"vehicles": store.list_vehicles(_state["data_dir"], status=status)}


@app.get("/vehicle/status")
def vehicle_status(vehicle_id: str):
    v = store.get_vehicle(_state["data_dir"], vehicle_id)
    if v is None:
        raise HTTPException(404, f"Unknown vehicle_id {vehicle_id!r}")
    return v


@app.delete("/vehicle/delete")
def delete_vehicle(vehicle_id: str):
    if not store.delete_vehicle(_state["data_dir"], vehicle_id):
        raise HTTPException(404, f"Unknown vehicle_id {vehicle_id!r}")
    return {"vehicle_id": vehicle_id, "deleted": True}


@app.get("/vehicle/shotlist")
def shotlist():
    """Vehicles still in "new" status -- CarCutter's own semantics for
    this endpoint (vehicles awaiting a photo shoot) don't map cleanly onto
    lotstretcher's pipeline (there's no separate "shoot" step; images arrive by
    URL, not an in-person capture app), so this is kept as the closest
    honest equivalent: registered vehicles nothing has been submitted
    for yet, not a literal port of CarCutter's own shoot-scheduling
    semantics."""
    return {"vehicles": store.list_vehicles(_state["data_dir"], status=store.STATUS_NEW)}


# -- Image processing: sync ----------------------------------------------------

@app.post("/vehicle/composition/single-segment")
def single_segment(body: SingleSegmentRequest):
    if body.cut_type not in ("complete", "normal", "blur", "none"):
        raise HTTPException(422, f"Unknown cut_type {body.cut_type!r}")
    try:
        content = fetch_image(body.image_url)
    except Exception as e:
        raise HTTPException(400, f"Could not fetch image_url: {e}")

    result = process_image(content, _state["out_dir"], f"single-{abs(hash(body.image_url))}",
                            cut_type=body.cut_type, scene_id=body.scene_id,
                            classifier=_state["classifier"], tiebreak_classifier=_state["tiebreak"])
    return {
        "category": result.category,
        "clip_label": result.clip_label,
        "clip_confidence": result.clip_confidence,
        "output_path": str(result.output_path) if result.output_path else None,
        "warning": result.warning,
    }


# -- Image processing: async batch ----------------------------------------------

def _run_submission(submission_id: str) -> None:
    data_dir, out_dir = _state["data_dir"], _state["out_dir"]
    sub = store.get_submission(data_dir, submission_id)
    if sub is None:
        return
    for img in sub["images"]:
        store.update_image_result(data_dir, submission_id, img["image_id"], store.STATUS_PROCESSING)
        try:
            content = fetch_image(img["url"])
            result = process_image(content, out_dir, f"{submission_id}-{img['image_id']}",
                                    classifier=_state["classifier"], tiebreak_classifier=_state["tiebreak"])
            store.update_image_result(data_dir, submission_id, img["image_id"], store.STATUS_DONE,
                                       output_path=str(result.output_path) if result.output_path else None,
                                       category=result.category, warning=result.warning)
        except Exception as e:
            store.update_image_result(data_dir, submission_id, img["image_id"], store.STATUS_FAILED,
                                       warning=str(e))

    if sub["webhook_url"]:
        import requests
        try:
            requests.post(sub["webhook_url"],
                          json=store.get_submission(data_dir, submission_id), timeout=15)
        except Exception:
            pass  # best-effort -- the client can still poll /vehicle/image/result


@app.post("/vehicle/image/submission")
def submit_images(body: ImageSubmission):
    if not body.images:
        raise HTTPException(422, "images must be non-empty")
    if len(body.images) > MAX_IMAGES_PER_SUBMISSION:
        raise HTTPException(422, f"At most {MAX_IMAGES_PER_SUBMISSION} images per submission")

    submission_id = store.create_submission(
        _state["data_dir"], body.vehicle_id, {img.id: img.url for img in body.images}, body.webhook_url)
    _state["executor"].submit(_run_submission, submission_id)
    return {"submission_id": submission_id}


@app.get("/vehicle/image/status")
def image_status(submission_id: str):
    sub = store.get_submission(_state["data_dir"], submission_id)
    if sub is None:
        raise HTTPException(404, f"Unknown submission_id {submission_id!r}")
    return {"submission_id": submission_id,
            "images": [{"id": i["image_id"], "status": i["status"]} for i in sub["images"]]}


@app.get("/vehicle/image/result")
def image_result(submission_id: str):
    sub = store.get_submission(_state["data_dir"], submission_id)
    if sub is None:
        raise HTTPException(404, f"Unknown submission_id {submission_id!r}")
    return {"submission_id": submission_id,
            "images": [{"id": i["image_id"], "status": i["status"], "output_path": i["output_path"],
                        "category": i["category"], "warning": i["warning"]} for i in sub["images"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="~/.lotstretcher-server", help="Where the SQLite registry lives")
    parser.add_argument("--out-dir", default=None, help="Where processed images are saved "
                                                          "(default: <data-dir>/images)")
    parser.add_argument("--workers", type=int, default=4, help="Async batch worker thread pool size")
    args = parser.parse_args()

    import uvicorn

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_dir / "images"
    configure(data_dir, out_dir, max_workers=args.workers)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
