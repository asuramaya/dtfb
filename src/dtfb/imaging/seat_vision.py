"""
Front-seat configuration (bucket+console vs. bench) from a real interior
photo, via a local vision model -- fills a gap the window-sticker text
can't: confirmed real case, a 2026 F-350SD Lariat's Monroney sticker never
discloses front-seat configuration for that trim at all (its equipment
grid mentions only "Htd/Ventilated Frt Seats"), so equipment_search.py has
nothing to search against for a query like "no folding bench" on that
vehicle even though the photo answers it.

Runs entirely local via Ollama (gemma4:e2b, ~2.9GB VRAM, ~7-11s/photo on
this GPU) -- no API key, no per-query cost, confirmed against real
interior photos before being wired in here: 2/2 clear cases (Transit-150,
F-150 Raptor) correctly identified bucket+console at high confidence
against sticker-confirmed ground truth, and the one genuinely ambiguous
photo (that same F-350) was correctly flagged as "can't tell" rather than
answered with a confident guess -- the failure mode that would have made
this untrustworthy to build on.

Requires `ollama serve` running locally (default http://127.0.0.1:11434)
with gemma4:e2b pulled. Advisory only, same pattern as the stock-render
check in vehicle_pipeline.py: if Ollama isn't running, the model isn't
pulled, or a single photo's response doesn't parse, this returns None and
callers fall back to whatever text-based signal they already had. Never
raises, never fails a scrape. NOT part of regression.py -- it depends on
a live local model server + GPU, which the standard suite doesn't assume.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma4:e2b"
REQUEST_TIMEOUT = 60  # a cold-loaded model's first call can take longer than the ~10s steady-state

SEAT_VISION_PROMPT = (
    "Look at this photo of a vehicle's interior. Does the FRONT row have "
    "individual bucket seats with a center console between them, or a bench "
    "seat (a continuous seat across, possibly split/folding)? "
    "If the photo doesn't show the front seats clearly enough to tell, say unclear. "
    'Respond with ONLY a JSON object, no other text: '
    '{"config": "bucket_console" | "bench" | "unclear", "confidence": "high" | "medium" | "low"}'
)

VALID_CONFIGS = {"bucket_console", "bench", "unclear"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def ollama_available() -> bool:
    """Cheap reachability check -- callers use this to skip the whole
    feature quietly rather than eating a connection-refused per photo."""
    try:
        resp = requests.get(f"{OLLAMA_URL.rsplit('/api/', 1)[0]}/api/version", timeout=2)
        return resp.ok
    except requests.RequestException:
        return False


def infer_seat_config(image_bytes: bytes) -> dict | None:
    """One photo -> {"config", "confidence"} or None on any failure
    (server unreachable, model not pulled, malformed response). The model
    is asked for bare JSON but small local models occasionally wrap it in
    prose or a code fence despite instructions -- confirmed worth guarding
    for, not a hypothetical, so this extracts the first {...} block rather
    than requiring the whole response to be clean JSON."""
    import base64

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": SEAT_VISION_PROMPT,
            "images": [base64.b64encode(image_bytes).decode()],
            "stream": False,
            "format": "json",
            # Unload the model from VRAM right after this call instead of
            # Ollama's default 5-minute keep-warm -- confirmed real
            # conflict without this: running gemma4:e2b resident alongside
            # the rest of this pipeline's GPU models (CLIP, rembg/BiRefNet,
            # SAM2/CLIPSeg for wheel routing) pushed a real regression run
            # over this GPU's 12GB and crashed with an ONNXRuntime OOM.
            # Costs a slower cold-load on the NEXT vehicle's call instead
            # (a few extra seconds) rather than starving whatever else is
            # using the GPU in between.
            "keep_alive": 0,
        }, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    except (requests.RequestException, ValueError):
        return None

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    config, confidence = parsed.get("config"), parsed.get("confidence")
    if config not in VALID_CONFIGS or confidence not in VALID_CONFIDENCE:
        return None
    return {"config": config, "confidence": confidence}


def pick_front_cabin_photo(interior_dir: Path, classifier=None) -> Path | None:
    """The single most likely wide front-cabin shot in a vehicle's interior
    gallery -- worth spending exactly one CLIP pass (cheap) to pick well
    rather than guessing by filename order, since a low-numbered photo is
    NOT reliably a good candidate (confirmed real case: a real Raptor's
    01.jpg is a rear-bench shot; 03.jpg is the actual dashboard view).
    Falls back to the first photo on disk if no classifier is given or
    none scores as "dashboard" -- still better than nothing."""
    interior_dir = Path(interior_dir)
    photos = sorted(interior_dir.glob("*.jpg"))
    if not photos:
        return None
    if classifier is None:
        return photos[0]

    best_path, best_score = None, -1.0
    for path in photos:
        try:
            result = classifier.classify(path.read_bytes())
        except Exception:
            continue
        if result.label == "dashboard" and result.confidence > best_score:
            best_path, best_score = path, result.confidence
    return best_path or photos[0]


def extract_seat_config(vehicle_folder: Path, classifier=None) -> dict | None:
    """End-to-end: pick the best candidate photo from this vehicle's
    interior gallery, run it through the vision model, return the result
    plus which photo was used (for spot-checking) -- or None if there's no
    interior gallery, Ollama isn't reachable, or the model's answer didn't
    parse. Callers should treat None as "no opinion", not "no console"."""
    vehicle_folder = Path(vehicle_folder)
    interior_dir = vehicle_folder / "images" / "interior"
    if not interior_dir.is_dir():
        return None

    photo = pick_front_cabin_photo(interior_dir, classifier)
    if photo is None or not ollama_available():
        return None

    result = infer_seat_config(photo.read_bytes())
    if result is None:
        return None
    result["source_photo"] = f"images/interior/{photo.name}"
    result["model"] = MODEL_NAME
    return result
