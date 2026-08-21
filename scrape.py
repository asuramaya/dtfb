"""
Fetching and normalizing one Tomball Ford vehicle detail page (VDP) into a
Vehicle record.

The site fronts every page with a Cloudflare JS challenge, so we drive a
real (headless) Chromium via Playwright to load the page, then pull the
vehicle's full data out of an inline JSON blob the page already embeds for
its own analytics (window.jzlAnalyticsObject / vdp_gtm_payload).

Recipe / irregularity notes (read before changing extraction logic):
  - Not every vehicle has every field (used vs. new, in-transit vs. on-lot,
    EVs vs. ICE, trucks with bed length, etc). Every extraction is best-effort
    and missing data is simply omitted from the output, never a crash.
  - The embedded blob's key layout has been stable across new/used/EV/truck
    VDPs tested so far; if a future page redesign changes the variable name,
    `extract_analytics_object()` is the one place to add a fallback pattern.
"""
from __future__ import annotations

import dataclasses
import html
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests
# Playwright is imported lazily inside fetch_rendered_html()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

# The vehicle blob is embedded as the default-parameter value of a JS
# function call: `jzlGa4AttachListenersToForms(jzlAnalyticsObject = {...})`.
DEALERINSPIRE_VAR_MARKER = "jzlAnalyticsObject = "

IMAGE_RESIZE_RE = re.compile(r"/resize/\d+x\d+/")
TARGET_IMAGE_SIZE = "2048x2048"


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

CARFAX_WAIT_MS = 12000


def fetch_rendered_html(page, url: str, retries: int = 4) -> str:
    """Load a URL in the given Playwright page and return the rendered HTML.
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    Cloudflare's managed challenge usually auto-resolves in a couple seconds
    for a real (even headless) Chromium with a normal UA -- no click/captcha
    needed -- but back-to-back navigations in the same session sometimes draw
    a harder/slower challenge. We poll the title and retry with backoff
    rather than assume a fixed resolve time.

    Once past that, on a used/CPO vehicle, the Carfax badge widget's actual
    report link isn't in the initial render -- its JS controller fills in
    the <a href> after an async lookup, confirmed by watching a real fetch:
    the container div (with data-vin/data-event-details) is present
    immediately, but empty, and the link only appears ~1-3s later.
    Capturing HTML right away silently drops the link (confirmed real case:
    extract_carfax_url() found nothing on a CPO Raptor that visibly has a
    working Carfax badge in a browser).

    The wait is gated on "used" appearing in the URL slug itself (e.g.
    .../vehicle/VIN/Used-2023-Ford-F--150-.../ vs .../vehicle/VIN/2026-Ford-
    Mustang-.../ for a new one) -- NOT on the ".carfax-logo" div existing:
    confirmed real case, that container renders on EVERY vehicle page, new
    or used, with data-vin already filled in either way, so it can't tell
    "used, report pending" apart from "new, no report will ever come."
    Gating on the div alone meant paying the full CARFAX_WAIT_MS timeout on
    every single new-vehicle fetch (the majority of the inventory) for a
    link that was never going to appear. The URL slug's used/new marker
    matched every vehicle's real condition across all 17 already on disk
    (Certified Pre-Owned and Used both carry it, New never does) -- cheaper
    to check than parsing the page for the same fact, since it costs
    nothing to look at before the page even loads.
    """
    is_used = "used" in url.lower()
    last_err = None
    for attempt in range(retries + 1):
        try:
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            for _ in range(20):
                if "Just a moment" not in page.title():
                    if is_used:
                        try:
                            page.wait_for_selector(".carfax-logo a", timeout=CARFAX_WAIT_MS)
                        except PlaywrightTimeoutError:
                            pass
                    return page.content()
                time.sleep(1)
            last_err = RuntimeError("Cloudflare challenge did not clear in time")
        except PlaywrightTimeoutError as e:
            last_err = e
        time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"Failed to load {url}: {last_err}")


def extract_balanced_json(html: str, marker: str) -> dict | None:
    """Extract a JSON object embedded at `marker` by matching balanced braces.

    Regex can't reliably match nested JSON, so we scan character-by-character
    from the first '{' after the marker, tracking string state, until braces
    balance back to zero.
    """
    idx = html.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    while start < len(html) and html[start] != "{":
        start += 1
    if start >= len(html):
        return None

    depth = 0
    in_str = False
    esc = False
    i = start
    while i < len(html):
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
        i += 1
    blob = html[start:i]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# CMS extractor registry
# --------------------------------------------------------------------------
# Each extractor is a (marker, validator) pair.  marker is the JS variable
# name to search for in the page HTML; validator checks that the extracted
# JSON is a real vehicle blob (not a JS function wrapper or unrelated data).
# Add new CMS platforms by appending to this list.
#
# Extractor functions take (html) -> dict|None and are tried in order until
# one returns a truthy result.

_EXTRACTORS: list[tuple[str, str, callable]] = []


def register_extractor(name: str, marker: str, validator: callable) -> None:
    """Register a CMS-specific extractor for the analytics blob.
    
    Args:
        name: Human-readable CMS name (e.g. "dealerinspire").
        marker: The JS variable marker string to search for in the HTML.
        validator: A callable(raw_dict) -> bool that returns True if the
                   extracted JSON looks like a real vehicle blob for this CMS.
    """
    _EXTRACTORS.append((name, marker, validator))


def _dealerinspire_validator(data: dict) -> bool:
    """DealerInspire sites embed a blob with a vdp_gtm_payload key."""
    return bool(data and "vdp_gtm_payload" in data)


register_extractor("dealerinspire", DEALERINSPIRE_VAR_MARKER, _dealerinspire_validator)


def extract_analytics_object(html: str) -> tuple[dict | None, str | None]:
    """Try every registered CMS extractor, return the first match.
    
    Returns:
        (data, cms_name) where data is the parsed JSON dict and cms_name is
        the name of the extractor that matched (e.g. "dealerinspire").
        (None, None) if no extractor matched.
    """
    for name, marker, validator in _EXTRACTORS:
        data = extract_balanced_json(html, marker)
        if data and validator(data):
            return data, name
    return None, None


CARFAX_LINK_RE = re.compile(r'class="carfax-logo"[^>]*>\s*<a href="([^"]+)"')


def extract_carfax_url(html: str) -> str | None:
    """The Carfax report link, pulled directly from the rendered ".carfax-
    logo" widget rather than the analytics blob's certifications.carfaxUrl
    field -- confirmed unreliable on real pages (saw certifications.
    hasCarfax=false / carfaxUrl=null on a CPO Raptor that visibly has a
    working Carfax badge in a browser). The widget's own <a href> is the
    real signal, and only appears once fetch_rendered_html()'s wait for
    ".carfax-logo a" has resolved -- on an un-waited fetch this reliably
    returns None even when a report exists, not just missing data."""
    m = CARFAX_LINK_RE.search(html)
    return m.group(1) if m else None


ONE_OWNER_RE = re.compile(r"\b1[\s-]*owner\b", re.IGNORECASE)


def extract_carfax_one_owner(features_structured: dict, main_features: list) -> bool:
    """Same bug as carfax_url, same fix: certifications.carfaxOneOwner is
    the same unreliable pre-Carfax-lookup snapshot (confirmed False on
    EVERY used vehicle checked, including ones that plainly carry a "1
    OWNER" badge elsewhere on the same page) -- so don't read it.

    What IS reliable: the dealer stamps "1 OWNER" as a literal tag string
    inside features.featuresStructured (rendered straight into the page's
    "Main Features" list) whenever Carfax's own report says so. Confirmed
    on two real vehicles: one shows both "CLEAN CARFAX" and "1 OWNER", a
    different one shows "CLEAN CARFAX" alone -- so the tag is a genuine
    per-vehicle signal, not boilerplate every used listing gets. Checked
    across every category (not just "Main Features") and against
    main_features too, since a future page layout could move it."""
    haystacks = main_features + [tag for tags in features_structured.values() for tag in tags]
    return any(ONE_OWNER_RE.search(tag) for tag in haystacks if isinstance(tag, str))


def extract_ldjson_car(html: str) -> dict | None:
    """Fallback: schema.org Car block. Thinner than the analytics blob but
    a useful cross-check / fallback for VIN, color, engine, description."""
    for m in re.finditer(
        r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        graph = data.get("@graph", [data]) if isinstance(data, dict) else data
        for node in graph:
            if isinstance(node, dict) and node.get("@type") == "Car":
                return node
    return None


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Vehicle:
    url: str
    vin: str | None = None
    stock_number: str | None = None
    year: str | None = None
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    condition: str | None = None  # New / Used / Certified
    vehicle_status: str | None = None  # On Lot / In Transit / etc
    mileage: int | None = None
    title: str | None = None

    exterior_color_factory: str | None = None
    exterior_color_generic: str | None = None
    interior_color: str | None = None

    engine: str | None = None
    transmission: str | None = None
    drivetrain: str | None = None
    fuel_type: str | None = None
    mpg_city: str | None = None
    mpg_highway: str | None = None
    body_type: str | None = None
    ev_battery_range: str | None = None
    ev_mpge_combined: str | None = None
    cab_style: str | None = None
    box_length: str | None = None

    display_price: str | None = None
    pricing_rows: list = dataclasses.field(default_factory=list)

    dealer_description: str | None = None
    dealer_comments: str | None = None

    main_features: list = dataclasses.field(default_factory=list)
    features_structured: dict = dataclasses.field(default_factory=dict)
    options: list = dataclasses.field(default_factory=list)
    tags: list = dataclasses.field(default_factory=list)

    dealer_name: str | None = None
    dealer_address: str | None = None
    dealer_phone: str | None = None

    photo_urls: list = dataclasses.field(default_factory=list)
    video_urls: list = dataclasses.field(default_factory=list)
    window_sticker_url: str | None = None
    sticker: dict | None = None  # structured data from imaging/sticker.py,
    # when a real window sticker was available and parsed successfully --
    # build_facebook_post() prefers this over main_features/features_structured
    # when present. None means fall back (used vehicles, unpublished stickers,
    # or a parse failure all leave this None -- see window_sticker.py).

    carfax_url: str | None = None
    carfax_one_owner: bool | None = None

    warnings: list = dataclasses.field(default_factory=list)


def _nonzero(value) -> bool:
    """True if value is a present, non-zero number/numeric-string.
    Several dealer.com fields (MPG, price) use "0" as a "not set / not
    rated" placeholder rather than omitting the key entirely."""
    if value is None or value == "":
        return False
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        return True  # non-numeric, non-empty -- treat as present


def g(d: dict | None, *path, default=None):
    """Safe nested-get: g(d, 'a', 'b') ~= d.get('a', {}).get('b')."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


TAG_RE = re.compile(r"<[^>]+>")
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def clean_html_text(text: str | None) -> str | None:
    """Dealer-written descriptions come through with embedded HTML (<br><br>
    for paragraph breaks, occasional stray tags). Facebook posts are plain
    text, so turn <br> into real newlines, strip anything else tag-shaped,
    and unescape entities."""
    if not text:
        return text
    text = BR_RE.sub("\n", text)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def upsize_image_url(url: str) -> str:
    if IMAGE_RESIZE_RE.search(url):
        return IMAGE_RESIZE_RE.sub(f"/resize/{TARGET_IMAGE_SIZE}/", url)
    return url


def vin_from_url(url: str) -> str | None:
    """The VIN segment of a .../vehicle/<VIN>/... VDP URL, or None if the
    URL isn't shaped like one. Used to catch the case where the page we
    actually landed on isn't the vehicle the URL named -- see
    vehicle_pipeline.py::process_vehicle()."""
    parts = urlparse(url).path.strip("/").split("/")
    for i, part in enumerate(parts[:-1]):
        if part.lower() == "vehicle":
            return parts[i + 1]
    return None


def normalize_vehicle(url: str, html: str) -> Vehicle:
    v = Vehicle(url=url)
    analytics, cms_name = extract_analytics_object(html)
    ld_car = extract_ldjson_car(html)

    if analytics is None:
        v.warnings.append(
            "Could not find embedded vehicle data blob; "
            "falling back to schema.org data only, most fields will be empty."
        )

    payload = g(analytics, "vdp_gtm_payload", default={}) if analytics else {}

    v.vin = payload.get("vin") or g(ld_car, "vehicleIdentificationNumber")
    v.stock_number = payload.get("stockNumber")
    v.year = str(payload.get("year")) if payload.get("year") else None
    v.make = payload.get("make")
    v.model = payload.get("model")
    v.trim = payload.get("trim")
    # `conditions`/`conditions_array` mix real condition with dealer/manufacturer
    # marketing badges (e.g. "Blue Certified,Certified,Used", or "Gold Certified"
    # on a vehicle that isn't actually manufacturer/dealer certified) -- so we
    # derive the label from the boolean flags instead of trusting that string.
    certs = payload.get("certifications") or {}
    is_certified = bool(certs.get("certifiedByManufacturer") or certs.get("certifiedByDealer"))
    if payload.get("used") is False:
        v.condition = "New"
    elif payload.get("used") is True:
        v.condition = "Certified Pre-Owned" if is_certified else "Used"
    else:
        conditions = payload.get("conditions_array") or []
        v.condition = conditions[-1] if conditions else None
    v.vehicle_status = payload.get("vehicleStatus")
    v.mileage = payload.get("mileage")

    title_bits = [v.year, v.make, v.model, v.trim]
    v.title = " ".join(b for b in title_bits if b) or g(ld_car, "name")

    colors = g(payload, "visual", "colors", default={})
    v.exterior_color_factory = g(colors, "exterior", "factory") or g(ld_car, "color")
    v.exterior_color_generic = g(colors, "exterior", "generic")
    v.interior_color = g(colors, "interior", "factory")

    specs = payload.get("specifications") or {}
    v.engine = specs.get("engine") or g(ld_car, "vehicleEngine", "name")
    transmission = specs.get("transmission_new") or specs.get("transmission")
    v.transmission = transmission.strip() if transmission else transmission
    v.drivetrain = specs.get("drivetrain")
    v.fuel_type = specs.get("fuelType") or g(ld_car, "fuelType")
    # "0" city/hwy means "not EPA-rated" (EVs use MPGe, some heavy trucks
    # aren't rated at all) rather than an actual 0 MPG, so don't surface it.
    if _nonzero(specs.get("mpgCityLow")) or _nonzero(specs.get("mpgCityHigh")):
        lo, hi = specs.get("mpgCityLow"), specs.get("mpgCityHigh")
        v.mpg_city = lo if lo == hi else f"{lo}-{hi}"
    if _nonzero(specs.get("mpgHighwayLow")) or _nonzero(specs.get("mpgHighwayHigh")):
        lo, hi = specs.get("mpgHighwayLow"), specs.get("mpgHighwayHigh")
        v.mpg_highway = lo if lo == hi else f"{lo}-{hi}"
    body_types = specs.get("vehicleType") or []
    v.body_type = ", ".join(body_types) if body_types else g(ld_car, "bodyType ") or g(ld_car, "bodyType")
    v.ev_battery_range = payload.get("evBatteryRange")
    v.ev_mpge_combined = payload.get("evMpgCombined")
    v.cab_style = payload.get("cabStyle") or None
    v.box_length = payload.get("boxLength")

    v.display_price = payload.get("displayPrice") or payload.get("price")
    if not _nonzero(v.display_price):
        v.display_price = None
    for row in payload.get("flattened_pricing") or []:
        pr = row.get("pricing") or {}
        label, text = pr.get("Label"), pr.get("Text")
        if label and text:
            v.pricing_rows.append({"label": label, "text": text})

    v.dealer_description = clean_html_text(payload.get("dealerDescription") or g(ld_car, "description"))
    v.dealer_comments = clean_html_text(payload.get("dealerComments")) or None

    features = payload.get("features") or {}
    v.main_features = features.get("mainFeatures") or []
    v.features_structured = features.get("featuresStructured") or {}
    v.options = payload.get("options") or []
    v.tags = payload.get("tags") or []

    loc = payload.get("location") or {}
    v.dealer_name = loc.get("name")
    addr_bits = [loc.get("address1"), loc.get("address2")]
    city_state_zip = ", ".join(
        b for b in [loc.get("city"), loc.get("state")] if b
    )
    if loc.get("zipCode"):
        city_state_zip = f"{city_state_zip} {loc['zipCode']}".strip()
    addr_bits = [b for b in addr_bits if b] + ([city_state_zip] if city_state_zip else [])
    v.dealer_address = ", ".join(addr_bits) if addr_bits else None
    v.dealer_phone = loc.get("contactNumber")

    visual = payload.get("visual") or {}
    combined = visual.get("combinedPhotos")
    if not combined:
        combined = (visual.get("dealerPhotos") or []) + (visual.get("stockPhotos") or [])
    seen = set()
    for photo in combined:
        src = photo.get("source") if isinstance(photo, dict) else None
        if src and src not in seen:
            seen.add(src)
            v.photo_urls.append(upsize_image_url(src))
    if not v.photo_urls:
        img = g(payload, "visual", "image", "source") or g(ld_car, "image")
        if img:
            v.photo_urls.append(upsize_image_url(img))

    for vid in visual.get("dealerVideos") or []:
        if isinstance(vid, dict) and vid.get("source"):
            v.video_urls.append(vid["source"])
        elif isinstance(vid, str):
            v.video_urls.append(vid)

    expando = payload.get("expando") or {}
    v.window_sticker_url = expando.get("WindowStickerUrl") or None

    v.carfax_url = extract_carfax_url(html)
    v.carfax_one_owner = extract_carfax_one_owner(v.features_structured, v.main_features)

    if not v.photo_urls:
        v.warnings.append("No photos found for this vehicle.")
    if not v.window_sticker_url:
        v.warnings.append("No window sticker URL found (common for used vehicles).")
    if not v.display_price:
        v.warnings.append("No price found.")

    return v


# --------------------------------------------------------------------------
# Output folder naming
# --------------------------------------------------------------------------

def slugify(*parts: str, maxlen: int = 80) -> str:
    text = "-".join(p for p in parts if p)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:maxlen] or "vehicle"


def condition_bucket(v: "Vehicle", url: str | None = None) -> str:
    """"new" or "used" -- the top-level split of the listings folder.

    CPO counts as used: a certified vehicle is a used vehicle with a
    warranty, and it's priced, posted and shopped as used (see
    facebook_post.py, which already treats anything non-new the same way).

    Prefers the scraped condition, falling back to the URL slug, which
    carries the same fact ("/Used-2023-Ford-F--150-..." vs
    "/2026-Ford-Mustang-...") and was validated 17/17 against real
    scraped conditions when the Carfax gate needed it. Unknown sorts to
    "used" rather than "new": mis-filing a used vehicle as new is the
    error that would put factory-warranty language on the wrong post.
    """
    condition = (v.condition or "").strip().lower()
    if condition:
        return "new" if condition == "new" else "used"
    if url:
        return "used" if "used" in url.lower() else "new"
    return "used"


def vehicle_folder_name(v: Vehicle) -> str:
    tail = (v.stock_number or v.vin or "unknown")[-8:]
    return slugify(v.year or "", v.make or "", v.model or "", v.trim or "", tail)


# --------------------------------------------------------------------------
# Generic download helper (shared by photos.py and window_sticker.py)
# --------------------------------------------------------------------------

def download_file(session: requests.Session, url: str, dest: Path, timeout=30) -> bool:
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"    ! download failed ({url}): {e}", file=sys.stderr)
        return False