"""
Short-form copy for the surfaces that aren't Marketplace.

facebook_post.py builds one post of about 2,500 characters, which is right
for a Marketplace listing where the reader has already clicked a specific
vehicle and wants the spec sheet. It is wrong everywhere else:

    Threads      500 characters, hard cap
    Instagram    2,200 allowed, but truncated at ~125 with "... more"
    Marketplace  no practical limit, reader is already committed

So the difference isn't length alone, it's what has to come FIRST. On
Marketplace the first three lines are a courtesy (greeting, vehicle,
address) because the reader is already looking at the car. On a feed the
first line is the only line most people read, so it has to carry the
vehicle and the reason to care, and the greeting moves to the end where it
becomes a call to action instead of a salutation.

Nothing here invents a claim. Every value is a scraped field, same as the
long post -- the only editorial decisions are ordering and what to drop
when the budget runs out. See TRIM_ORDER for the drop order.

Hashtags exist here and deliberately not in facebook_post.py: they're
expected on Threads and Instagram and read as spam on a Marketplace
listing, which is a merchandising difference between surfaces rather than
a formatting one.
"""
from __future__ import annotations

import re

from dtfb.dealer_config import get as _get_dealer_config
from dtfb.facebook_post import resolve_display_price
from dtfb.scrape import Vehicle

THREADS_LIMIT = 500
# Instagram cuts the caption here with a "... more" link. Everything that
# has to be read without a tap must fit inside it.
INSTAGRAM_VISIBLE = 125

# Below this, the odometer is delivery mileage and saying it out loud
# reads as odd rather than informative -- a real case shipped as
# "New 2027 Ford Expedition King Ranch - $90,460 - 2 mi". Keyed on the
# number rather than on condition=="New" so a new-but-high-mileage demo or
# loaner still shows its miles, which is exactly when a buyer wants them.
# The long Marketplace post still lists it either way: that one is a spec
# sheet where completeness is the point, this one is curated.
DELIVERY_MILEAGE_CEILING = 100

# City tags are now loaded via dealer_config.get().city_tags
MAX_HASHTAGS = 14
# Past this a tag stops being searchable and starts looking like a mistake.
# The real case: a full trim string produced
# "#VolkswagenAtlasCrossSport36LV6SEWTechnologyRLine".
MAX_TAG_LENGTH = 28

# Body types the dealer's own field uses, mapped to the tag people
# actually search. Anything not listed just gets its own name cleaned up.
BODY_TYPE_TAGS = {
    "suv": "SUV",
    "sport utility": "SUV",
    "truck": "Truck",
    "pickup": "Truck",
    "crew cab pickup": "Truck",
    "sedan": "Sedan",
    "coupe": "Coupe",
    "convertible": "Convertible",
    "hatchback": "Hatchback",
    "van": "Van",
    "cargo van": "WorkVan",
    "minivan": "Minivan",
    "wagon": "Wagon",
}


def _tag(text: str) -> str | None:
    """One hashtag from a phrase: alphanumerics only, words joined, so
    'F-150 Raptor' becomes 'F150Raptor'.

    A phrase that is already a single token is passed through untouched --
    str.capitalize() lowercases everything after the first letter, which
    silently destroyed the casing of tags written correctly at the call
    site ("CertifiedPreOwned" -> "Certifiedpreowned", "SUV" -> "Suv").

    Returns None for anything that empties out or ends up too long to be
    a usable tag.
    """
    parts = re.findall(r"[A-Za-z0-9]+", str(text or ""))
    if not parts:
        return None
    joined = parts[0] if len(parts) == 1 else "".join(
        p if p.isupper() else p.capitalize() for p in parts)
    return joined if len(joined) <= MAX_TAG_LENGTH else None


def build_hashtags(v: Vehicle, limit: int = MAX_HASHTAGS) -> list[str]:
    """Tags derived from what was scraped, most specific first.

    Ordered specific -> broad on purpose. A niche tag is worth more than a
    broad one (fewer posts to compete with, more intent behind the search),
    and ordering this way means truncating to fit a character budget drops
    the least valuable end.
    """
    tags: list[str] = []

    def add(value):
        t = _tag(value)
        if t and t not in tags:
            tags.append(t)

    if v.make and v.model:
        add(f"{v.make} {v.model}")
        if v.trim:
            add(f"{v.make} {v.model} {v.trim}")
        if v.year:
            add(f"{v.year} {v.make} {v.model}")
    add(v.make)

    # Substring match, longest key first: the dealer's own field carries
    # values like "Sport Utility/SUV/Crossover", which an exact lookup
    # missed and which then tagged itself verbatim.
    body = (v.body_type or "").strip().lower()
    if body:
        matched = next((tag for key, tag in sorted(BODY_TYPE_TAGS.items(),
                                                     key=lambda kv: -len(kv[0]))
                         if key in body), None)
        add(matched or body)

    condition = (v.condition or "").strip().lower()
    if condition.startswith("cert"):
        add("CertifiedPreOwned")
    elif condition == "new":
        add("NewCar")
    elif condition:
        add("UsedCars")

    for city_tag in _get_dealer_config().city_tags:
        add(city_tag)
    for generic in ("CarsForSale", "ForSale", "CarDealership"):
        add(generic)

    return [f"#{t}" for t in tags[:limit]]


def _price_phrase(v: Vehicle) -> str | None:
    resolved = resolve_display_price(v)
    if not resolved:
        return None
    try:
        return f"${int(float(str(resolved).replace(',', '').replace('$', ''))):,}"
    except ValueError:
        return str(resolved)


def _headline(v: Vehicle) -> str:
    condition = (v.condition or "").strip()
    prefix = "New " if condition.lower() == "new" else ""
    return f"{prefix}{v.title or 'Vehicle'}".strip()


def _showable_mileage(v: Vehicle) -> int | None:
    """The odometer, when it's worth a reader's attention."""
    if v.mileage is None or v.mileage < DELIVERY_MILEAGE_CEILING:
        return None
    return v.mileage


def _hook(v: Vehicle) -> str:
    """The first line, built to survive Instagram's ~125-character cut.

    Vehicle then price, because those are the two things a scroller
    decides on. Mileage only joins them when it still fits -- on a used
    vehicle it's the third thing people ask, but it is never worth pushing
    the price past the fold.
    """
    parts = [_headline(v)]
    price = _price_phrase(v)
    if price:
        parts.append(price)
    line = " — ".join(parts)
    mileage = _showable_mileage(v)
    if mileage is not None:
        with_miles = f"{line} — {mileage:,} mi"
        if len(with_miles) <= INSTAGRAM_VISIBLE:
            return with_miles
    return line


# What gets dropped first when the budget runs out. Least load-bearing at
# the top: the hook and the call to action are never in this list, because
# a post without them isn't a shorter post, it's a different one.
TRIM_ORDER = ["stock", "vin", "drivetrain", "engine", "color", "carfax", "mileage"]


def _city_state(addr: str) -> str:
    """Extract 'City, ST' from a full address string (e.g. '123 Main, Austin, TX 78701' -> 'Austin, TX')."""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if len(parts) >= 3:
        st = parts[2].split()[0]
        return f"{parts[1]}, {st}"
    elif len(parts) == 2:
        return f"{parts[0]}, {parts[1]}"
    return addr


def build_threads_post(v: Vehicle, limit: int = THREADS_LIMIT) -> str:
    """A Threads post inside the 500-character cap.

    Built by assembling every candidate line and then dropping them in
    TRIM_ORDER until it fits, rather than by truncating the finished text.
    Truncation produces a post that stops mid-sentence; dropping whole
    facts produces a shorter post that still reads as written.

    Hashtags are added last and only with whatever budget survives, so a
    verbose vehicle loses tags rather than losing the car.
    """
    hook = _hook(v)
    _cfg = _get_dealer_config()
    cta = f"{_cfg.dealer_greeting} {_city_state(_cfg.dealer_address)}"

    optional: dict[str, str] = {}
    # Only when the hook didn't already fit it -- otherwise the post says
    # the mileage twice, three lines apart.
    mileage = _showable_mileage(v)
    if mileage is not None and f"{mileage:,} mi" not in hook:
        optional["mileage"] = f"{mileage:,} miles"
    if v.exterior_color_factory:
        optional["color"] = str(v.exterior_color_factory)
    if v.engine:
        optional["engine"] = str(v.engine)
    if v.drivetrain:
        optional["drivetrain"] = str(v.drivetrain)
    if v.carfax_one_owner:
        optional["carfax"] = "CARFAX 1-Owner"
    if v.vin:
        optional["vin"] = f"VIN {v.vin}"
    if v.stock_number:
        optional["stock"] = f"Stock #{v.stock_number}"

    keep = [k for k in ("mileage", "carfax", "color", "engine", "drivetrain", "vin", "stock") if k in optional]

    def assemble(keys, tags):
        body = " · ".join(optional[k] for k in keys)
        blocks = [hook]
        if body:
            blocks.append(body)
        blocks.append(cta)
        if tags:
            blocks.append(" ".join(tags))
        return "\n\n".join(blocks)

    tags = build_hashtags(v, limit=6)
    for drop in TRIM_ORDER:
        if len(assemble(keep, tags)) <= limit:
            break
        if drop in keep:
            keep.remove(drop)
    while tags and len(assemble(keep, tags)) > limit:
        tags = tags[:-1]
    text = assemble(keep, tags)
    # Last resort: no combination of drops fits (an absurdly long title).
    # Cut on a word boundary rather than mid-word.
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" ·—\n")
    return text + "\n"


def build_instagram_caption(v: Vehicle) -> str:
    """An Instagram caption: hook above the fold, detail below it, tags last.

    Instagram allows 2,200 characters, so unlike Threads nothing needs to
    be dropped -- the whole job is ordering. The hook has to stand alone
    because most readers never expand the caption, and the hashtag block
    goes at the very bottom where it doesn't interrupt the copy.
    """
    blocks = [_hook(v)]

    detail = []
    mileage = _showable_mileage(v)
    if mileage is not None:
        detail.append(f"Mileage: {mileage:,} mi")
    if v.exterior_color_factory:
        detail.append(f"Exterior: {v.exterior_color_factory}")
    if v.interior_color:
        detail.append(f"Interior: {v.interior_color}")
    if v.engine:
        detail.append(f"Engine: {v.engine}")
    if v.transmission:
        detail.append(f"Transmission: {v.transmission}")
    if v.drivetrain:
        detail.append(f"Drivetrain: {v.drivetrain}")
    if v.mpg_city or v.mpg_highway:
        mpg = "/".join(filter(None, [v.mpg_city and f"{v.mpg_city} city",
                                      v.mpg_highway and f"{v.mpg_highway} hwy"]))
        detail.append(f"MPG: {mpg}")
    elif v.ev_mpge_combined:
        detail.append(f"MPGe: {v.ev_mpge_combined} combined")
    if v.ev_battery_range:
        detail.append(f"EV range: {v.ev_battery_range} mi")
    if v.carfax_one_owner:
        detail.append("CARFAX 1-Owner")
    if v.stock_number:
        detail.append(f"Stock #{v.stock_number}")
    if detail:
        blocks.append("\n".join(detail))

    _cfg = _get_dealer_config()
    blocks.append(f"{_cfg.dealer_greeting}\n{_cfg.dealer_address}")
    blocks.append(" ".join(build_hashtags(v)))
    return "\n\n".join(blocks) + "\n"
