"""
Build the ready-to-paste Facebook post for a vehicle.
"""
from __future__ import annotations

import re

from dealer_config import get as _get_dealer_config
from scrape import Vehicle

# Fixed dealer/salesperson greeting + address for every FB post -- this is
# the dealer's own boilerplate, not vehicle data, so it isn't sourced from
# the scrape or the window sticker (matches the contact bar baked into
# assets/borders/tomball-ford-hector-chavez.png). Goes FIRST/THIRD in the
# post (sandwiching the vehicle headline), not last: Facebook's feed
# preview only shows a post's first ~3 lines before truncating behind "See
# more", and this is what should be visible without a click. No phone
# number -- Facebook hides phone numbers typed into post text, so listing
# one here is dead text.
# New-vehicle posts are always priced at MSRP -- dealer incentives/discounts
# aren't advertised in the post itself (people call/visit to find out).
_INCENTIVE_KEYWORDS = ("price includes", "rebate", "incentive", "discount", "cash allowance")


def _looks_like_incentive_disclosure(text: str) -> bool:
    lower = text.lower()
    return "$" in text and any(k in lower for k in _INCENTIVE_KEYWORDS)


def resolve_display_price(v: Vehicle) -> str | None:
    """Used vehicles just show the listed price -- there's no MSRP concept
    for them. New vehicles always post at MSRP: prefer the window sticker's
    own Total MSRP (independently confirmed against the site's own "MSRP"
    pricing_rows label on every vehicle checked so far), then that
    pricing_rows entry, and only fall back to the site's live/discounted
    display_price if a new vehicle has neither (no sticker yet and the site
    didn't surface an MSRP row either)."""
    if (v.condition or "").strip().lower() != "new":
        return v.display_price
    sticker_msrp = (v.sticker or {}).get("pricing", {}).get("total_msrp")
    if sticker_msrp:
        return sticker_msrp
    for row in v.pricing_rows:
        if row.get("label", "").strip().upper() == "MSRP":
            return row.get("text")
    return v.display_price


def resolve_original_msrp_comparison(v: Vehicle) -> str | None:
    """For a used/CPO vehicle with a real window sticker, the sticker's
    Total MSRP reflects what it listed for brand new -- worth showing next
    to the current asking price as a value comparison (confirmed real
    example: a 2023 Raptor with an $87,980 original MSRP now listed at
    $72,623 -- the dealer sells at the listed price, not MSRP, for
    anything that isn't new, but the MSRP is still useful context).

    New vehicles already show MSRP as their primary price (see
    resolve_display_price()), so a separate comparison line there would
    just repeat the same number. Also returns None if there's no sticker,
    no MSRP figure, or the MSRP isn't actually higher than the current
    price -- a used vehicle priced at or above its original MSRP (rare,
    but possible in a high-demand market) wouldn't read as a meaningful
    "was/now" comparison, it'd just look like mismatched data."""
    if (v.condition or "").strip().lower() == "new":
        return None
    sticker_msrp = (v.sticker or {}).get("pricing", {}).get("total_msrp")
    if not sticker_msrp:
        return None
    current = resolve_display_price(v)
    try:
        msrp_num = float(str(sticker_msrp).replace(",", "").replace("$", ""))
        current_num = float(str(current).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None
    if msrp_num <= current_num:
        return None
    return f"${msrp_num:,.0f}"


def _to_float(text) -> float | None:
    try:
        return float(str(text).replace(",", "").replace("$", "").lstrip("+"))
    except (TypeError, ValueError):
        return None


def check_pricing_consistency(v: Vehicle) -> list[str]:
    """Lightweight sanity checks on the site's own pricing data -- flags a
    mismatch as a warning, never blocks anything, so a bad/inconsistent
    listing gets caught before posting instead of after. The site isn't
    consistent about how it labels or signs discount-vs-fee rows (a real
    example: two "Retail Customer Cash" rows with no +/- sign at all, that
    still needed subtracting to make the total add up), so this doesn't
    try to fully reconstruct the arithmetic -- just checks things that
    should always hold no matter the sign convention used that day."""
    issues = []

    if v.pricing_rows and v.display_price is not None:
        display_num = _to_float(v.display_price)
        last_row_num = _to_float(v.pricing_rows[-1].get("text"))
        if display_num is not None and last_row_num is not None and abs(display_num - last_row_num) > 1:
            issues.append(
                f"Pricing mismatch: display_price (${display_num:,.0f}) doesn't match the last "
                f"pricing_rows entry {v.pricing_rows[-1].get('label')!r} (${last_row_num:,.0f})."
            )

    if (v.condition or "").strip().lower() != "new":
        sticker_msrp = (v.sticker or {}).get("pricing", {}).get("total_msrp")
        if sticker_msrp:
            msrp_num = _to_float(sticker_msrp)
            current_num = _to_float(resolve_display_price(v))
            if msrp_num is not None and current_num is not None and msrp_num <= current_num:
                issues.append(
                    f"Sticker MSRP (${msrp_num:,.0f}) is not higher than the current listed price "
                    f"(${current_num:,.0f}) -- unusual for a used/CPO vehicle, worth a manual check."
                )

    return issues


def ford_qr_link(vin: str) -> str:
    """Ford's own short-link redirector for the QR code printed on every
    Monroney window sticker -- format confirmed from a real scanned QR
    (http://v.ford.com/?v=<VIN>&c=1&s=1, expanding server-side to a full
    ford.com inventory VDP with spotBuy/qrcid tracking params). c=1&s=1
    look like fixed template constants shared across vehicles, not
    per-VIN, based on the one real sample seen -- not independently
    verified further: ford.com's own site throttled/blackholed repeated
    probe requests almost immediately, well beyond what tomballford.com's
    Cloudflare challenge ever did, so there was no way to confirm this
    resolves for an arbitrary VIN without hammering their edge. Treat this
    as a best-effort bonus link for the post, not verified data -- if it's
    ever wrong, that's a link a reader clicks, not a fact dtfb asserts."""
    return f"http://v.ford.com/?v={vin}&c=1&s=1"


def _ford_more_details_link(v: Vehicle) -> str | None:
    """Ford's QR short-link only exists on a real Ford Monroney sticker --
    gated on v.sticker (only set once one was actually found and parsed,
    see window_sticker.py::parse_window_sticker()) rather than v.condition."""
    if not (v.sticker and v.vin):
        return None
    return ford_qr_link(v.vin)


# This dealer's used inventory isn't all Ford (Dodge/Kia/GMC trade-ins show
# up too), and other manufacturers may have their own equivalent QR/short-
# link scheme on their own window stickers -- unconfirmed so far. Keyed by
# make (lowercased) so a resolver can be dropped in the moment one's found,
# without touching the dispatch logic below. No entry for a make just means
# "more details" is omitted from the post -- never falls back to the
# dealership URL (see build_facebook_post()).
MANUFACTURER_MORE_DETAILS_LINKS = {
    "ford": _ford_more_details_link,
}


def resolve_more_details_link(v: Vehicle) -> str | None:
    resolver = MANUFACTURER_MORE_DETAILS_LINKS.get((v.make or "").strip().lower())
    return resolver(v) if resolver else None


def _feature_key(feature: str) -> str:
    """Normalised form for spotting a feature already listed.

    Case- and punctuation-insensitive because the two sources disagree on
    both: the window sticker's parser TitleCases its output while the
    site's own feature list keeps the manufacturer's casing, which shipped
    posts reading "LED Fog Lamps" and "Led Fog Lamps" two lines apart.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(feature).lower()).strip()


def build_facebook_post(v: Vehicle) -> str:
    lines = []
    # A feature can legitimately appear in more than one sticker category
    # (and again in main_features), but a reader just sees it twice --
    # measured at 18 repeated lines across the fleet's 1,862. First
    # occurrence wins, so it stays under the most specific heading.
    seen_features: set[str] = set()

    def new_features(feats):
        out = []
        for f in feats or []:
            key = _feature_key(f)
            if key and key not in seen_features:
                seen_features.add(key)
                out.append(f)
        return out

    # "Used"/"Certified Pre-Owned" aren't selling points worth the headline
    # space -- only "New" is called out; a used/CPO vehicle's headline just
    # reads as the bare title.
    condition_word = (v.condition or "").strip()
    headline = f"{'New ' if condition_word.lower() == 'new' else ''}{v.title or 'Vehicle'}".strip()

    # First 3 lines are what Facebook's feed preview shows before "See
    # more" truncates the rest -- greeting, vehicle, address, no phone
    # number (see cfg.dealer_greeting/cfg.dealer_address).
    _cfg = _get_dealer_config()
    lines.append(_cfg.dealer_greeting)
    lines.append(headline)
    lines.append(_cfg.dealer_address)
    lines.append("")

    resolved_price = resolve_display_price(v)
    if resolved_price:
        try:
            price_num = int(float(str(resolved_price).replace(",", "").replace("$", "")))
            lines.append(f"Price: ${price_num:,}")
        except ValueError:
            lines.append(f"Price: {resolved_price}")
    else:
        # Matches the dealer site's own "Call" placeholder for unpriced
        # (usually incoming/unstocked) vehicles -- better than a bare $0
        # or silently dropping the line from a post meant to be posted as-is.
        lines.append("Price: Call for Price")
    original_msrp = resolve_original_msrp_comparison(v)
    if original_msrp:
        lines.append(f"Original MSRP: {original_msrp}")
    if v.mileage is not None:
        lines.append(f"Mileage: {v.mileage:,} mi")
    if v.vin:
        lines.append(f"VIN: {v.vin}")
    if v.stock_number:
        lines.append(f"Stock #: {v.stock_number}")
    lines.append("")

    specs_lines = []
    if v.exterior_color_factory:
        specs_lines.append(f"Exterior: {v.exterior_color_factory}")
    if v.interior_color:
        specs_lines.append(f"Interior: {v.interior_color}")
    if v.engine:
        specs_lines.append(f"Engine: {v.engine}")
    if v.transmission:
        specs_lines.append(f"Transmission: {v.transmission}")
    if v.drivetrain:
        specs_lines.append(f"Drivetrain: {v.drivetrain}")
    if v.mpg_city or v.mpg_highway:
        mpg = "/".join(filter(None, [v.mpg_city and f"{v.mpg_city} city", v.mpg_highway and f"{v.mpg_highway} hwy"]))
        specs_lines.append(f"MPG: {mpg}")
    elif v.ev_mpge_combined:
        specs_lines.append(f"MPGe: {v.ev_mpge_combined} combined")
    if v.ev_battery_range:
        specs_lines.append(f"EV Range: {v.ev_battery_range} mi")
    if v.cab_style:
        specs_lines.append(f"Cab: {v.cab_style}")
    if v.box_length:
        specs_lines.append(f"Bed length: {v.box_length}")
    if specs_lines:
        lines.extend(specs_lines)
        lines.append("")

    is_new = (v.condition or "").strip().lower() == "new"
    if v.dealer_description and not (is_new and _looks_like_incentive_disclosure(v.dealer_description)):
        lines.append(v.dealer_description.strip())
        lines.append("")

    sticker_equipment = (v.sticker or {}).get("equipment") or {}
    if sticker_equipment:
        for key, label in (
            ("exterior", "Exterior"),
            ("interior", "Interior"),
            ("functional_tech", "Functional & Tech"),
            ("safety_security", "Safety & Security"),
        ):
            feats = new_features(sticker_equipment.get(key))
            if not feats:
                continue
            lines.append(f"{label}:")
            for feat in feats:
                lines.append(f"- {feat}")
            lines.append("")
        optional = new_features(v.sticker.get("optional_equipment"))
        if optional:
            lines.append("Optional Equipment:")
            for feat in optional:
                lines.append(f"- {feat}")
            lines.append("")
        # Ford's factory warranty (e.g. ~3yr/36k basic, ~5yr/60k
        # powertrain) transfers to a used/CPO buyer as long as the vehicle
        # is still within those mileage/time limits -- it isn't voided by
        # resale, so this isn't new-only. Gated on make=="ford" since the
        # sticker's warranty terms are Ford-specific and this dealer's used
        # inventory includes non-Ford trade-ins the sticker data wouldn't
        # apply to.
        warranties = v.sticker.get("warranties")
        if warranties and (v.make or "").strip().lower() == "ford":
            lines.append("Factory Warranties:")
            for w in warranties:
                lines.append(f"- {w}")
            lines.append("")
    elif v.main_features:
        feats = new_features(v.main_features)
        if feats:
            lines.append("Features:")
            lines.extend(f"- {f}" for f in feats)
            lines.append("")
    elif v.features_structured:
        feats = [f for _, group in v.features_structured.items() for f in new_features(group)]
        if feats:
            lines.append("Features:")
            lines.extend(f"- {f}" for f in feats)
            lines.append("")

    if v.carfax_one_owner:
        lines.append("CARFAX 1-Owner")
        lines.append("")

    more_details = resolve_more_details_link(v)
    if more_details:
        lines.append(f"More details: {more_details}")
    # No dealership-URL fallback here on purpose -- the post links out to
    # the manufacturer's own site when available, never back to
    # tomballford.com itself.

    return "\n".join(lines).strip() + "\n"


def explain_facebook_post(v: Vehicle) -> dict:
    """Small audit trail of which condition-dependent branch fired for
    this vehicle's post -- saved into details.json, never shown to the
    customer. With this much condition-dependent branching now (new vs.
    used/CPO pricing, warranty eligibility, incentive-language suppression,
    manufacturer-specific links), "why does this one look different"
    should be a five-second read of details.json, not a re-derivation of
    the logic by eye. Mirrors build_facebook_post()'s own branching rather
    than folding into its return value, so post-building doesn't have to
    carry bookkeeping alongside the text it emits."""
    is_new = (v.condition or "").strip().lower() == "new"
    sticker_msrp = (v.sticker or {}).get("pricing", {}).get("total_msrp")

    if not is_new:
        price_source = "site_display_price"
    elif sticker_msrp:
        price_source = "sticker_total_msrp"
    elif any(r.get("label", "").strip().upper() == "MSRP" for r in v.pricing_rows):
        price_source = "site_pricing_rows_msrp"
    else:
        price_source = "site_display_price_fallback"

    return {
        "condition_bucket": "new" if is_new else "used_or_cpo",
        "price_source": price_source,
        "factory_warranties_shown": bool(
            (v.sticker or {}).get("warranties") and (v.make or "").strip().lower() == "ford"
        ),
        "original_msrp_comparison_shown": resolve_original_msrp_comparison(v) is not None,
        "incentive_description_suppressed": bool(
            is_new and v.dealer_description and _looks_like_incentive_disclosure(v.dealer_description)
        ),
        "more_details_link": resolve_more_details_link(v),
        "pricing_consistency_issues": check_pricing_consistency(v),
    }