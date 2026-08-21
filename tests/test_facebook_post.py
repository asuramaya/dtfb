"""Tests for Facebook post generation (config-aware version)."""

import pytest

from dealer_config import reload
from scrape import Vehicle


def _make_vehicle(**overrides) -> Vehicle:
    """Build a Vehicle with defaults and overrides."""
    base = dict(
        url="https://example.com/vehicle/ABC123/",
        vin="1HGCY1F24SA123456",
        year="2025",
        make="Honda",
        model="Accord",
        trim="LX",
        title="2025 Honda Accord LX",
        condition="Used",
        mileage=15000,
        display_price="$25,000",
        exterior_color_factory="Gray Metallic",
        interior_color="Gray",
        engine="1.5T I4",
        transmission="CVT",
        drivetrain="FWD",
        fuel_type="Gasoline",
        mpg_city="29",
        mpg_highway="37",
        body_type="Sedan",
    )
    base.update(overrides)
    return Vehicle(**base)


def test_post_includes_greeting() -> None:
    """The dealer greeting is in the first line of every post."""
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle())
    lines = post.strip().split("\n")
    assert "Hector Chavez" in lines[0]


def test_post_includes_address() -> None:
    """The dealer address is present."""
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle())
    assert "22702 TX-249" in post


def test_post_headline_new() -> None:
    """New vehicles get 'New' prepended to the title."""
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle(condition="New"))
    lines = post.strip().split("\n")
    assert "New 2025 Honda Accord LX" in lines[1]


def test_post_headline_used() -> None:
    """Used vehicles do NOT get 'New' prepended."""
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle(condition="Used"))
    lines = post.strip().split("\n")
    assert "New" not in lines[1]


def test_post_includes_mileage() -> None:
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle(mileage=15000))
    assert "15,000" in post


def test_post_includes_vin() -> None:
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle(vin="1HGCY1F24SA123456"))
    assert "1HGCY1F24SA123456" in post


def test_post_price_call() -> None:
    """Unpriced vehicles show 'Call for Price'."""
    from facebook_post import build_facebook_post

    reload()
    post = build_facebook_post(_make_vehicle(display_price=None))
    assert "Call for Price" in post


def test_resolve_display_price_new_msrp() -> None:
    """New vehicles prefer MSRP over display_price."""
    from facebook_post import resolve_display_price

    v = _make_vehicle(condition="New", display_price="$50,495",
                       pricing_rows=[{"label": "MSRP", "text": "$54,770"}])
    assert resolve_display_price(v) == "$54,770"


def test_resolve_display_price_used() -> None:
    """Used vehicles just show display_price."""
    from facebook_post import resolve_display_price

    v = _make_vehicle(condition="Used", display_price="$25,000")
    assert resolve_display_price(v) == "$25,000"


def test_resolve_original_msrp() -> None:
    """Used vehicles with a sticker show original MSRP comparison."""
    from facebook_post import resolve_original_msrp_comparison

    v = _make_vehicle(condition="Used", sticker={"pricing": {"total_msrp": "$35,000"}},
                       display_price="$25,000")
    msrp = resolve_original_msrp_comparison(v)
    assert msrp == "$35,000"


def test_resolve_original_msrp_new() -> None:
    """New vehicles don't show original MSRP (it's the same as current)."""
    from facebook_post import resolve_original_msrp_comparison

    v = _make_vehicle(condition="New", sticker={"pricing": {"total_msrp": "$35,000"}},
                       display_price="$35,000")
    assert resolve_original_msrp_comparison(v) is None


def test_manufacturer_more_details_ford() -> None:
    """Ford vehicles with a sticker get a QR link."""
    from facebook_post import resolve_more_details_link

    v = _make_vehicle(make="Ford", sticker={"pricing": {"total_msrp": "50000"}})
    link = resolve_more_details_link(v)
    assert link is not None
    assert "v.ford.com" in link


def test_manufacturer_more_details_honda() -> None:
    """Honda vehicles get no link (no resolver registered)."""
    from facebook_post import resolve_more_details_link

    v = _make_vehicle(make="Honda", sticker={"pricing": {"total_msrp": "50000"}})
    assert resolve_more_details_link(v) is None
