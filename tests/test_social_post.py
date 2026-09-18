"""Tests for social media post generation."""

from lotstretcher.scrape import Vehicle
from lotstretcher.dealer_config import reload


def _make_vehicle(**overrides) -> Vehicle:
    base = dict(
        url="https://example.com/vehicle/ABC123/",
        vin="1HGCY1F24SA123456",
        year="2025",
        make="Honda",
        model="Accord",
        trim="LX",
        condition="Used",
        mileage=15000,
        display_price="$25,000",
        exterior_color_factory="Gray Metallic",
        interior_color="Gray",
        engine="1.5T I4",
        transmission="CVT",
        drivetrain="FWD",
        fuel_type="Gasoline",
        body_type="Sedan",
    )
    base.update(overrides)
    return Vehicle(**base)


def test_threads_limit() -> None:
    """Threads post is under 500 characters."""
    from lotstretcher.social_post import build_threads_post

    reload()
    post = build_threads_post(_make_vehicle())
    assert len(post) <= 500


def test_instagram_caption() -> None:
    """Instagram caption is non-empty."""
    from lotstretcher.social_post import build_instagram_caption

    reload()
    caption = build_instagram_caption(_make_vehicle())
    assert len(caption) > 0


def test_tags_in_instagram() -> None:
    """Instagram caption includes hashtags from city_tags."""
    from lotstretcher.social_post import build_instagram_caption

    reload()
    caption = build_instagram_caption(_make_vehicle())
    assert "#" in caption


def test_threads_has_price() -> None:
    """Threads post contains the vehicle price."""
    from lotstretcher.social_post import build_threads_post

    reload()
    post = build_threads_post(_make_vehicle(display_price="$25,000"))
    assert "$25,000" in post
