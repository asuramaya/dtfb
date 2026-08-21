"""Tests for the listing URL expander."""

from dtfb.listing import is_vdp_url, VDP_LINK_RE


def test_is_vdp_url_true() -> None:
    """A /vehicle/<VIN>/... URL is recognised as a VDP."""
    assert is_vdp_url("https://www.tomballford.com/vehicle/1HGCY1F24SA123456/Used-2025-Honda/")


def test_is_vdp_url_false() -> None:
    """An inventory listing URL is not a VDP."""
    assert not is_vdp_url("https://www.tomballford.com/inventory/all-vehicles/")


def test_is_vdp_url_with_params() -> None:
    """Query params don't break VDP detection."""
    assert is_vdp_url("https://www.tomballford.com/vehicle/1HGCY1F24SA123456/?foo=bar")


def test_vdp_link_re() -> None:
    """The regex captures VIN-anchored links."""
    html = '<a href="/vehicle/1HGCY1F24SA123456/Used-2025-Honda/">'
    assert VDP_LINK_RE.search(html) is not None


def test_vdp_link_re_no_match() -> None:
    """Links without the /vehicle/<VIN>/ pattern are not matched."""
    html = '<a href="/inventory/all-vehicles/">'
    assert VDP_LINK_RE.search(html) is None
