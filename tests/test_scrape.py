"""Tests for the pluggable CMS extractor registry."""


def test_dealerinspire_extractor_registered() -> None:
    """The DealerInspire extractor is registered by default."""
    from scrape import _EXTRACTORS
    names = [name for name, _, _ in _EXTRACTORS]
    assert "dealerinspire" in names


def test_extract_balanced_json_valid() -> None:
    """A simple balanced JSON object is extracted correctly."""
    from scrape import extract_balanced_json
    html = '<script>var data = {"key": "value"}</script>'
    marker = 'var data = '
    result = extract_balanced_json(html, marker)
    assert result == {"key": "value"}


def test_extract_balanced_json_nested() -> None:
    """Nested objects with braces inside strings work."""
    from scrape import extract_balanced_json
    html = 'var x = {"outer": {"inner": "text with {brace}"}}'
    result = extract_balanced_json(html, 'var x = ')
    assert result == {"outer": {"inner": "text with {brace}"}}


def test_extract_balanced_json_no_match() -> None:
    """No match returns None."""
    from scrape import extract_balanced_json
    assert extract_balanced_json("<html></html>", "nothing") is None


def test_extract_analytics_object_returns_cms_name() -> None:
    """extract_analytics_object returns (data, cms_name)."""
    from scrape import extract_analytics_object, DEALERINSPIRE_VAR_MARKER
    # Craft a fake blob that matches DealerInspire"s validator
    payload = '{"vdp_gtm_payload": {"vin": "1HGCY1F24SA123456"}}'
    html = "<script>" + DEALERINSPIRE_VAR_MARKER + payload + "</script>"
    data, cms = extract_analytics_object(html)
    assert data is not None
    assert cms == "dealerinspire"
    assert data["vdp_gtm_payload"]["vin"] == "1HGCY1F24SA123456"


def test_extract_analytics_object_none() -> None:
    """No matching extractor returns (None, None)."""
    from scrape import extract_analytics_object
    data, cms = extract_analytics_object("<html>no data blob here</html>")
    assert data is None
    assert cms is None


def test_register_custom_extractor() -> None:
    """A custom extractor can be registered and found."""
    from scrape import extract_analytics_object, register_extractor, _EXTRACTORS
    original_len = len(_EXTRACTORS)
    register_extractor("test_cms", "CUSTOM_DATA = ", lambda d: bool(d and "vehicle" in d))
    try:
        payload = '{"vehicle": {"vin": "TEST"}}'
        html = "<script>CUSTOM_DATA = " + payload + "</script>"
        data, cms = extract_analytics_object(html)
        assert data is not None
        assert cms == "test_cms"
        assert data["vehicle"]["vin"] == "TEST"
    finally:
        while len(_EXTRACTORS) > original_len:
            _EXTRACTORS.pop()
