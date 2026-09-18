# Contributing to lotstretcher

Thank you for your interest in contributing to **lotstretcher**!

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/lotstretcher.git
   cd lotstretcher
   ```

2. **Set up a virtual environment** (Python 3.11+ required):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   playwright install chromium
   ```

3. **System Dependencies**:
   - Ensure `ffmpeg` is installed on your system if you are testing video generation.

## Project Guidelines

- **CLI + Server, Two Modes, One Pipeline**: `lotstretcher` is a high-performance CLI/automation pipeline (the default), plus an opt-in server mode (`src/lotstretcher/server/`, `pip install -e ".[server]"`) exposing that same pipeline behind a CarCutter-API-shaped HTTP surface for drop-in compatibility with existing integrations — self-hosted, fully open, unlike the closed SaaS API it mirrors. Server mode is a different front door onto the same imaging code, not a parallel implementation, and stays a separate optional dependency so installing lotstretcher to run it as a script never drags in a web framework. Please do not submit PRs adding a web *UI* (dashboards, admin panels) or vector search / RAG layers — that line still holds; an API surface for programmatic integration is a different thing than a UI.
- **Resilient Scraping**: Dealership websites are dynamic and frequently sit behind Cloudflare challenges. Scrapers must fail gracefully without throwing uncaught exceptions on missing optional fields.
- **Dealer-Agnostic Core**: Keep core post generators and composition logic dealer-agnostic via `dealer_config.py`. Never hardcode dealership-specific names, addresses, or phone numbers in library modules.

## Testing

Run all unit tests and regression test suites before submitting a pull request:

```bash
# 1. Run unit tests
pytest tests/

# 2. Run scrape regression tests (against checked-in HTML fixtures)
python scrape_regression.py

# 3. Run imaging regression tests (if GPU/models are available)
python regression.py
```

When adding support for a new vehicle edge case or CMS feature, add a corresponding test under `tests/` or a new fixture under `scrape_fixtures/`.

## Adding a CMS Extractor

If adding support for a new dealership CMS platform (e.g., Dealer.com, CDK Global, DealerOn):
1. Register the extractor in `scrape.py` using `register_extractor(name, marker, validator)`.
2. Ensure `normalize_vehicle()` cleanly maps the CMS payload to the unified `Vehicle` dataclass.
3. Add a test fixture under `scrape_fixtures/` and a test case in `scrape_regression.py` and `tests/test_scrape.py`.

## Pull Request Checklist

- [ ] All unit tests pass (`pytest tests/`).
- [ ] Scrape regression passes (`python scrape_regression.py`).
- [ ] No hardcoded dealer specifics added to core modules.
- [ ] New features or config options are documented in `README.md`.
