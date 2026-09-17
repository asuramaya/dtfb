# dtfb · dealer-to-facebook

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Interface: CLI](https://img.shields.io/badge/interface-CLI-informational.svg)](#cli-command-reference)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen.svg)](tests/)

**dtfb** ("Dealer to Facebook") automates the workflow of turning a vehicle listing from a dealership website
into polished, platform-specific social-media posts and high-converting marketing visuals.

Given a vehicle detail page (VDP) URL, it:

1. **Scrapes** vehicle specs, full-resolution photo galleries, window stickers, and Carfax history via Playwright (bypassing Cloudflare challenges).
2. **Processes** photos through a computer vision pipeline — zero-shot CLIP classification, rembg (BiRefNet) background removal, perceptual-hash deduplication (including a cross-vehicle photo cache that reuses an already-processed cutout when the dealer's CDN serves the same manufacturer stock photo to multiple listings), dealer banner cropping, and super-resolution upscaling.
3. **Composes** hero collages, solo framed images with branded borders, and animated MP4 video carousels — hero-video rendering runs across parallel workers with GPU-accelerated compositing, and overlaps with the next vehicle's scrape/CV work during a batch sync.
4. **Generates** ready-to-copy-paste posts for Facebook Marketplace, Instagram, and Threads — each tailored to platform character limits, hashtag strategies, and preview rules.
5. **Tracks** processed inventory via a persistent manifest, making daily syncs fast and incremental.

dtfb was originally built for [Tomball Ford](https://www.tomballford.com) (a DealerInspire CMS site) and is designed to work plug-and-play with any DealerInspire-powered dealership, and with any other dealership CMS via a pluggable extractor architecture.

---

## Why dtfb?

Vehicle photo/video merchandising software is a real market with real incumbents (CarCutter, Impel/SpinCar,
Spyne, PBS Systems, and others), and most of it is built and priced for multi-rooftop dealer *groups* —
contract terms, sales-call-gated pricing, brand-compliance dashboards across locations. That's a different
customer than the single-rooftop independent dealer this project was actually built for.

The closer comparison is the narrow pure-play tier — standalone AI background-removal/compositing tools —
where pricing is public: as of this research (2026-09), one such vendor lists **$0.20–$0.40 per photo, no
subscription**. dtfb does the same core job (photo classification, background removal, hero/video
compositing, post copy) self-hosted, **for the marginal cost of your own compute after setup** — no
per-photo fee, no monthly subscription, no contract, and your dealership's photos never leave your own
server. It's free and open source (MIT), not a crippled trial of a paid product.

That's the tradeoff, stated plainly: you run it yourself (a GPU with 4+ GB VRAM recommended, see below), and
in exchange you own the pipeline outright.

dtfb runs in **two modes on the same pipeline**: the CLI (the default, one-shot scripts and cron) and an
opt-in **server mode** exposing a [CarCutter](https://cloud.car-cutter.com/doc/api.html)-API-shaped HTTP
surface — see [Server Mode](#server-mode) below. The point isn't cloning CarCutter; it's that a lot of
dealer-software integrations already speak that shape, so switching the base URL to a self-hosted, fully
open alternative is a realistic option rather than a rewrite. Nothing about your inventory photos leaves
your own server either way.

---

## Hardware & System Requirements

### Hardware & GPU Acceleration
- **GPU (Recommended)**: NVIDIA GPU with **4+ GB VRAM** (CUDA support). `inventory-sync`'s batch pipeline (parallel hero-video rendering across CPU workers, GPU-accelerated compositing, overlapped scrape/CV work across vehicles) measures **~35–50 seconds of effective wall-clock time per vehicle** end-to-end on real inventory, not counting network waits on the dealer site.
- **CPU (Fallback)**: CPU-only execution is fully supported via PyTorch and ONNX Runtime CPU fallbacks, but processing time will be several minutes per vehicle due to deep-learning models (CLIP, BiRefNet, SwinIR/Real-ESRGAN, SAM2) and CPU-side video compositing.

> **Dependency & CUDA Note**:
> This package requires `onnxruntime-gpu` (not the CPU-only `onnxruntime` package) for `rembg`'s background-removal model to actually run on the GPU — the CPU package silently falls back to ~10x slower CPU inference with no error. It also installs `torch`, `torchvision`, `open_clip_torch`, `rembg`, and `spandrel`. Ensure your NVIDIA drivers and CUDA runtime are compatible with your installed PyTorch wheel. If running on a system without a GPU, ONNX Runtime and PyTorch will automatically execute on the CPU.

### System Dependencies
- **Python 3.11+**
- **FFmpeg**: Required for generating animated MP4 hero videos (`hero-video`).
  - *Ubuntu/Debian*: `sudo apt update && sudo apt install -y ffmpeg`
  - *macOS*: `brew install ffmpeg`
  - *Arch Linux*: `sudo pacman -S ffmpeg`

---

## Installation

Because `dtfb` is distributed as an open-source source repository, clone the repository and install it inside a Python virtual environment:

```bash
# 1. Clone the repository
git clone https://github.com/asuramaya/dtfb.git
cd dtfb

# 2. Create and activate a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dtfb in editable mode
pip install --upgrade pip
pip install -e .

# Optional: Install development and test dependencies
pip install -e ".[dev]"

# 4. Install Playwright browser binaries (Chromium)
playwright install chromium
```

---

## Quick Start

Process a single vehicle listing:

```bash
dtfb "https://www.tomballford.com/vehicle/1HGCY1F24SA035661/Used-2025-Honda-Accord-Tomball-TX/"
```

Output lands in the current working directory under `new/` or `used/`, bucketed by condition. Each vehicle folder contains structured data, source assets, and finished marketing deliverables:

```
<year>-<make>-<model>-<trim>-<stock>/
├── details.json                  # Full scraped Vehicle record
├── images/
│   ├── exterior/                 # Originals + CLIP-classified exterior photos
│   │   ├── cutout/               # Transparent background cutouts
│   │   └── wheels/               # Extracted wheel close-ups
│   └── interior/                 # White-balance-corrected interior photos
├── window-sticker.pdf            # Original Monroney window sticker (if found)
├── window-sticker.json           # Parsed window sticker options & equipment
└── bundle/
    ├── hero.png                  # Composed hero collage (1:1 square for Facebook Marketplace)
    ├── hero-portrait.png         # 4:5 portrait (Instagram & Facebook feed)
    ├── hero-video.mp4            # Animated video carousel (1:1 square)
    ├── hero-video-vertical.mp4   # 9:16 vertical video (Instagram Reels / TikTok / Shorts)
    ├── hero-video-horizontal.mp4 # 16:9 widescreen video (YouTube)
    ├── framed/                   # Every exterior cutout framed individually
    ├── window-sticker-a.png      # Readable window sticker slide A
    ├── window-sticker-b.png      # Readable window sticker slide B
    ├── facebook.txt              # Ready-to-paste Facebook Marketplace listing text
    ├── instagram.txt             # Hook-first caption + targeted hashtags
    └── threads.txt               # Character-capped Threads post
```

---

## Configuration

dtfb uses a flexible configuration hierarchy:
**CLI Arguments > Environment Variables (`DTFB_*`) > JSON Config File (`--dealer-config` / `DTFB_CONFIG`) > Default Values**

### 1. JSON Configuration File

Create a `dealer-config.json` file for your dealership:

```json
{
  "dealer_name": "Apex Ford of Austin",
  "dealer_greeting": "Ask for Alex in Sales!",
  "dealer_address": "4500 Motorway Blvd, Austin, TX 78701",
  "city_tags": ["Austin", "AustinCars", "ATXAuto", "TexasTrucks"],
  "default_border_tag": "dealer-frame",
  "inventory_url": "https://www.apexfordaustin.com/inventory/all-vehicles/",
  "inventory_urls": {
    "used": "https://www.apexfordaustin.com/inventory/used-vehicles/",
    "new": "https://www.apexfordaustin.com/inventory/new-vehicles/",
    "all": "https://www.apexfordaustin.com/inventory/all-vehicles/"
  }
}
```

`inventory_urls` is what `inventory-sync --scope <name>` reads, so you can run `inventory-sync --scope used` instead of retyping your dealer's full URL every time. Scope names are entirely up to you -- they're just keys in this object, not a fixed set dtfb understands. There's deliberately no built-in default for these (unlike `dealer_name`/`dealer_greeting`/etc): an unconfigured `--scope` fails loudly rather than silently pointing at whichever dealership this tool happened to ship with example values for.

Pass it on any command with `--dealer-config` or by setting the `DTFB_CONFIG` environment variable:

```bash
dtfb <vdp-url> --dealer-config /path/to/dealer-config.json
```

### 2. Environment Variables

You can also configure dtfb directly using environment variables (ideal for Docker or CI/CD pipelines):

| Environment Variable | Description | Default |
|---|---|---|
| `DTFB_CONFIG` | Path to a JSON configuration file | `None` |
| `DTFB_DEALER_NAME` | Dealership name used in copy & captions | `Tomball Ford` |
| `DTFB_DEALER_GREETING` | Greeting line in Facebook / social posts | `Ask for us at the front desk!` |
| `DTFB_DEALER_ADDRESS` | Physical address included in listing copy | `22702 TX-249, Tomball, TX 77375` |
| `DTFB_CITY_TAGS` | Comma-separated hashtags for social copy | `Tomball,TomballCars,Houston,HoustonCars` |
| `DTFB_DEFAULT_BORDER_TAG`| Default border tag from `assets/manifest.json` | `tomball-dealer-frame` |
| `DTFB_INVENTORY_URL` | Full inventory search URL for batch crawling | `None` |
| `DTFB_INVENTORY_URL_<SCOPE>` | Per-scope inventory URL, e.g. `DTFB_INVENTORY_URL_USED` for `--scope used` | `None` |
| `DTFB_LISTINGS_ROOT` | Default output directory for listings | `./listings` or current directory |

See `.env.example` and `dtfb-config.json.example` for template files.

---

## Branded Assets & Borders

dtfb composites vehicle cutouts onto branded border frames. These assets live in `assets/` and are cataloged in `assets/manifest.json`:

- `assets/borders/`: Frame PNGs with transparent center windows (standard size: **1254×1254**). The frame sits on top of the cutouts so logos and phone numbers stay sharp.
- `assets/backgrounds/`: Background textures and graphics (used when not using the per-vehicle gradient generator).
- `assets/video/` & `assets/audio/`: Looping video backdrops and background tracks for animated hero videos.

### Adding Your Own Dealership Frame
1. Design a **1254×1254 PNG** with a transparent center where the vehicle should appear.
2. Save it to `assets/borders/my-dealer-frame.png`.
3. Register it in `assets/manifest.json`:
   ```json
   {
     "name": "My Dealership Frame",
     "file": "borders/my-dealer-frame.png",
     "tags": ["dealer-frame", "custom"]
   }
   ```
4. Use `--border "My Dealership Frame"` or set `"default_border_tag": "custom"` in your configuration.

---

## CLI Reference & Usage

Installing `dtfb` registers 8 CLI commands (a 9th, `dtfb-serve`, is opt-in -- see Server Mode below):

### 1. `dtfb` — Full Ingestion Pipeline
Scrapes the VDP, runs CV segmentation, composes images/videos, and generates copy.

```bash
# Single vehicle
dtfb https://www.yourdealer.com/vehicle/12345/Used-2023-Ford-F-150/

# Batch from a URL list file
dtfb --file urls.txt --out ~/Documents/listings

# Crawl an inventory listing page (expands all matching VDPs)
dtfb "https://www.yourdealer.com/inventory/all-vehicles/?make=Ford&model=Mustang" --out ~/Documents/listings

# Dry-run: preview what a listing URL would expand to without scraping
dtfb "https://www.yourdealer.com/inventory/all-vehicles/" --dry-run

# Run full inventory sync (processes new vehicles and flags delisted ones)
dtfb "https://www.yourdealer.com/inventory/all-vehicles/" --out ~/Documents/listings --sync
```

### 2. `inventory-sync` — Automated Inventory Synchronizer
A dedicated CLI designed for cron jobs. It crawls live inventory, skips already-processed vehicles via `manifest.json`, processes new arrivals, and marks delisted inventory.

```bash
inventory-sync --inventory-url "https://www.yourdealer.com/inventory/all-vehicles/" --out ~/Documents/listings

# Also run the local-vision front-seat-config check (needs `ollama serve` with
# gemma4:e2b pulled) -- batched at the end of the run so the model loads once,
# not once per vehicle
inventory-sync --inventory-url "..." --out ~/Documents/listings --vision-seat-check

# Once your dealer-config.json has an "inventory_urls" map (see Configuration
# above), skip retyping the URL entirely:
inventory-sync --scope used
inventory-sync --scope new
```

**Safety valves**: a sync refuses to flag anything as delisted if the listing crawl came back incomplete (`--headed` to debug why), or if more than 30% of previously-seen active inventory would suddenly be flagged missing in one cycle (almost always a bad/narrow crawl, not real turnover). Pass `--confirm-mass-delist` to override either one if you're sure it's genuine.

**"Just Arrived, Photos Coming Soon" placeholders**: a vehicle whose gallery is nothing but the dealer's stock "photos coming soon" graphic is recognized as junk (perceptual-hash matched, see `imaging/templates/`) and left with zero photos rather than a fake cutout of the placeholder. It's marked `photos_pending` in the manifest instead of "done," so the next sync automatically retries it — no manual re-run needed once real photos go up.

### 3. `posts` — Regenerate Post Copy
Fast copy re-generation from existing `details.json` files without re-scraping or re-running computer vision models.

```bash
# Rebuild copy for all listings in a directory
posts ~/Documents/listings

# Rebuild copy for a single vehicle folder
posts ~/Documents/listings/used/2023-Ford-F-150-Raptor-PFA30435/
```

### 4. `recompose` — Recompose Images with New Borders / Layouts
Re-renders `bundle/hero.png` and `bundle/framed/*.png` using already-extracted cutouts. Ideal when you update your dealership logo or frame.

```bash
recompose ~/Documents/listings --border "Generic Dealer Frame"
```

`--resweep` self-heals photos that were misfiled into `images/interior/` under an older classifier or
threshold: it re-runs *current* classification against every interior photo already on disk and promotes
any that the pipeline would now call exterior — without re-scraping. It's conservative by design (a
promotion only happens if the photo also produces a valid cutout AND survives the same gallery-consistency
check `--prune-foreign` runs in reverse), so most candidates get correctly left alone; use `--dry-run`
first to see what it would do.

```bash
recompose ~/Documents/listings --resweep --interiors --dry-run   # preview
recompose ~/Documents/listings --resweep --interiors             # apply
# then, for any vehicle it reports as changed:
hero-video ~/Documents/listings/used/2023-Ford-.../ --format all
```

### 5. `compose` — Custom Single-Vehicle Hero Composer
Compose custom hero layouts for a single vehicle with specific layouts (`quad`, `corners`, `trio`, `split`), backgrounds, and lighting effects.

```bash
# List available backgrounds, borders, and layouts
compose --list-assets

# Compose with a specific layout and background
compose ~/Documents/listings/used/2023-Ford-F-150-Raptor-PFA30435/ --layout quad --background american-flag
```

### 6. `hero-video` — Animated Video Carousel Generator
Generates animated MP4 video carousels with multi-angle cutouts timed to background music and video loops.

```bash
hero-video ~/Documents/listings/used/2023-Ford-F-150-Raptor-PFA30435/
```

### 7. `export-feed` — Inventory Syndication Feed
Builds a CSV/TSV inventory feed from vehicles already scraped, in the shape third-party listing platforms
(Meta/Facebook Automotive Inventory Ads, and by convergent convention most others) expect to ingest — VIN,
price, mileage, colors, the dealer's own real photo URLs, one row per vehicle. There's no open, universal
inventory feed standard (every DMS integration is a bespoke, gatekept vendor relationship), but *exporting*
a feed needs nobody's permission — this just re-shapes data dtfb already has.

```bash
export-feed ~/Documents/listings --out feed.csv
export-feed ~/Documents/listings/used --out used-feed.csv --condition used --tsv
```

Read `src/dtfb/export_feed.py`'s module docstring before relying on this in production: the exact field
names were cross-confirmed from several independent secondary sources rather than pulled directly from
Meta's own (JS-rendered, partially login-gated) spec page — validate the output against your destination
platform's own feed validator, and adjust `FEED_COLUMNS`/`vehicle_to_row()` if reality differs; it's
deliberately kept as one flat mapping table, not scattered logic.

### 8. `spin-video` — Rotating "Spin" Video from Real Angles
Turns a vehicle's own real angle-labeled cutouts (front → front_3q → side → rear_3q → rear) into a rotating
video via cross-dissolve interpolation between them — the used-inventory-appropriate alternative to a
licensed generic 3D CAD model (right for new inventory, wrong for used: it shows the SKU, not the actual
physical car). This is **not** true 3D reconstruction — that's confirmed research-stage industry-wide, not
something any vendor ships at scale today. It's the same class of trick behind commercial "360 spin"
products: synthesize a smooth rotation from a handful of real photos instead of requiring a turntable shoot.
dtfb typically has 5 real angles per vehicle already — more anchor frames than the reference product's own
4-photo baseline. See `src/dtfb/imaging/compose/spin.py`'s module docstring for the full scope/honesty
notes (one side's half-turn, not a full 360; classical cross-dissolve, not a trained view-synthesis model).

```bash
spin-video ~/Documents/listings/used/2023-Ford-F-150-Raptor-PFA30435/
```

---

## Server Mode

`dtfb-serve` is a 9th command, opt-in and separate from the 8 above — a long-running HTTP service instead
of a one-shot script, exposing dtfb's own classify/cutout/composite pipeline behind a
[CarCutter](https://cloud.car-cutter.com/doc/api.html)-API-shaped surface: same request/response shapes
where it matters for drop-in compatibility, self-hosted, and every line of processing code is readable
(unlike the closed SaaS API it mirrors). It's a different front door onto the same imaging pipeline the CLI
uses, not a parallel implementation.

```bash
# Install the server extra (kept separate so `pip install -e .` for CLI-only use never pulls in a web framework)
pip install -e ".[server]"

dtfb-serve --data-dir ~/.dtfb-server --port 8000
```

Core endpoints implemented (see `src/dtfb/server/app.py`'s module docstring for the full scope, including
what's accepted-for-compatibility but not yet behavioral, e.g. `cut_type="blur"`):

- `POST /vehicle/composition/single-segment` — sync, one image in, one composited result out.
- `POST /vehicle/image/submission` — async, up to 60 images, optional `webhook_url` callback.
- `GET /vehicle/image/status` / `GET /vehicle/image/result` — poll a submission.
- `POST /vehicle/submission`, `POST /vehicle/list`, `GET /vehicle/status`, `DELETE /vehicle/delete`,
  `GET /vehicle/shotlist` — a light vehicle registry (SQLite-backed), bookkeeping only.

```bash
curl -X POST http://127.0.0.1:8000/vehicle/composition/single-segment \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/car.jpg", "cut_type": "complete"}'
```

---

## Daily Automation (Cron)

The included `daily_sync.sh` wraps `inventory-sync` for unattended cron use: starts Ollama if configured
(for `--vision-seat-check`), writes a timestamped log per run, and prunes logs older than 30 days.
Configure it entirely via environment variables (defaults shown, all optional):

```bash
export DTFB_REPO_DIR="/path/to/dtfb"                      # default: this deployment's checkout
export DTFB_SCOPE="used"                                  # a name from dealer-config.json's inventory_urls
# ...or set DTFB_INVENTORY_URL directly to skip scope resolution entirely
export DTFB_LISTINGS_ROOT="/path/to/listings"
export DTFB_CONFIG="/path/to/dealer-config.json"           # dealer_name/greeting/inventory_urls/etc
export DTFB_LOG_DIR="/path/to/sync-logs"                   # default: ~/.local/sync-logs
```

Add a cron job (`crontab -e`) to run daily at 6:00 AM:

```cron
0 6 * * * /path/to/dtfb/daily_sync.sh >> /var/log/dtfb-sync.log 2>&1
```

(`daily_sync.sh` writes its own per-run log under `DTFB_LOG_DIR` regardless -- the redirect above just
catches anything printed before that log file exists, e.g. a missing `.venv`.)

---

## Adding Support for Other Dealership CMS Platforms

`dtfb` comes out-of-the-box with support for **DealerInspire** CMS platforms. Adding support for another CMS (e.g. Dealer.com, DealerOn, CDK Global) is simple thanks to the pluggable extractor registry in `scrape.py`.

### How CMS Extraction Works
Most automotive CMS platforms embed vehicle data directly into a JavaScript variable on the page for Google Tag Manager / analytics. `dtfb` searches the rendered HTML for this marker and extracts the balanced JSON object.

### Example: Adding a Custom CMS Extractor

```python
# in scrape.py or an extension script:
from dtfb.scrape import register_extractor, normalize_vehicle, Vehicle

# 1. Define the JavaScript marker and validation function
CUSTOM_CMS_MARKER = "window.digitalData = "

def custom_cms_validator(data: dict) -> bool:
    # Verify that the parsed JSON contains expected vehicle keys
    return bool(data and "vehicle" in data and "vin" in data["vehicle"])

# 2. Register the extractor
register_extractor("custom_cms", CUSTOM_CMS_MARKER, custom_cms_validator)

# 3. Update normalize_vehicle() to map your CMS fields to the Vehicle dataclass
# (e.g. mapping data['vehicle']['vin'] -> Vehicle.vin)
```

---

## Troubleshooting

### 1. Cloudflare Challenge Timeouts / "Just a moment..."
- **Symptom**: `RuntimeError: Cloudflare challenge did not clear in time`.
- **Cause**: The dealership website is presenting an interactive Cloudflare turnstile or rate-limiting requests.
- **Solution**:
  - `dtfb` automatically retries with exponential backoff and passes realistic browser headers.
  - Avoid running dozens of parallel threads against the same domain simultaneously.
  - Test the URL in standard non-headless Chromium to verify your IP is not banned.

### 2. Missing Playwright Browser Binaries
- **Symptom**: `playwright._impl._errors.Error: Executable doesn't exist at ...`
- **Solution**: Run `playwright install chromium` inside your virtual environment. If running on headless Linux, also install OS dependencies with `playwright install-deps chromium`.

### 3. PyTorch / CUDA Out Of Memory (OOM)
- **Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory`, or (specifically for hero videos) `h264_nvenc`'s `CreateInputBuffer failed: out of memory`.
- **Cause**: `dtfb` clears GPU cache between vehicles for the classification/cutout models, but `inventory-sync`'s batch pipeline deliberately overlaps a vehicle's hero-video rendering (parallel worker processes, each doing GPU-accelerated compositing plus an `h264_nvenc` encode) with the *next* vehicle's CLIP/rembg work on the main process — real concurrent GPU pressure, not a leak. On a smaller card (the reference numbers above assume 8+ GB) several concurrent encode sessions plus the classification models in memory at once can genuinely exceed what's free.
- **Solution**:
  - Hero-video rendering already self-heals for this specific case: if `h264_nvenc` fails to open, `render_hero_video()` automatically retries the same video with the software `libx264` encoder rather than losing it — you'll see `[fell back to libx264, GPU was too busy for h264_nvenc]` in the sync log. No video is lost, it's just slower under load.
  - If OOM shows up elsewhere (classification/cutout, not video encoding), or the fallback itself is triggering constantly and slowing your syncs more than you'd like, lower `VIDEO_WORKERS`/`MAX_INFLIGHT_VEHICLES` in `inventory_sync.py`, or drop `--nvenc`/`video_encoder="h264_nvenc"` entirely to encode on CPU only.
  - If you have limited VRAM (< 4 GB), set `export CUDA_VISIBLE_DEVICES=""` to force CPU execution mode.

### 4. Missing FFmpeg Error
- **Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` when running `hero-video`.
- **Solution**: Install FFmpeg via your system package manager (`sudo apt install ffmpeg` or `brew install ffmpeg`).

### 5. Carfax Link Missing on Used Vehicles
- **Symptom**: `Vehicle.carfax_url` is `None` even though a badge is present on the website.
- **Explanation**: DealerInspire sites populate the Carfax link via an asynchronous client-side API call into the `.carfax-logo` element. `fetch_rendered_html()` waits up to 12 seconds for this element when "used" is in the URL. If the network is exceptionally slow, the link may not have loaded before timeout.

---

## Testing

`dtfb` includes a complete test suite:

```bash
# 1. Run unit tests
pytest tests/

# 2. Run scrape regression tests (against offline HTML fixtures)
python scrape_regression.py

# 3. Run computer vision & imaging calibration regression
python regression.py
```

---

## Project Structure

```
src/dtfb/            — Core package
  ├── cli.py         — Main CLI entry point ('dtfb')
  ├── inventory_sync.py — Cron-friendly inventory crawler and delist detector
  ├── compose_cli.py — Custom hero image layout composer
  ├── hero_video_cli.py — Animated video carousel generator
  ├── posts_cli.py   — Fast social copy regenerator (Marketplace, IG, Threads)
  ├── recompose_cli.py — Recompose bundles with updated borders/assets
  ├── export_feed_cli.py — Inventory syndication feed generator (CSV/TSV)
  ├── spin_video_cli.py — Rotating spin video from real angle cutouts
  ├── server/        — Opt-in CarCutter-API-shaped HTTP server (`dtfb-serve`)
  │   ├── app.py     — FastAPI endpoints (vehicle registry + image processing)
  │   ├── store.py   — SQLite-backed vehicle/submission registry
  │   └── processing.py — Shared fetch->classify->cutout->composite pipeline
  ├── scrape.py      — Playwright scraper & pluggable CMS extraction registry
  ├── listing.py     — Inventory search page crawler (VDP link discovery)
  ├── photos.py      — Photo gallery downloader and batch pipeline
  ├── window_sticker.py — Monroney window sticker PDF downloader and parser
  ├── facebook_post.py — Facebook Marketplace post builder
  ├── social_post.py — Instagram and Threads post builders
  ├── vehicle_pipeline.py — Unified orchestrator for end-to-end vehicle processing
  ├── manifest.py    — Incremental fetch tracking & delist detection
  ├── dealer_config.py — Centralized multi-dealer configuration manager
  └── imaging/       — Computer vision & media pipeline:
      ├── classify.py    — CLIP zero-shot vehicle angle classification
      ├── cutout.py      — rembg (BiRefNet) background removal with alpha gating
      ├── pipeline.py    — Per-photo analysis (exterior/interior/detail scoring)
      ├── gallery.py     — Consensus filtering (removes foreign/mismatched cars)
      ├── letterbox.py   — Automated dealer watermark/banner cropping
      ├── interior.py    — Interior white-balance correction & feature extraction
      ├── wheel.py       — SAM2 / CLIPSeg wheel extraction and enhancement
      ├── select.py      — Hero and accent photo selection algorithms
      ├── dedupe.py      — Perceptual-hash image deduplication
      ├── sticker.py     — Window sticker PDF text and option parser
      ├── upscale.py     — Real-ESRGAN / SwinIR super-resolution upscaler
      ├── seat_vision.py — Seating configuration classifier (YOLO + CLIP)
      ├── palette.py     — Dominant vehicle paint color extractor
      └── compose/       — Composition engines for hero collages and video
assets/              — Branded borders, background images, video/audio loops
scrape_fixtures/     — Checked-in HTML fixtures for regression tests
tests/               — Pytest unit test suite
daily_sync.sh        — Reference daily automation script
scrape_regression.py — Scraper regression test runner
regression.py        — Computer vision & calibration regression test runner
setup.py             — Package configuration & entry points
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and submitting pull requests.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
