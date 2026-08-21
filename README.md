# dtfb · dealer-to-facebook

**dtfb** ("Dealer to Facebook", the acronym is a [double entendre](https://www.urbandictionary.com/define.php?term=DTF))
automates the workflow of turning a vehicle listing from a dealership website
into polished, platform-specific social-media posts and high-converting marketing visuals.

Given a vehicle detail page (VDP) URL, it:

1. **Scrapes** vehicle specs, full-resolution photo galleries, window stickers, and Carfax history via Playwright (bypassing Cloudflare challenges).
2. **Processes** photos through a computer vision pipeline — zero-shot CLIP classification, rembg (BiRefNet) background removal, perceptual-hash deduplication, dealer banner cropping, and super-resolution upscaling.
3. **Composes** hero collages, solo framed images with branded borders, and animated MP4 video carousels.
4. **Generates** ready-to-copy-paste posts for Facebook Marketplace, Instagram, and Threads — each tailored to platform character limits, hashtag strategies, and preview rules.
5. **Tracks** processed inventory via a persistent manifest, making daily syncs fast and incremental.

dtfb was originally built for [Tomball Ford](https://www.tomballford.com) (a DealerInspire CMS site) and is designed to work plug-and-play with any DealerInspire-powered dealership, and with any other dealership CMS via a pluggable extractor architecture.

---

## Hardware & System Requirements

### Hardware & GPU Acceleration
- **GPU (Recommended)**: NVIDIA GPU with **4+ GB VRAM** (CUDA support). On a modern GPU, end-to-end processing takes ~20–30 seconds per vehicle.
- **CPU (Fallback)**: CPU-only execution is fully supported via PyTorch and ONNX Runtime CPU fallbacks, but processing time will be 2–5 minutes per vehicle due to deep-learning models (CLIP, BiRefNet, SwinIR/Real-ESRGAN, SAM2).

> **Dependency & CUDA Note**:
> This package installs heavy computer-vision and ML libraries (`torch`, `torchvision`, `onnxruntime-gpu`, `open_clip_torch`, `rembg`, `spandrel`). Ensure your NVIDIA drivers and CUDA runtime are compatible with your installed PyTorch wheel. If running on a system without a GPU, ONNX Runtime and PyTorch will automatically execute on the CPU.

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
git clone https://github.com/your-org/dtfb.git
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
  "inventory_url": "https://www.apexfordaustin.com/inventory/all-vehicles/"
}
```

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

Installing `dtfb` registers 6 CLI commands:

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
```

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

---

## Daily Automation (Cron)

Use the included `daily_sync.sh` script to run automated daily inventory synchronizations:

```bash
#!/usr/bin/env bash
set -euo pipefail

export DTFB_CONFIG="/path/to/dealer-config.json"
export DTFB_INVENTORY_URL="https://www.yourdealer.com/inventory/all-vehicles/"
export DTFB_LISTINGS_ROOT="/path/to/listings"

cd /path/to/dtfb
source .venv/bin/activate

inventory-sync
```

Add a cron job (`crontab -e`) to run daily at 6:00 AM:

```cron
0 6 * * * /path/to/dtfb/daily_sync.sh >> /var/log/dtfb-sync.log 2>&1
```

---

## Adding Support for Other Dealership CMS Platforms

`dtfb` comes out-of-the-box with support for **DealerInspire** CMS platforms. Adding support for another CMS (e.g. Dealer.com, DealerOn, CDK Global) is simple thanks to the pluggable extractor registry in `scrape.py`.

### How CMS Extraction Works
Most automotive CMS platforms embed vehicle data directly into a JavaScript variable on the page for Google Tag Manager / analytics. `dtfb` searches the rendered HTML for this marker and extracts the balanced JSON object.

### Example: Adding a Custom CMS Extractor

```python
# in scrape.py or an extension script:
from scrape import register_extractor, normalize_vehicle, Vehicle

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
- **Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory`.
- **Solution**:
  - `dtfb` processes photos sequentially and clears GPU cache between vehicle runs.
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
dtfb.py              — CLI entry point, batch processing, and arg parsing
inventory_sync.py    — Cron-friendly inventory crawler and delist detector
compose_cli.py       — Custom hero image layout composer
hero_video_cli.py    — Animated video carousel generator
posts_cli.py         — Fast social copy regenerator (Marketplace, IG, Threads)
recompose_cli.py     — Recompose bundles with updated borders/assets
scrape.py            — Playwright scraper & pluggable CMS extraction registry
listing.py           — Inventory search page crawler (VDP link discovery)
photos.py            — Photo gallery downloader and batch pipeline
window_sticker.py    — Monroney window sticker PDF downloader and parser
facebook_post.py     — Facebook Marketplace post builder
social_post.py       — Instagram and Threads post builders
vehicle_pipeline.py  — Unified orchestrator for end-to-end vehicle processing
manifest.py          — Incremental fetch tracking & delist detection
dealer_config.py     — Centralized multi-dealer configuration manager
imaging/             — Computer vision & media pipeline:
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
tests/               — Pytest test suite
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and submitting pull requests.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
