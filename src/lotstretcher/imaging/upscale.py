"""
Super-resolution for the final cutout images, via spandrel (no basicsr/
realesrgan pip package -- those depend on an abandoned basicsr release that
fails to build against current setuptools/Python).

Tomball Ford's source photos cap out at 960x720 (confirmed: requesting
larger CDN resize params like 2048x2048 just returns the same native size,
there's nothing bigger to fetch) -- so "higher resolution" has to mean
actual upscaling of what we have, not a bigger download.

Model choice matters a lot here. RealESRGAN_x4plus (the "standard" choice,
what this used originally) is trained with an adversarial/GAN loss, which
optimizes for "looks plausible" over "matches the source" -- on a real test
photo it fabricated the grille badge text into unreadable nonsense while
leaving the rest of the image looking sharp and confident, which is a much
worse failure mode than blur: it's not obviously wrong at a glance. This is
not a "current models aren't good enough yet" bug, it's the perception-
distortion trade-off (Blau & Michaeli 2018) -- upscaling is fundamentally
an ill-posed inverse problem, and any method producing more apparent detail
than the source contains is by definition inventing something. GAN/
diffusion models sit at the sharp-but-sometimes-fabricated end of that
trade-off; models trained WITHOUT the adversarial loss (pure L1/PSNR
objective) sit at the faithful-but-softer end, because confidently
inventing wrong pixels is directly penalized by their training loss.

Default model: SwinIR's real-world PSNR-oriented checkpoint (transformer-
based, trained on BSRGAN-style real degradations without a GAN loss).
Tested side-by-side against RealESRNet_x4plus (the non-GAN ESRGAN sibling)
and plain Lanczos on the same failing grille-badge crop: all three stayed
faithful (unlike the GAN model), but SwinIR recovered visibly more real
detail (sharper mesh pattern, crisper text edges) than RealESRNet, which in
turn beat Lanczos. Costs ~4x more time/memory than RealESRNet per image
(~32s vs ~8s on this GPU) -- pass model_name="realesrnet" for the faster/
lighter option if that trade-off matters more than the extra sharpness.

Weights are fetched once from each model's official release and cached in
imaging/weights/.

WHERE IT ACTUALLY PAYS OFF: measured across the fleet on disk (296 exterior
cutouts, 29 wheel crops) by running the real layout math (compose_placement)
against every composed canvas size (hero stills, hero video, all formats).
The general exterior cutout gallery lands at a median ~1.1x scale in the
composite -- only ~4% of cutouts get scaled past 2x, the range where 4x SR
starts to visibly beat plain Lanczos. Below that, the model's own tiled
pass costs 8-32s per image for a difference that isn't there. That's why
--upscale (lotstretcher.py) stays off by default for that gallery -- there's no
fabricated evidence it helps, and real evidence (this measurement) that it
mostly doesn't.

Wheel money shots are the opposite case, and get run through this
unconditionally (photos.py, not gated by --upscale): they're cropped tight
to a single wheel (median ~520px), then composited solo at hero size, which
scales them ~2.1x on average -- 59% of real wheel crops exceed 2x. That's
exactly the regime this module was built and tested for (the grille-badge
test above). Re-measure this split with `find imaging/upscale.py` in mind
if the photo vendor's crop framing or the hero canvas size ever changes --
both numbers above are empirical, not assumed.
"""
from __future__ import annotations

import io
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODELS = {
    "swinir": {
        "filename": "SwinIR_x4_PSNR.pth",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
               "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
    },
    "realesrnet": {
        "filename": "RealESRNet_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    },
}
DEFAULT_MODEL = "swinir"

_models = {}
_device = None


def _ensure_weights(name: str) -> Path:
    spec = MODELS[name]
    path = WEIGHTS_DIR / spec["filename"]
    if path.exists():
        return path
    import requests

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    resp = requests.get(spec["url"], timeout=180)
    resp.raise_for_status()
    tmp.write_bytes(resp.content)
    tmp.rename(path)
    return path


def _get_model(name: str):
    global _device
    if name not in _models:
        import spandrel
        import torch

        path = _ensure_weights(name)
        _device = _device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = spandrel.ModelLoader(device=_device).load_from_file(str(path))
        model.eval()
        _models[name] = model
    return _models[name]


TILE_SIZE = 256
TILE_PAD = 16
SCALE = 4


def _upscale_tensor_tiled(model, arr: "torch.Tensor") -> "torch.Tensor":
    """
    Run the model tile-by-tile instead of on the whole image at once.

    A single un-tiled pass over a 960x720 image needs a multi-GB peak
    allocation (confirmed: OOM'd at ~2.6GB with CLIP + BiRefNet already
    resident in the same process's GPU memory during a real lotstretcher.py run,
    despite working fine in isolation). Each tile is padded with
    TILE_PAD pixels of surrounding context so the model has something to
    work with at the tile edges, then that padding is cropped back out of
    the upscaled result before stitching -- standard approach, avoids any
    visible seams without needing to blend overlapping regions.
    """
    import torch

    _, c, h, w = arr.shape
    output = torch.zeros(1, c, h * SCALE, w * SCALE, device=arr.device, dtype=arr.dtype)

    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            y0, y1 = max(0, y - TILE_PAD), min(h, y + TILE_SIZE + TILE_PAD)
            x0, x1 = max(0, x - TILE_PAD), min(w, x + TILE_SIZE + TILE_PAD)
            tile = arr[:, :, y0:y1, x0:x1]

            with torch.no_grad():
                out_tile = model(tile)

            core_h = min(TILE_SIZE, h - y)
            core_w = min(TILE_SIZE, w - x)
            oy, ox = (y - y0) * SCALE, (x - x0) * SCALE
            core = out_tile[:, :, oy:oy + core_h * SCALE, ox:ox + core_w * SCALE]
            output[:, :, y * SCALE:y * SCALE + core_h * SCALE, x * SCALE:x * SCALE + core_w * SCALE] = core

            del out_tile, tile
            if arr.is_cuda:
                torch.cuda.empty_cache()

    return output


def upscale(img: "Image.Image", model_name: str = DEFAULT_MODEL) -> "Image.Image":
    """4x upscale. Preserves an existing alpha channel (RGBA in -> RGBA out)
    by upscaling color and alpha separately, since the model expects RGB."""
    import numpy as np
    import torch
    from PIL import Image

    model = _get_model(model_name)
    has_alpha = img.mode == "RGBA"
    alpha = img.split()[-1] if has_alpha else None
    rgb = img.convert("RGB")

    arr = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(_device)
    out = _upscale_tensor_tiled(model, arr)
    out_arr = out.squeeze(0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray(out_arr, mode="RGB")

    if has_alpha:
        alpha_big = alpha.resize(result.size, Image.LANCZOS)
        result = result.convert("RGBA")
        result.putalpha(alpha_big)

    return result
