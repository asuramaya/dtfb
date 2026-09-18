#!/usr/bin/env python3
"""
Turn a scraped vehicle's real angle-labeled cutouts into a rotating
"spin" video -- classical cross-dissolve interpolation between the real
photos already in the gallery, NOT true 3D reconstruction. See
imaging/compose/spin.py's module docstring for what this is and isn't,
and why.

    spin_video_cli.py ~/Documents/listings/2023-Ford-F-150-Raptor-PFA15687
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from lotstretcher.imaging.compose import render_spin_video
from lotstretcher.imaging.compose.spin import DEFAULT_CANVAS_SIZE, MIN_ANCHORS
from lotstretcher.imaging.palette import colors_from_details, vehicle_gradient_colors
from lotstretcher.vehicle_pipeline import video_output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vehicle_folder", help="A folder produced by lotstretcher.py (contains images/exterior/cutout/)")
    parser.add_argument("--budget-mb", type=float, default=50.0, help="Target max file size in MB (default: 50)")
    parser.add_argument("--nvenc", action="store_true",
                         help="Encode on the GPU (h264_nvenc). Falls back to libx264 automatically if the "
                              "GPU is too busy to open the encoder -- see spin.py's render_spin_video().")
    parser.add_argument("--out", help="Output path (default: <vehicle_folder>/bundle/hero-video-spin.mp4)")
    args = parser.parse_args()

    vehicle_folder = Path(args.vehicle_folder)
    cutout_dir = vehicle_folder / "images" / "exterior" / "cutout"
    if not cutout_dir.is_dir():
        sys.exit(f"No cutouts found at {cutout_dir} -- did you run lotstretcher.py on this vehicle yet?")

    out_path = Path(args.out) if args.out else video_output_path(vehicle_folder, "spin")

    ext, inr = colors_from_details(vehicle_folder)
    sample = next(iter(sorted(cutout_dir.glob("*.png"))), None)
    start, end = vehicle_gradient_colors(ext, inr, sample)
    gradient_colors = (random.Random(vehicle_folder.name).uniform(0, 360), start, end)

    report = render_spin_video(
        cutout_dir, out_path, gradient_colors,
        canvas_size=DEFAULT_CANVAS_SIZE, budget_mb=args.budget_mb,
        encoder="h264_nvenc" if args.nvenc else "libx264",
    )
    if report is None:
        sys.exit(f"Not enough distinct real angles for a spin (need {MIN_ANCHORS}+, "
                  f"see {cutout_dir}/angles.json) -- skipped, this isn't an error.")

    print(f"Saved: {report['out_path']} ({report['file_size_mb']} MB, encoder={report['encoder']})")
    print(f"  duration: {report['duration_s']}s, {report['n_anchors']} real anchor(s): "
          f"{' -> '.join(report['angles'])}")


if __name__ == "__main__":
    main()
