"""
mcps/hair_color/dump_training_crops.py

Walk a raw Kaggle hair-type dataset and produce masked, grey-background
hair crops using the exact same segmentation pipeline as the server.

Input layout (default: ~/Downloads/data/):
    Straight/  Wavy/  curly/  dreadlocks/  Bald/

Output layout (default: video-insight-engine/data/hair_crops/):
    straight/  wavy/  curly/  dreadlocks/  bald/

KEY DIFFERENCE FROM ORIGINAL:
    Bald images are handled with debug_extract_bald_crop() instead of
    debug_extract_masked_crop(). Bald heads have near-zero hair mask pixels
    so the original function returned None for most of them — that's why
    you only got 77 bald samples. The bald crop function instead produces
    a grey-filled face-above crop, which is exactly what the server sends
    EfficientNet for a bald person at inference time.

Usage:
    python -m mcps.hair_color.dump_training_crops
    python -m mcps.hair_color.dump_training_crops --input /path/to/data --output /path/to/crops
    python -m mcps.hair_color.dump_training_crops --bald-only   # re-run bald only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [dump_crops] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CLASS_NORMALIZE: dict[str, str] = {
    "Straight":    "straight",
    "straight":    "straight",
    "Wavy":        "wavy",
    "wavy":        "wavy",
    "Curly":       "curly",
    "curly":       "curly",
    "Dreadlocks":  "dreadlocks",
    "dreadlocks":  "dreadlocks",
    "Bald":        "bald",
    "bald":        "bald",
}

# Bald uses a different extraction function — grey-filled face crop
BALD_LABELS = {"bald"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_INPUT  = Path.home() / "Downloads" / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "hair_crops"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump masked hair crops for training")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Root of raw Kaggle dataset (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Root of output crop directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--bald-only", action="store_true",
        help="Only re-process the Bald class (useful for topping up bald samples)",
    )
    parser.add_argument(
        "--clear-bald", action="store_true",
        help="Delete existing bald crops before re-running (use with --bald-only)",
    )
    args = parser.parse_args()

    input_root:  Path = args.input
    output_root: Path = args.output

    if not input_root.is_dir():
        logger.error("Input directory does not exist: %s", input_root)
        sys.exit(1)

    # Late import so MediaPipe / OpenCV only load when actually running
    from mcps.hair_color.server import (
        debug_extract_masked_crop,
        debug_extract_bald_crop,
    )

    # Discover class folders
    class_dirs = sorted(
        d for d in input_root.iterdir()
        if d.is_dir() and d.name in CLASS_NORMALIZE
    )
    if not class_dirs:
        logger.error(
            "No recognized class folders in %s. Expected: %s",
            input_root, list(CLASS_NORMALIZE.keys()),
        )
        sys.exit(1)

    if args.bald_only:
        class_dirs = [d for d in class_dirs if CLASS_NORMALIZE[d.name] in BALD_LABELS]
        if not class_dirs:
            logger.error("--bald-only specified but no Bald folder found in %s", input_root)
            sys.exit(1)
        logger.info("--bald-only: processing only %s", [d.name for d in class_dirs])

    if args.clear_bald:
        for label in BALD_LABELS:
            bald_out = output_root / label
            if bald_out.exists():
                import shutil
                shutil.rmtree(bald_out)
                logger.info("Cleared existing bald crops: %s", bald_out)

    logger.info("Input:  %s", input_root)
    logger.info("Output: %s", output_root)
    logger.info("Classes found: %s", [d.name for d in class_dirs])

    total_saved    = 0
    total_skipped  = 0

    for class_dir in class_dirs:
        label   = CLASS_NORMALIZE[class_dir.name]
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        is_bald = label in BALD_LABELS

        images = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        logger.info(
            "[%s] %d images found — using %s extractor",
            label, len(images),
            "BALD (grey face-crop)" if is_bald else "HAIR (masked crop)",
        )

        saved   = 0
        skipped = 0

        for img_path in images:
            try:
                image_bytes = img_path.read_bytes()

                if is_bald:
                    result = debug_extract_bald_crop(image_bytes)
                else:
                    result = debug_extract_masked_crop(image_bytes)

                if result is None:
                    skipped += 1
                    continue

                crop_pil, hair_px = result

                # For non-bald: skip if too few hair pixels
                # For bald: always save (hair_px will be 0, that's correct)
                if not is_bald and hair_px < 20:
                    skipped += 1
                    continue

                stem      = f"{img_path.stem}_{uuid4().hex[:6]}"
                save_path = out_dir / f"{stem}.jpg"
                crop_pil.save(save_path, "JPEG", quality=95)
                saved += 1

            except Exception:
                logger.exception("Failed on %s", img_path.name)
                skipped += 1

        logger.info("[%s] saved=%d  skipped=%d", label, saved, skipped)
        total_saved   += saved
        total_skipped += skipped

    logger.info("DONE — total saved=%d  skipped=%d", total_saved, total_skipped)
    logger.info("")

    # Print final class distribution
    logger.info("Final class distribution in %s:", output_root)
    for cls_dir in sorted(output_root.iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob("*.jpg")))
            logger.info("  %-14s  %d", cls_dir.name, n)


if __name__ == "__main__":
    main()