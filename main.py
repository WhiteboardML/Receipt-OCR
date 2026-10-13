"""
main.py — Command-line interface for the Receipt OCR Pipeline.

Usage examples:
    # Single image, no database
    python main.py --image receipts/photo.jpg

    # Single image with product database for fuzzy matching
    python main.py --image receipts/photo.jpg --db data/products.csv

    # Batch process a whole folder
    python main.py --batch receipts/

    # Save the preprocessed debug image to output/ for inspection
    python main.py --image receipts/photo.jpg --debug

    # If photos are already straight-on (skip perspective correction)
    python main.py --image receipts/photo.jpg --no-perspective

    # Lower the fuzzy match threshold (more matches, less strict)
    python main.py --image receipts/photo.jpg --db data/products.csv --threshold 65

    # Verbose logging (shows every decision the pipeline makes)
    python main.py --image receipts/photo.jpg -v
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import ReceiptOCRPipeline
from config.settings import VL_BACKEND, VL_SERVER_URL


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("receipt_ocr.log", encoding="utf-8"),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="receipt_ocr",
        description=(
            "Extract table data from receipt photos and export "
            "to GBS Market-compatible Excel files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input — single image XOR batch folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str, metavar="PATH",
        help="Path to a single receipt image (.jpg or .png)."
    )
    input_group.add_argument(
        "--batch", type=str, metavar="FOLDER",
        help="Path to a folder. All .jpg and .png files inside will be processed."
    )

    # Optional: product database for fuzzy matching
    parser.add_argument(
        "--db", type=str, default=None, metavar="CSV",
        help=(
            "Path to a CSV of your GBS Market product catalogue. "
            "Used for fuzzy name matching. "
            "Export your products from GBS Market → save as products.csv."
        )
    )

    # Fuzzy match sensitivity
    parser.add_argument(
        "--threshold", type=int, default=75, metavar="0-100",
        help=(
            "Minimum RapidFuzz score to accept a product name match (default: 75). "
            "Lower = more matches but possibly wrong; Higher = fewer but more accurate."
        )
    )

    # Debug
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Save the preprocessed (cleaned) image to output/ as debug_<name>.png. "
            "Useful for checking if blue marks were removed and threshold is good."
        )
    )

    # Perspective correction toggle
    parser.add_argument(
        "--no-perspective", action="store_true",
        help=(
            "Disable automatic perspective correction. "
            "Use this if your photos are already taken straight-on, "
            "or if the perspective detection is cropping the image incorrectly."
        )
    )

    # Blue marks removal toggle
    parser.add_argument(
        "--no-blue-removal", action="store_true",
        help="Disable blue ink/pen mark removal from receipt images."
    )

    # OCR Engine options
    parser.add_argument(
        "--engine", type=str, default="vl", choices=["vl", "structure"],
        help="OCR engine: 'vl' (PaddleOCR-VL ViT model, default) or 'structure' (legacy PP-StructureV3)."
    )
    parser.add_argument(
        "--vl-version", type=str, default="v1.6", choices=["v1.6", "v1.5", "v1"],
        help="PaddleOCR-VL pipeline version (default: v1.6)."
    )
    parser.add_argument(
        "--vl-backend", type=str, default=VL_BACKEND,
        help=f"Inference backend for PaddleOCR-VL: mlx-vlm-server, native, etc. (default: {VL_BACKEND})."
    )
    parser.add_argument(
        "--vl-server-url", type=str, default=VL_SERVER_URL,
        help=f"Server URL for MLX/remote VLM backend (default: {VL_SERVER_URL})."
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable GPU acceleration if available."
    )

    # Verbose logging
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG-level) logging."
    )

    return parser


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    setup_logging(args.verbose)

    # Validate paths before initialising heavy models
    if args.image and not Path(args.image).exists():
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        return 1
    if args.batch and not Path(args.batch).is_dir():
        print(f"Error: batch folder not found: {args.batch}", file=sys.stderr)
        return 1

    # Build the pipeline
    pipeline = ReceiptOCRPipeline(
        product_db_path=args.db,
        save_debug_images=args.debug,
        fuzzy_threshold=args.threshold,
        auto_perspective=not args.no_perspective,
        remove_blue_marks=not args.no_blue_removal,
        engine_type=args.engine,
        vl_version=args.vl_version,
        vl_backend=args.vl_backend,
        vl_server_url=args.vl_server_url,
        use_gpu=args.gpu,
    )

    # ── Single image ─────────────────────────────────────────────────
    if args.image:
        output = pipeline.run(args.image)
        if output:
            print(f"✓ Saved → {output}")
            return 0
        else:
            print("✗ Pipeline failed. Check receipt_ocr.log for details.")
            return 1

    # ── Batch ─────────────────────────────────────────────────────────
    if args.batch:
        outputs = pipeline.run_batch(args.batch)
        success = [o for o in outputs if o is not None]
        failed  = len(outputs) - len(success)

        print(f"\n✓ Batch complete: {len(success)}/{len(outputs)} images processed.")
        for path in success:
            print(f"  → {path}")
        if failed:
            print(f"  ✗ {failed} image(s) failed — see receipt_ocr.log for details.")

        return 0 if not failed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())