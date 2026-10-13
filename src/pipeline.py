"""
pipeline.py — Top-level orchestrator for the receipt OCR workflow.

This class wires all four stages together in the correct order:
    1. ImagePreprocessor  →  clean image
    2. TableExtractor     →  raw DataFrame
    3. DataValidator      →  Product list + failed rows
    4. Export             →  .xlsx for GBS Market

Keeping the orchestration here (not in main.py) means you can use the
pipeline from a notebook, a test, or a future GUI without touching main.py.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import OCRResult
from .preprocessor import ImagePreprocessor
from .extractor import TableExtractor
from .validator import DataValidator
from config.settings import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    BLUE_HSV_LOWER,
    BLUE_HSV_UPPER,
    GBS_COLUMN_MAP,
    OCR_ENGINE,
    OUTPUT_DIR,
    PRIMARY_LANG,
    SECONDARY_LANG,
    VL_BACKEND,
    VL_PIPELINE_VERSION,
    VL_SERVER_URL,
    VL_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class ReceiptOCRPipeline:
    """
    Processes a receipt image from start to finish and writes an Excel file
    ready for import into GBS Market.

    Quick start:
        pipeline = ReceiptOCRPipeline(product_db_path="data/products.csv")
        output_path = pipeline.run("receipts/my_photo.jpg")

    With debug images:
        pipeline = ReceiptOCRPipeline(save_debug_images=True)
        pipeline.run("receipts/my_photo.jpg")
        # Check output/debug_my_photo.png to see what the preprocessor produced
    """

    def __init__(
        self,
        product_db_path:    Optional[str] = None,
        save_debug_images:  bool          = False,
        fuzzy_threshold:    int           = 75,
        auto_perspective:   bool          = True,
        remove_blue_marks:  bool          = True,
        engine_type:        str           = OCR_ENGINE,
        vl_version:         str           = VL_PIPELINE_VERSION,
        vl_backend:         str           = VL_BACKEND,
        vl_server_url:      str           = VL_SERVER_URL,
        vl_model_name:      Optional[str] = VL_MODEL_NAME,
        use_gpu:            bool          = False,
    ):
        """
        Args:
            product_db_path:   CSV exported from GBS Market for fuzzy name matching.
            save_debug_images: If True, saves the preprocessed image alongside output.
            fuzzy_threshold:   0–100. Minimum match score to accept a product name.
            auto_perspective:  Disable if photos are already straight-on.
            remove_blue_marks: Disable if receipts have no handwritten marks.
            engine_type:       'vl' (PaddleOCR-VL ViT) or 'structure' (PP-StructureV3).
            vl_version:        PaddleOCR-VL version ('v1.6', 'v1.5', 'v1').
            vl_backend:        PaddleOCR-VL inference backend ('native', 'mlx-vlm-server', etc.).
            vl_server_url:     Server URL for MLX/remote VLM backend.
            vl_model_name:     Model identifier or path for VLM backend.
            use_gpu:           Enable GPU acceleration if available.
        """
        self.save_debug_images = save_debug_images
        self.engine_type       = engine_type
        self.vl_version        = vl_version
        self.vl_backend        = vl_backend
        self.vl_server_url     = vl_server_url
        self.vl_model_name     = vl_model_name
        self.use_gpu           = use_gpu

        # Instantiate each stage
        self.preprocessor = ImagePreprocessor(
            blue_lower=BLUE_HSV_LOWER,
            blue_upper=BLUE_HSV_UPPER,
            block_size=ADAPTIVE_BLOCK_SIZE,
            adaptive_c=ADAPTIVE_C,
            auto_perspective=auto_perspective,
            remove_blue_marks=remove_blue_marks,
        )

        self.extractor = TableExtractor(
            engine_type=engine_type,
            pipeline_version=vl_version,
            vl_backend=vl_backend,
            vl_server_url=vl_server_url,
            vl_model_name=vl_model_name,
            primary_lang=PRIMARY_LANG,
            secondary_lang=SECONDARY_LANG,
            use_gpu=use_gpu,
        )

        self.validator = DataValidator(
            product_db_path=product_db_path,
            fuzzy_threshold=fuzzy_threshold,
        )

        logger.info(
            "ReceiptOCRPipeline ready. engine=%s (version=%s), "
            "debug_images=%s, fuzzy_threshold=%d, db=%s",
            engine_type, vl_version, save_debug_images, fuzzy_threshold, product_db_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image_path: str) -> Optional[Path]:
        """
        Full pipeline for a single receipt image.

        Returns:
            Path to the output .xlsx file, or None if the pipeline failed.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("=" * 60)
        logger.info("Processing: %s", image_path.name)
        logger.info("=" * 60)

        # ── Stage 1: Preprocess ──────────────────────────────────────
        logger.info("[1/4] Image preprocessing ...")
        clean_image = self.preprocessor.process(str(image_path))

        if self.save_debug_images:
            debug_path = OUTPUT_DIR / f"debug_{image_path.stem}.png"
            self.preprocessor.save(clean_image, str(debug_path))
            logger.info("Debug image saved → %s", debug_path)

        # ── Stage 2: Extract table ───────────────────────────────────
        logger.info("[2/4] Extracting table with PP-Structure ...")
        raw_df = self.extractor.extract(clean_image)

        if raw_df.empty:
            logger.error(
                "No table data extracted from %s. "
                "Suggestions: use --debug to inspect the preprocessed image; "
                "try --no-perspective if the document boundary was wrongly detected.",
                image_path.name,
            )
            return None

        # ── Stage 3: Validate & clean ────────────────────────────────
        logger.info("[3/4] Validating and cleaning extracted data ...")
        products, failed_df = self.validator.validate(raw_df)

        if not products:
            logger.error("Validation produced 0 products. Check the raw log above.")
            return None

        # ── Stage 4: Export ──────────────────────────────────────────
        logger.info("[4/4] Exporting to Excel ...")
        output_path = self._export_to_excel(
            products=products,
            failed_df=failed_df,
            stem=image_path.stem,
        )

        self._print_summary(products, failed_df, output_path)
        return output_path

    def run_batch(self, image_folder: str) -> list[Optional[Path]]:
        """
        Processes every .jpg and .png in a folder.

        Returns a list of output paths (None entries for failed images).
        """
        folder = Path(image_folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not images:
            logger.warning("No .jpg or .png files found in %s", folder)
            return []

        logger.info("Batch processing %d images from %s ...", len(images), folder)
        results = []
        for img in images:
            try:
                results.append(self.run(str(img)))
            except Exception as exc:
                logger.error("Failed to process %s: %s", img.name, exc)
                results.append(None)

        success = sum(1 for r in results if r is not None)
        logger.info("Batch complete: %d/%d succeeded.", success, len(images))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _export_to_excel(
        self,
        products:   list,
        failed_df:  pd.DataFrame,
        stem:       str,
    ) -> Path:
        """
        Writes a multi-sheet Excel workbook:
          Sheet 1 "GBS Import"          — ready-to-import data
          Sheet 2 "Требует проверки"    — rows that failed (if any)
          Sheet 3 "Сводка"              — summary statistics
        """
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{stem}_{timestamp}.xlsx"
        gbs_df      = self.validator.to_gbs_dataframe(products)

        # Confidence breakdown for the summary sheet
        high   = sum(1 for p in products if p.confidence == "high")
        medium = sum(1 for p in products if p.confidence == "medium")
        low    = sum(1 for p in products if p.confidence == "low")

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

            # Sheet 1: GBS Market import data
            gbs_df.to_excel(writer, sheet_name="GBS Import", index=False)

            # Sheet 2: Failed rows (manual review)
            if not failed_df.empty:
                flag_cols = [c for c in failed_df.columns if str(c).startswith("_")]
                failed_df.drop(columns=flag_cols, errors="ignore").to_excel(
                    writer, sheet_name="Требует проверки", index=False
                )

            # Sheet 3: Summary
            model_label = (
                f"PaddleOCR-VL {self.vl_version} (ViT)"
                if self.engine_type == "vl"
                else f"PP-StructureV3 ({PRIMARY_LANG})"
            )
            summary_data = {
                "Показатель": [
                    "Модель OCR",
                    "Всего строк",
                    "Успешно распознано",
                    "Требует проверки",
                    "— Высокая точность (≥90)",
                    "— Средняя точность (75–89)",
                    "— Низкая точность (<75)",
                    "Дата обработки",
                    "Исходный файл",
                ],
                "Значение": [
                    model_label,
                    len(products) + len(failed_df),
                    len(products),
                    len(failed_df),
                    high,
                    medium,
                    low,
                    timestamp,
                    stem,
                ],
            }
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Сводка", index=False
            )

        logger.info("Exported %d products → %s", len(products), output_path)
        return output_path

    def _print_summary(self, products: list, failed_df: pd.DataFrame, path: Path) -> None:
        """Prints a human-readable summary to stdout after each run."""
        high   = sum(1 for p in products if p.confidence == "high")
        medium = sum(1 for p in products if p.confidence == "medium")
        low    = sum(1 for p in products if p.confidence == "low")
        model_name = (
            f"PaddleOCR-VL {self.vl_version} (ViT)"
            if self.engine_type == "vl"
            else "PP-StructureV3"
        )

        print("\n" + "=" * 50)
        print(f"  ✓ Output: {path.name}")
        print(f"  ✓ Engine: {model_name}")
        print(f"  ✓ Products extracted : {len(products)}")
        print(f"    ● High confidence  : {high}")
        print(f"    ● Medium confidence: {medium}")
        print(f"    ● Low confidence   : {low}")
        if len(failed_df):
            print(f"  ✗ Failed rows       : {len(failed_df)} → sheet 'Требует проверки'")
        print("=" * 50 + "\n")
