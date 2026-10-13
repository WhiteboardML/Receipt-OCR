"""
settings.py — Central configuration for the Receipt OCR Pipeline.

All tunable parameters live here. When something breaks (e.g. checkmarks
are still showing up in OCR output), this is the first file to adjust.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# OCR Model & Language Strategy
#
# Primary Model: PaddleOCR-VL (Vision Transformer / NaViT-based Document VLM)
#   - pipeline_version="v1.6" (PaddleOCR-VL-1.6-0.9B)
#   - Uses dynamic-resolution ViT visual encoder + lightweight LLM
#   - End-to-end receipt table recognition (replaces legacy SLANet / PP-Structure)
#
# Fallback / Alternative: PP-StructureV3
# ---------------------------------------------------------------------------
import platform

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

OCR_ENGINE = "vl"            # "vl" (PaddleOCR-VL ViT) or "structure" (PP-StructureV3)
VL_PIPELINE_VERSION = "v1.6" # "v1.6", "v1.5", or "v1"
VL_BACKEND = "mlx-vlm-server" if IS_APPLE_SILICON else "native"
VL_SERVER_URL = "http://127.0.0.1:8111/v1"
VL_MODEL_NAME = str(Path.home() / ".paddlex" / "official_models" / "PaddleOCR-VL-1.6")

PRIMARY_LANG = "ru"          # Handles Russian + Uzbek Cyrillic (for structure engine)
SECONDARY_LANG = "en"        # Handles Uzbek Latin + English

# If more than this fraction of alpha chars in a cell are ASCII Latin,
# treat the cell as Latin-script and flag it.
LATIN_CHAR_RATIO_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Image Preprocessing — OpenCV
# ---------------------------------------------------------------------------

# HSV colour range for blue handwritten checkmarks.
# Tweak BLUE_HSV_LOWER/UPPER if marks aren't being erased fully.
# Useful tool: run the debug script (see README) to inspect HSV values
# of your specific pen colour.
BLUE_HSV_LOWER = (100, 50,  50)   # (H_min, S_min, V_min)
BLUE_HSV_UPPER = (130, 255, 255)  # (H_max, S_max, V_max)

# Adaptive threshold — controls how aggressively shadows are removed.
# BLOCK_SIZE must be an odd number. Larger = smoother but slower.
# C is subtracted from the mean; higher = more aggressive binarization.
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# ---------------------------------------------------------------------------
# Fuzzy Matching — RapidFuzz
# ---------------------------------------------------------------------------
FUZZY_MATCH_THRESHOLD = 75   # 0–100. Below this, match is rejected.
FUZZY_MATCH_LIMIT = 3        # How many candidates to evaluate per cell.

# ---------------------------------------------------------------------------
# GBS Market Column Mapping
#
# Maps internal field names → exact column headers GBS Market expects.
# If GBS Market uses different headers, only edit the VALUES here.
# ---------------------------------------------------------------------------
GBS_COLUMN_MAP = {
    "sku":              "Артикул",
    "name":             "Наименование",
    "category":         "Категория",
    "quantity":         "Количество",
    "unit_measurement": "Единица измерения",
    "unit_price":       "Цена",
    "total_price":      "Сумма",
}

# ---------------------------------------------------------------------------
# Unit Measurement Normalisation
#
# Structure: { canonical_unit : [list of aliases from OCR] }
# The validator collapses all aliases to the canonical form.
# Add new aliases as you discover what your specific receipts produce.
# ---------------------------------------------------------------------------
UNIT_ALIASES: dict[str, list[str]] = {
    "шт":  ["шт.", "шт", "dona", "don", "шт,", "unit", "ед", "ед.", "piece", "pcs"],
    "кг":  ["кг", "кг.", "kg", "кило", "kilogram"],
    "г":   ["г", "г.", "gr", "гр", "гр.", "gramm", "gram"],
    "л":   ["л", "л.", "l", "litr", "литр", "litre"],
    "мл":  ["мл", "мл.", "ml", "millilitr"],
    "м":   ["м", "м.", "m", "метр", "metr"],
    "см":  ["см", "см.", "cm", "santimetr"],
    "уп":  ["уп", "уп.", "упак", "упаковка", "pack", "packaging", "qadoq"],
    "пара":["пара", "пар", "para", "pair"],
    "к-т": ["к-т", "комплект", "komplekt", "set"],
}