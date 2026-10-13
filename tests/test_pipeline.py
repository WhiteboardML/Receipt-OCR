"""
test_pipeline.py — Unit tests for the receipt OCR pipeline.

All tests here use SYNTHETIC / MOCK data so you can run them without
PaddleOCR installed. They verify the parsing, normalisation, and fuzzy
matching logic — the parts most likely to break as you tune the pipeline.

Run:
    pytest tests/ -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make the project root importable regardless of where pytest is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import Product
from src.preprocessor import ImagePreprocessor
from src.validator import DataValidator
from src.extractor import (
    _is_latin_dominant,
    _markdown_to_df,
    _parse_table_string,
    TableExtractor,
)


# ===========================================================================
# DataValidator — number parsing
# ===========================================================================

class TestParseNumber:
    """Covers the price/quantity parsing that OCR commonly mangles."""

    def setup_method(self):
        self.v = DataValidator()

    def test_space_as_thousands_separator(self):
        assert self.v._parse_number("97 700") == 97700.0

    def test_space_thousands_with_decimal(self):
        assert self.v._parse_number("1 234,50") == 1234.5

    def test_plain_float(self):
        assert self.v._parse_number("2.5") == 2.5

    def test_integer_string(self):
        assert self.v._parse_number("1") == 1.0

    def test_nan_returns_zero(self):
        assert self.v._parse_number(float("nan")) == 0.0

    def test_non_numeric_returns_zero(self):
        assert self.v._parse_number("шт.") == 0.0

    def test_price_with_currency_symbol(self):
        # OCR might include "сум" or "UZS" after the number
        assert self.v._parse_number("97700 сум") == 97700.0

    def test_already_float(self):
        assert self.v._parse_number(150.0) == 150.0

    def test_zero_string(self):
        assert self.v._parse_number("0") == 0.0


# ===========================================================================
# DataValidator — unit normalisation
# ===========================================================================

class TestNormalizeUnit:
    """Verifies that all alias forms collapse to canonical units."""

    def setup_method(self):
        self.v = DataValidator()

    def test_kg_variants(self):
        for alias in ["кг", "кг.", "kg", "кило"]:
            assert self.v._normalize_unit(alias) == "кг", f"Failed for: {alias}"

    def test_sht_variants(self):
        for alias in ["шт.", "шт", "dona", "don", "unit"]:
            assert self.v._normalize_unit(alias) == "шт", f"Failed for: {alias}"

    def test_gram_variants(self):
        for alias in ["г", "г.", "gr", "гр", "гр."]:
            assert self.v._normalize_unit(alias) == "г", f"Failed for: {alias}"

    def test_litre_variants(self):
        for alias in ["л", "л.", "l", "litr"]:
            assert self.v._normalize_unit(alias) == "л", f"Failed for: {alias}"

    def test_ml_variants(self):
        for alias in ["мл", "мл.", "ml"]:
            assert self.v._normalize_unit(alias) == "мл", f"Failed for: {alias}"

    def test_unknown_unit_kept_as_is(self):
        assert self.v._normalize_unit("xyz") == "xyz"

    def test_leading_trailing_whitespace(self):
        assert self.v._normalize_unit("  кг.  ") == "кг"


# ===========================================================================
# DataValidator — column detection
# ===========================================================================

class TestRenameColumns:
    def setup_method(self):
        self.v = DataValidator()

    def test_russian_headers(self):
        df = pd.DataFrame(columns=["Наименование", "Количество", "Цена", "Сумма", "Ед. изм."])
        result = self.v._rename_columns(df)
        assert "name" in result.columns
        assert "quantity" in result.columns
        assert "unit_price" in result.columns
        assert "total_price" in result.columns
        assert "unit_measurement" in result.columns

    def test_uzbek_latin_headers(self):
        df = pd.DataFrame(columns=["Nomi", "Miqdor", "O'lchov", "Narx", "Jami"])
        result = self.v._rename_columns(df)
        assert "name" in result.columns
        assert "quantity" in result.columns
        assert "unit_measurement" in result.columns
        assert "unit_price" in result.columns
        assert "total_price" in result.columns

    def test_english_headers(self):
        df = pd.DataFrame(columns=["Product Name", "Qty", "Unit", "Price", "Total"])
        result = self.v._rename_columns(df)
        assert "name" in result.columns
        assert "quantity" in result.columns

    def test_positional_fallback_on_numeric_headers(self):
        df = pd.DataFrame(columns=[0, 1, 2, 3, 4])
        result = self.v._rename_columns(df)
        # Positional fallback should assign name, quantity, unit_measurement, unit_price, total_price
        assert "name" in result.columns
        assert "quantity" in result.columns

    def test_sku_detection(self):
        df = pd.DataFrame(columns=["Артикул", "Наименование"])
        result = self.v._rename_columns(df)
        assert "sku" in result.columns

    def test_category_detection(self):
        df = pd.DataFrame(columns=["Категория", "Наименование"])
        result = self.v._rename_columns(df)
        assert "category" in result.columns


# ===========================================================================
# DataValidator — full row validation
# ===========================================================================

class TestRowToProduct:
    def setup_method(self):
        self.v = DataValidator()

    def test_basic_row(self):
        row = pd.Series({
            "name":             "Скраб для лица",
            "quantity":         "2",
            "unit_measurement": "шт.",
            "unit_price":       "15 000",
            "total_price":      "30 000",
        })
        product = self.v._row_to_product(row)
        assert product.name == "Скраб для лица"
        assert product.quantity == 2.0
        assert product.unit_measurement == "шт"
        assert product.unit_price == 15000.0
        assert product.total_price == 30000.0

    def test_total_price_computed_when_missing(self):
        row = pd.Series({
            "name":             "Молоко",
            "quantity":         "3",
            "unit_measurement": "л",
            "unit_price":       "8000",
            "total_price":      "0",     # missing — should be computed
        })
        product = self.v._row_to_product(row)
        assert product.total_price == 24000.0

    def test_optional_fields_none_when_absent(self):
        row = pd.Series({
            "name":             "Test Product",
            "quantity":         "1",
            "unit_measurement": "шт",
            "unit_price":       "1000",
            "total_price":      "1000",
        })
        product = self.v._row_to_product(row)
        assert product.sku is None
        assert product.category is None


# ===========================================================================
# DataValidator — fuzzy matching
# ===========================================================================

class TestFuzzyMatch:
    def setup_method(self):
        self.v = DataValidator()
        # Manually load a small product list (simulates the GBS Market CSV)
        self.v.product_names = [
            "Скраб для лица 100г",
            "Молоко 3.5% Parmalat 1л",
            "Скотч прозрачный 48мм",
            "Шампунь Head & Shoulders 400мл",
        ]

    def test_exact_match(self):
        name, score = self.v._fuzzy_match("Молоко 3.5% Parmalat 1л")
        assert name == "Молоко 3.5% Parmalat 1л"
        assert score >= 95

    def test_ocr_error_match(self):
        # "Скpa6" is a common OCR error for "Скраб" (Cyrillic б → 6)
        name, score = self.v._fuzzy_match("Скpa6 для лица 100г", threshold=60)
        assert name is not None
        assert "Скраб" in name

    def test_no_match_below_threshold(self):
        name, score = self.v._fuzzy_match("xyz абракадабра", threshold=75)
        assert name is None
        assert score == 0.0

    def test_no_db_returns_none(self):
        v = DataValidator()   # no DB
        name, score = v._fuzzy_match("Anything")
        assert name is None
        assert score == 0.0


# ===========================================================================
# DataValidator — to_gbs_dataframe
# ===========================================================================

class TestGBSExport:
    def test_column_headers_match_gbs_map(self):
        from config.settings import GBS_COLUMN_MAP
        v = DataValidator()
        products = [
            Product(
                name="Молоко",
                quantity=1.0,
                unit_measurement="л",
                unit_price=8000.0,
                total_price=8000.0,
            )
        ]
        df = v.to_gbs_dataframe(products)
        for internal, gbs_header in GBS_COLUMN_MAP.items():
            assert gbs_header in df.columns, f"Missing GBS column: {gbs_header}"


# ===========================================================================
# Product model
# ===========================================================================

class TestProduct:
    def test_to_gbs_dict_uses_matched_name(self):
        from config.settings import GBS_COLUMN_MAP
        p = Product(
            name="Скpa6 лица",
            raw_name="Скpa6 лица",
            matched_name="Скраб для лица 100г",  # corrected via fuzzy match
            quantity=2.0,
            unit_measurement="шт",
            unit_price=15000.0,
            total_price=30000.0,
        )
        d = p.to_gbs_dict(GBS_COLUMN_MAP)
        assert d[GBS_COLUMN_MAP["name"]] == "Скраб для лица 100г"

    def test_to_gbs_dict_falls_back_to_raw_name(self):
        from config.settings import GBS_COLUMN_MAP
        p = Product(
            name="Unique Product",
            quantity=1.0,
            unit_measurement="шт",
            unit_price=5000.0,
            total_price=5000.0,
            matched_name=None,
        )
        d = p.to_gbs_dict(GBS_COLUMN_MAP)
        assert d[GBS_COLUMN_MAP["name"]] == "Unique Product"


# ===========================================================================
# ImagePreprocessor — geometric helpers
# ===========================================================================

class TestImagePreprocessor:
    def test_order_points_correct_order(self):
        pts = np.array([
            [100, 100],   # bottom-right
            [0,   0  ],   # top-left
            [100, 0  ],   # top-right
            [0,   100],   # bottom-left
        ], dtype="float32")
        ordered = ImagePreprocessor._order_points(pts)
        assert list(ordered[0]) == [0.0,   0.0  ]   # top-left
        assert list(ordered[1]) == [100.0, 0.0  ]   # top-right
        assert list(ordered[2]) == [100.0, 100.0]   # bottom-right
        assert list(ordered[3]) == [0.0,   100.0]   # bottom-left

    def test_load_missing_file_raises(self):
        prep = ImagePreprocessor()
        with pytest.raises(FileNotFoundError, match="cv2.imread could not open"):
            prep._load("this_file_does_not_exist.jpg")

    def test_binarize_returns_3channel(self):
        prep  = ImagePreprocessor()
        # Create a synthetic 100×100 BGR image
        img   = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = prep._binarize(img)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_blue_mark_removal_replaces_blue_with_white(self):
        prep = ImagePreprocessor()
        # Create a white image with a blue square in the centre
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[40:60, 40:60] = [255, 0, 0]   # BGR blue (H≈120 in HSV)

        result = prep._remove_blue_marks(img.copy())
        # Blue square should now be white
        assert np.all(result[45:55, 45:55] == 255)


# ===========================================================================
# Extractor helpers
# ===========================================================================

class TestLatinDetection:
    def test_pure_latin_is_dominant(self):
        assert _is_latin_dominant("Olma sharbati") is True

    def test_pure_cyrillic_not_dominant(self):
        assert _is_latin_dominant("Скраб для лица") is False

    def test_mixed_mostly_cyrillic(self):
        assert _is_latin_dominant("Скраб 200ml") is False

    def test_numbers_only_not_dominant(self):
        assert _is_latin_dominant("97 700") is False

    def test_empty_string(self):
        assert _is_latin_dominant("") is False

    def test_english_is_dominant(self):
        assert _is_latin_dominant("Head and Shoulders") is True

    def test_threshold_respected(self):
        # "AB Цена" — 2 Latin + 4 Cyrillic = 33% Latin → not dominant at 0.6
        assert _is_latin_dominant("AB Цена", threshold=0.6) is False
        # but dominant at 0.3
        assert _is_latin_dominant("AB Цена", threshold=0.3) is True


class TestTableParsing:
    def test_markdown_table_parsing(self):
        md = """
| Товар | Количество | Цена |
|---|---|---|
| Скраб для лица | 2 | 15000 |
| Хлеб | 1 | 4500 |
"""
        df = _markdown_to_df(md)
        assert df is not None
        assert df.shape == (2, 3)
        assert list(df.columns) == ["Товар", "Количество", "Цена"]
        assert df.iloc[0]["Товар"] == "Скраб для лица"

    def test_html_table_parsing(self):
        html = """
<table>
  <tr><th>Товар</th><th>Цена</th></tr>
  <tr><td>Молоко</td><td>9500</td></tr>
</table>
"""
        df = _parse_table_string(html)
        assert df is not None
        assert df.shape == (1, 2)
        assert "Товар" in df.columns
        assert df.iloc[0]["Товар"] == "Молоко"

    def test_table_extractor_config(self):
        extractor_vl = TableExtractor(engine_type="vl", pipeline_version="v1.6")
        assert extractor_vl.engine_type == "vl"
        assert extractor_vl.pipeline_version == "v1.6"

        extractor_struct = TableExtractor(engine_type="structure")
        assert extractor_struct.engine_type == "structure"

    def test_flag_latin_cells(self):
        extractor = TableExtractor()
        df = pd.DataFrame({
            "name": ["Скраб для лица", "Head and Shoulders"],
            "unit": ["шт", "dona"],
        })
        flagged = extractor._flag_latin_cells(df)
        assert "_latin_name_0" in flagged.columns
        assert flagged["_latin_name_0"].tolist() == [False, True]