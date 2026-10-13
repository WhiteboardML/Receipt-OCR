"""
validator.py — Cleans messy OCR text into validated Product objects.

Three jobs:
  1. Column detection: receipts don't always have clean headers.
     We scan column names for known Russian/Uzbek/English keywords.

  2. Data parsing: prices like "97 700" → 97700.0; units like "кг." → "кг".

  3. Fuzzy name matching: OCR might read "Скраб" as "Скpa6".
     We compare against your GBS Market product database to find the
     closest real product name.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from .models import Product
from config.settings import (
    FUZZY_MATCH_THRESHOLD,
    GBS_COLUMN_MAP,
    UNIT_ALIASES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the reverse unit lookup once at import time:
#   {"кг." → "кг", "kg" → "кг", "dona" → "шт", ...}
# ---------------------------------------------------------------------------
UNIT_LOOKUP: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in UNIT_ALIASES.items()
    for alias in aliases
}

# ---------------------------------------------------------------------------
# Regex patterns for price/number parsing
# ---------------------------------------------------------------------------
# Strips everything that isn't a digit, decimal point, or comma.
_NON_NUMERIC_RE = re.compile(r"[^\d.,]")
# Detects a comma acting as decimal separator: "1234,50" (European format)
_COMMA_DECIMAL_RE = re.compile(r",(\d{1,2})$")


class DataValidator:
    """
    Converts a raw OCR DataFrame into a list of clean Product objects.

    Usage:
        validator = DataValidator(product_db_path="data/products.csv")
        products, failed_df = validator.validate(raw_df)
        gbs_df = validator.to_gbs_dataframe(products)
    """

    def __init__(
        self,
        product_db_path: Optional[str] = None,
        fuzzy_threshold: int = FUZZY_MATCH_THRESHOLD,
    ):
        """
        Args:
            product_db_path: Path to a CSV exported from GBS Market.
                             Must have a column named 'name' (or a
                             recognised variant — see _find_name_column).
                             If None, fuzzy matching is skipped.
            fuzzy_threshold: Minimum rapidfuzz score (0–100) to accept a match.
                             75 is a good starting point; lower if you get too many
                             "low confidence" flags, raise if you get wrong matches.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.product_db:    list[dict] = []
        self.product_names: list[str]  = []

        if product_db_path and Path(product_db_path).exists():
            self._load_product_db(product_db_path)
        elif product_db_path:
            logger.warning("Product DB path '%s' not found. Fuzzy matching disabled.", product_db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> Tuple[list[Product], pd.DataFrame]:
        """
        Main entry point.

        Returns:
            products:   List of validated Product objects, ready for export.
            failed_df:  DataFrame of rows that failed parsing (for manual review).
                        Saved to 'Требует проверки' sheet in the Excel output.
        """
        df = self._drop_flag_cols_copy(df)          # work on a clean copy
        df = self._rename_columns(df)
        df = self._drop_empty_rows(df)

        products:   list[Product] = []
        failed_rows: list[dict]   = []

        for idx, row in df.iterrows():
            # Check if this row's name cell was flagged as Latin-dominant
            # (the flag travels alongside as _latin_<original_col>)
            is_latin = self._was_latin_cell(row, df)

            try:
                product = self._row_to_product(row, is_latin=is_latin)
                products.append(product)
            except Exception as exc:
                logger.warning(
                    "Row %d failed validation: %s | Error: %s",
                    idx, {k: v for k, v in row.items() if not str(k).startswith("_")}, exc
                )
                failed_rows.append(
                    {k: v for k, v in row.items() if not str(k).startswith("_")}
                )

        failed_df = pd.DataFrame(failed_rows)
        logger.info(
            "Validation complete — %d products OK, %d rows failed.",
            len(products), len(failed_rows),
        )
        return products, failed_df

    def to_gbs_dataframe(self, products: list[Product]) -> pd.DataFrame:
        """
        Builds a DataFrame that matches GBS Market's expected import format.
        Column headers come from GBS_COLUMN_MAP in config/settings.py.
        """
        rows = [p.to_gbs_dict(GBS_COLUMN_MAP) for p in products]
        df   = pd.DataFrame(rows)
        logger.info("GBS Market DataFrame shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    # Keywords that identify each semantic column role.
    # Format: { internal_name: [keyword_fragments_in_any_language] }
    _COL_KEYWORDS: dict[str, list[str]] = {
        "name":             ["наим", "name", "nom", "nomi", "товар", "product", "назв"],
        "sku":              ["артик", "sku", "код", "barcode", "code", "штрих"],
        "category":         ["катег", "categ", "группа", "guruh", "group"],
        "quantity":         ["кол", "qty", "количест", "dona", "miqdor", "son"],
        "unit_measurement": ["ед", "unit", "изм", "o'lch", "birligi", "мера"],
        "unit_price":       ["цена", "price", "narx", "нарх", "бahо"],
        "total_price":      ["сумм", "итог", "total", "jami", "summ", "итого"],
    }

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps each column to an internal name by keyword matching.
        Unrecognised columns are left as-is (they'll be ignored during row parsing).
        """
        col_map: dict = {}
        used_targets: set = set()

        for col in df.columns:
            if str(col).startswith("_"):   # skip internal flag columns
                continue
            col_str = str(col).lower().strip()
            for internal_name, keywords in self._COL_KEYWORDS.items():
                if internal_name in used_targets:
                    continue
                if any(kw in col_str for kw in keywords):
                    col_map[col] = internal_name
                    used_targets.add(internal_name)
                    break

        if col_map:
            logger.info("Column mapping detected: %s", col_map)
        else:
            logger.warning(
                "No columns could be auto-detected from headers: %s. "
                "The validator will attempt positional fallback.",
                list(df.columns)
            )
            df = self._positional_fallback(df)
            return df

        return df.rename(columns=col_map)

    def _positional_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        When column headers are unrecognisable (e.g. generic 0,1,2,3...),
        assumes the most common receipt column order and assigns names.
        Order assumed: name, quantity, unit_measurement, unit_price, total_price
        """
        positional = ["name", "quantity", "unit_measurement", "unit_price", "total_price"]
        new_cols = {
            old: positional[i]
            for i, old in enumerate(df.columns)
            if i < len(positional) and not str(old).startswith("_")
        }
        logger.info("Using positional fallback column mapping: %s", new_cols)
        return df.rename(columns=new_cols)

    # ------------------------------------------------------------------
    # Row → Product
    # ------------------------------------------------------------------

    def _row_to_product(self, row: pd.Series, is_latin: bool = False) -> Product:
        """Parses a single DataFrame row into a Product."""
        name     = str(row.get("name", "")).strip()
        raw_name = name

        # Optional fields
        sku      = self._clean_str(row.get("sku"))
        category = self._clean_str(row.get("category"))

        # Numeric fields
        quantity    = self._parse_number(row.get("quantity",    1))
        unit_price  = self._parse_number(row.get("unit_price",  0))
        total_price = self._parse_number(row.get("total_price", 0))

        # If total_price is missing but we have quantity × unit_price, compute it
        if total_price == 0.0 and quantity and unit_price:
            total_price = round(quantity * unit_price, 2)

        # Unit measurement
        raw_unit = str(row.get("unit_measurement", "шт"))
        unit     = self._normalize_unit(raw_unit)

        # Fuzzy match — use a slightly lower threshold for Latin cells
        # since OCR accuracy is lower for that script pass
        threshold = max(60, self.fuzzy_threshold - 10) if is_latin else self.fuzzy_threshold
        matched_name, score = self._fuzzy_match(name, threshold=threshold)

        confidence = (
            "high"   if score >= 90                         else
            "medium" if score >= self.fuzzy_threshold       else
            "low"
        )

        return Product(
            name=name,
            raw_name=raw_name,
            sku=sku,
            category=category,
            quantity=quantity,
            unit_measurement=unit,
            unit_price=unit_price,
            total_price=total_price,
            matched_name=matched_name,
            match_score=score,
            confidence=confidence,
            is_latin_cell=is_latin,
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_number(self, value) -> float:
        """
        Robustly converts messy OCR number strings to float.

        Handles:
            "97 700"    → 97700.0   (thousands separator is a space)
            "1 234,50"  → 1234.5    (comma as decimal, space as thousands)
            "2.5"       → 2.5
            "1"         → 1.0
            "шт."       → 0.0       (non-numeric → 0)
            NaN         → 0.0
        """
        if pd.isna(value):
            return 0.0

        s = str(value).strip()

        # Remove everything except digits, dot, comma
        s = _NON_NUMERIC_RE.sub("", s)

        if not s:
            return 0.0

        # Decide: is the comma a decimal separator or a thousands sep?
        # "1234,50" → comma is decimal; "1,234" → comma is thousands
        if _COMMA_DECIMAL_RE.search(s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

        # Remove spaces (thousands separator)
        s = s.replace(" ", "")

        try:
            return float(s)
        except ValueError:
            logger.debug("Could not parse '%s' as number — returning 0.0", value)
            return 0.0

    def _normalize_unit(self, raw: str) -> str:
        """
        Normalises OCR unit strings to canonical forms.

        Examples:
            "кг."  → "кг"
            "kg"   → "кг"
            "dona" → "шт"
            "уп."  → "уп"
        """
        cleaned = raw.strip().lower().rstrip(".")
        # Also strip trailing digit (e.g. "шт1" from OCR error)
        cleaned = re.sub(r"\d+$", "", cleaned).strip()
        canonical = UNIT_LOOKUP.get(cleaned)
        if canonical is None:
            logger.debug("Unknown unit '%s' — keeping as-is.", raw)
            return raw.strip()
        return canonical

    def _fuzzy_match(
        self, name: str, threshold: Optional[int] = None
    ) -> Tuple[Optional[str], float]:
        """
        Finds the closest product name in the database using RapidFuzz.

        Uses token_sort_ratio scorer which handles word-order differences:
            "Молоко 3.5% Parmalat" vs "Parmalat Молоко 3.5%" → high score.

        Returns:
            (matched_name, score) — score is 0 if no match above threshold.
        """
        if not self.product_names or not name:
            return None, 0.0

        cutoff = threshold if threshold is not None else self.fuzzy_threshold

        result = process.extractOne(
            name,
            self.product_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=cutoff,
        )

        if result:
            matched, score, _ = result
            logger.debug("Fuzzy match: '%s' → '%s' (score=%d)", name, matched, score)
            return matched, float(score)

        logger.debug("No fuzzy match above %d for: '%s'", cutoff, name)
        return None, 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _load_product_db(self, path: str) -> None:
        """Loads the GBS Market product CSV for fuzzy matching."""
        df = pd.read_csv(path, dtype=str)
        name_col = self._find_name_column(df)

        if name_col is None:
            logger.error(
                "Product DB '%s' has no recognisable name column. "
                "Expected one of: name, наименование, nom, название, товар. "
                "Fuzzy matching disabled.",
                path
            )
            return

        self.product_db    = df.to_dict("records")
        self.product_names = df[name_col].dropna().str.strip().tolist()
        logger.info(
            "Loaded product DB '%s' — %d products for fuzzy matching.",
            path, len(self.product_names)
        )

    @staticmethod
    def _find_name_column(df: pd.DataFrame) -> Optional[str]:
        candidates = ["name", "наименование", "nom", "название", "товар", "nomi"]
        for col in df.columns:
            if col.strip().lower() in candidates:
                return col
        return None

    @staticmethod
    def _clean_str(value) -> Optional[str]:
        """Returns stripped string or None if empty/NaN."""
        if pd.isna(value):
            return None
        s = str(value).strip()
        return s if s and s.lower() not in ("nan", "none", "") else None

    @staticmethod
    def _drop_flag_cols_copy(df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of df WITH the latin flag columns (don't drop yet — needed for _was_latin_cell)."""
        return df.copy()

    @staticmethod
    def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows where all non-flag cells are empty."""
        data_cols = [c for c in df.columns if not str(c).startswith("_")]
        return df.dropna(subset=data_cols, how="all").reset_index(drop=True)

    def _was_latin_cell(self, row: pd.Series, df: pd.DataFrame) -> bool:
        """
        Checks if the 'name' column of this row was flagged as Latin-dominant
        by the TableExtractor.
        """
        # The flag column is named "_latin_<original_column_name>".
        # After renaming, the original col name is lost, so we check all flags.
        flag_cols = [c for c in row.index if str(c).startswith("_latin_")]
        return any(bool(row.get(fc, False)) for fc in flag_cols)