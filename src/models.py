"""
models.py — Data structures for the receipt OCR pipeline.

Using @dataclass keeps field definitions clean and gives us __repr__
and __eq__ for free. The Product.to_gbs_dict() method is the single
place responsible for the internal→GBS Market column name translation.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Product:
    """
    A single validated line-item from a receipt.

    Fields split deliberately:
      - quantity     : the numeric amount  (e.g. 2.5)
      - unit_measurement : the unit label (e.g. "кг", "шт", "уп")

    This is because your receipts have them in SEPARATE columns.
    A combined "2.5 кг" string is harder to parse and sort/filter
    later in GBS Market.
    """

    # Core product fields
    name:             str
    quantity:         float
    unit_measurement: str
    unit_price:       float
    total_price:      float

    # Optional fields — may not be present on every receipt
    sku:      Optional[str] = None
    category: Optional[str] = None

    # ---------------------------------------------------------------
    # Metadata — used internally, NOT exported to GBS Market
    # ---------------------------------------------------------------
    # raw_name: what the OCR engine literally read before fuzzy matching
    raw_name: str = field(default="", repr=False)

    # matched_name: best match found in product database (may be None if
    # no match above threshold, or if no database was provided)
    matched_name: Optional[str] = field(default=None, repr=False)

    # match_score: 0–100 rapidfuzz confidence for the name match
    match_score: float = field(default=0.0, repr=False)

    # confidence: human-readable summary of match quality
    #   "high"   → score ≥ 90  — safe to auto-import
    #   "medium" → score ≥ 75  — import with caution
    #   "low"    → score < 75  — flagged for manual review
    confidence: str = field(default="low", repr=False)

    # is_latin_cell: True if the name cell was detected as Latin-dominant.
    # These cells had a second-pass attempt with the English OCR model.
    is_latin_cell: bool = field(default=False, repr=False)

    def to_gbs_dict(self, column_map: dict) -> dict:
        """
        Converts this Product to a flat dict using GBS Market column headers.
        Uses matched_name if available (higher accuracy), else raw OCR name.
        """
        internal = {
            "sku":              self.sku or "",
            "name":             self.matched_name or self.name,
            "category":         self.category or "",
            "quantity":         self.quantity,
            "unit_measurement": self.unit_measurement,
            "unit_price":       self.unit_price,
            "total_price":      self.total_price,
        }
        return {column_map[k]: v for k, v in internal.items()}


@dataclass
class OCRResult:
    """
    Wraps the raw output from PaddleOCR before any validation.
    Useful for debugging: save this to see exactly what the OCR engine produced.
    """
    raw_dataframe:     object           # pd.DataFrame
    source_image_path: str
    timestamp:         str  = field(default_factory=lambda: datetime.now().isoformat())
    language_used:     str  = "ru"
    tables_found:      int  = 0
    had_fallback:      bool = False     # True if secondary (en) engine was used