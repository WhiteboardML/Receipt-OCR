"""
extractor.py — PaddleOCR Table Extraction Wrapper.

Supports:
1. PaddleOCR-VL (PRIMARY / RECOMMENDED):
   PaddlePaddle's newest Vision-Language Model (PaddleOCR-VL-1.6 / 0.9B).
   Powered by a dynamic-resolution Vision Transformer (NaViT ViT) visual encoder
   and lightweight language decoder. Parses complex receipts and table regions
   end-to-end into structured HTML / Markdown tables with superior accuracy.

2. PP-StructureV3 (FALLBACK / LEGACY):
   Classic multi-module layout + SLANet table structure recognition.

Mixed-language strategy for Russian + Uzbek Cyrillic + Uzbek Latin + English:
─────────────────────────────────────────────────────────────────────────────
All text cells are analyzed for Latin dominance (>60% ASCII Latin characters).
Identified cells are flagged with `_latin_<colname>_<idx> = True` so that the
downstream DataValidator can apply adaptive fuzzy matching.
"""

import logging
import re
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    OCR_ENGINE,
    PRIMARY_LANG,
    SECONDARY_LANG,
    VL_BACKEND,
    VL_PIPELINE_VERSION,
    VL_SERVER_URL,
    VL_MODEL_NAME,
    LATIN_CHAR_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (pure functions, no API dependency)
# ---------------------------------------------------------------------------

def _is_latin_dominant(text: str, threshold: float = 0.6) -> bool:
    """
    Returns True if more than `threshold` fraction of alphabetic characters
    in `text` are ASCII Latin (ord < 128).

    Examples:
        "Olma sharbati"  → True   (pure Latin)
        "Скраб 200ml"   → False  (mostly Cyrillic; 'ml' is minority)
        "97 700"         → False  (no alphabetic chars → False)
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    latin_count = sum(1 for c in alpha if ord(c) < 128)
    return (latin_count / len(alpha)) > threshold


def _ensure_3channel(img: np.ndarray) -> np.ndarray:
    """Converts grayscale (2D) to 3-channel BGR. Passes 3-channel through unchanged."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


def _markdown_to_df(md_str: str) -> Optional[pd.DataFrame]:
    """
    Parses a markdown table string into a pandas DataFrame.
    Handles standard markdown tables with `|` delimiters and divider rows.
    """
    lines = [line.strip() for line in md_str.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    if len(table_lines) < 2:
        return None

    rows = []
    for line in table_lines:
        # Strip outer pipes and split by cell delimiter
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Skip divider row (e.g. |---|:---|---:|)
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            continue
        rows.append(cells)

    if len(rows) < 2:
        return None

    headers = rows[0]
    data = []
    for r in rows[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[:len(headers)]
        data.append(r)

    df = pd.DataFrame(data, columns=headers)
    return df


def _html_table_to_df(table_str: str) -> Optional[pd.DataFrame]:
    """
    Lightweight fallback parser for basic HTML tables using regular expressions.
    Used when pandas.read_html fails or optional parsers (lxml/beautifulsoup4) are missing.
    """
    tr_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", table_str, re.DOTALL | re.IGNORECASE)
    if not tr_matches:
        return None

    rows = []
    for tr in tr_matches:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cleaned:
            rows.append(cleaned)

    if not rows:
        return None

    if len(rows) == 1:
        return pd.DataFrame(columns=rows[0])

    headers = rows[0]
    data = []
    for r in rows[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[:len(headers)]
        data.append(r)

    return pd.DataFrame(data, columns=headers)


def _parse_table_string(table_str: str) -> Optional[pd.DataFrame]:
    """
    Converts either an HTML table string or a Markdown table string into a DataFrame.
    Promotes row 0 to header if default integer column names are returned.
    """
    table_str = table_str.strip()
    if not table_str:
        return None

    # Case 1: HTML table
    if "<table" in table_str.lower():
        try:
            parsed = pd.read_html(StringIO(table_str))
            if parsed:
                df = parsed[0]
                if all(isinstance(c, int) for c in df.columns):
                    df.columns = df.iloc[0]
                    df = df.drop(index=0).reset_index(drop=True)
                return df
        except Exception as exc:
            logger.debug("HTML table parsing via read_html failed: %s", exc)

        # Fallback to pure-Python regex parser if read_html fails (e.g. missing lxml/bs4)
        df = _html_table_to_df(table_str)
        if df is not None and not df.empty:
            return df

    # Case 2: Markdown table
    if "|" in table_str:
        df = _markdown_to_df(table_str)
        if df is not None and not df.empty:
            return df

    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TableExtractor:
    """
    Extracts structured tables from receipt images.

    Defaults to PaddleOCR-VL (ViT-based Vision-Language Model) for high-accuracy
    end-to-end receipt table extraction, with PP-StructureV3 available as fallback.

    Example:
        extractor = TableExtractor(engine_type="vl")
        df = extractor.extract(preprocessed_image_array)
    """

    def __init__(
        self,
        engine_type:      str           = OCR_ENGINE,
        pipeline_version: str           = VL_PIPELINE_VERSION,
        vl_backend:       str           = VL_BACKEND,
        vl_server_url:    str           = VL_SERVER_URL,
        vl_model_name:    Optional[str] = VL_MODEL_NAME,
        primary_lang:     str           = PRIMARY_LANG,
        secondary_lang:   str           = SECONDARY_LANG,
        latin_threshold:  float         = LATIN_CHAR_RATIO_THRESHOLD,
        use_gpu:          bool          = False,
    ):
        """
        Args:
            engine_type:      'vl' (PaddleOCR-VL ViT) or 'structure' (PP-StructureV3).
            pipeline_version: Version of PaddleOCR-VL ('v1.6', 'v1.5', 'v1').
            vl_backend:       Backend for PaddleOCR-VL ('native', 'mlx-vlm-server', etc.).
            vl_server_url:    Server URL for remote/MLX backend.
            vl_model_name:    Model name or directory for server backend.
            primary_lang:     Language code for PP-Structure pass ('ru').
            secondary_lang:   Secondary language code ('en').
            latin_threshold:  Ratio of Latin characters to flag cell as Latin-dominant.
            use_gpu:          Whether to enable GPU acceleration.
        """
        self.engine_type      = engine_type.lower()
        self.pipeline_version = pipeline_version
        self.vl_backend       = vl_backend
        self.vl_server_url    = vl_server_url
        self.vl_model_name    = vl_model_name
        self.primary_lang     = primary_lang
        self.secondary_lang   = secondary_lang
        self.latin_threshold  = latin_threshold
        self.use_gpu          = use_gpu

        # Engines are initialised lazily
        self._vl_engine:        Optional[object] = None
        self._primary_engine:   Optional[object] = None
        self._secondary_engine: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> pd.DataFrame:
        """
        Runs table extraction on the given image array.
        Returns a consolidated pandas DataFrame with flagged Latin cells.
        """
        image = _ensure_3channel(image)

        if self.engine_type == "vl":
            df = self._extract_with_vl(image)
            if df.empty:
                logger.warning("PaddleOCR-VL returned no tables. Attempting PP-StructureV3 fallback...")
                df = self._extract_with_structure(image)
        else:
            df = self._extract_with_structure(image)

        if df.empty:
            logger.warning(
                "Table extraction produced no tabular data. "
                "Try disabling perspective correction (--no-perspective) "
                "or inspecting preprocessed image with --debug."
            )
            return pd.DataFrame()

        df = self._flag_latin_cells(df)
        logger.info(
            "Extracted DataFrame — shape: %s, columns: %s",
            df.shape, list(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Extraction implementations
    # ------------------------------------------------------------------

    def _extract_with_vl(self, image: np.ndarray) -> pd.DataFrame:
        """Runs the PaddleOCR-VL Vision Transformer pipeline."""
        engine = self._get_vl_engine()
        logger.info(
            "Running PaddleOCR-VL (ViT %s, backend=%s) on image shape %s ...",
            self.pipeline_version, self.vl_backend, image.shape,
        )

        results = engine.predict(image)
        tables = self._parse_vl_results(results)

        if not tables:
            return pd.DataFrame()

        logger.info("PaddleOCR-VL found %d table(s). Concatenating ...", len(tables))
        return pd.concat(tables, ignore_index=True)

    def _extract_with_structure(self, image: np.ndarray) -> pd.DataFrame:
        """Runs the classic PP-StructureV3 pipeline."""
        engine = self._get_primary_engine()
        logger.info(
            "Running PP-StructureV3 (lang=%s) on image shape %s ...",
            self.primary_lang, image.shape,
        )

        results = engine.predict(image)
        tables = self._parse_pp_structure_results(results)

        if not tables:
            return pd.DataFrame()

        logger.info("PP-StructureV3 found %d table(s). Concatenating ...", len(tables))
        return pd.concat(tables, ignore_index=True)

    # ------------------------------------------------------------------
    # Engine Initializers (Lazy)
    # ------------------------------------------------------------------

    def _ensure_mlx_server_running(self) -> None:
        """
        Ensures the local mlx-vlm server is running for Apple Silicon GPU acceleration.
        If not reachable, starts it automatically as a background process.
        """
        import subprocess
        import sys
        import time
        import urllib.request
        from pathlib import Path
        from urllib.parse import urlparse

        parsed = urlparse(self.vl_server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8111
        health_url = f"http://{host}:{port}/health"

        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as response:
                if response.status == 200:
                    logger.info("MLX-VLM server is already running on port %d.", port)
                    return
        except Exception:
            pass

        model_path = self.vl_model_name or str(
            Path.home() / ".paddlex" / "official_models" / "PaddleOCR-VL-1.6"
        )
        logger.info(
            "Auto-starting local MLX-VLM server on Apple Silicon GPU (port %d, model: %s) ...",
            port, model_path,
        )
        cmd = [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            model_path,
            "--port",
            str(port),
            "--trust-remote-code",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        start_time = time.time()
        while time.time() - start_time < 15:
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(health_url, timeout=1.0) as response:
                    if response.status == 200:
                        logger.info("MLX-VLM server ready on GPU (PID %d).", proc.pid)
                        return
            except Exception:
                continue

        logger.warning(
            "MLX-VLM server did not respond within 15 seconds. "
            "Pipeline will attempt to connect anyway."
        )

    def _get_vl_engine(self):
        """Lazy-initialises the PaddleOCR-VL ViT engine."""
        if self._vl_engine is None:
            logger.info(
                "Initialising PaddleOCR-VL ViT engine (version=%s, backend=%s). "
                "First run will download PaddleOCR-VL weights (~0.9B ViT + LLM). Please wait ...",
                self.pipeline_version, self.vl_backend,
            )
            from paddleocr import PaddleOCRVL

            kwargs = {"pipeline_version": self.pipeline_version}
            if self.vl_backend and self.vl_backend != "native":
                kwargs["vl_rec_backend"] = self.vl_backend
                if self.vl_server_url:
                    kwargs["vl_rec_server_url"] = self.vl_server_url
                if self.vl_model_name:
                    kwargs["vl_rec_api_model_name"] = self.vl_model_name

            if self.vl_backend == "mlx-vlm-server":
                self._ensure_mlx_server_running()

            self._vl_engine = PaddleOCRVL(**kwargs)
        return self._vl_engine

    def _get_primary_engine(self):
        """Lazy-initialises the primary (Cyrillic) PP-StructureV3 engine."""
        if self._primary_engine is None:
            device = "gpu" if self.use_gpu else "cpu"
            logger.info(
                "Initialising PP-StructureV3 engine (lang=%s, device=%s). "
                "First run will download model files (~500MB). Please wait ...",
                self.primary_lang, device,
            )
            from paddleocr import PPStructureV3
            self._primary_engine = PPStructureV3(
                lang=self.primary_lang,
                use_table_recognition=True,
                device=device,
            )
        return self._primary_engine

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_vl_results(self, results: list) -> list[pd.DataFrame]:
        """
        Extracts tabular DataFrames from PaddleOCRVLResult objects.
        Checks table_res_list, parsing_res_list, and html/markdown properties.
        """
        dfs = []
        for result in results:
            # 1. Inspect table_res_list (HTML table predictions)
            table_res_list = []
            if hasattr(result, "get"):
                table_res_list = result.get("table_res_list", [])
            elif hasattr(result, "table_res_list"):
                table_res_list = getattr(result, "table_res_list", [])

            for table_entry in table_res_list:
                html = ""
                if hasattr(table_entry, "html") and isinstance(table_entry.html, dict):
                    html = table_entry.html.get("pred", "")
                elif isinstance(table_entry, dict):
                    html = table_entry.get("pred_html", "") or table_entry.get("html", "")

                if html:
                    df = _parse_table_string(html)
                    if df is not None and not df.empty:
                        dfs.append(df)

            # 2. Inspect parsing_res_list for blocks labeled 'table'
            parsing_res_list = []
            if hasattr(result, "get"):
                parsing_res_list = result.get("parsing_res_list", [])
            elif hasattr(result, "parsing_res_list"):
                parsing_res_list = getattr(result, "parsing_res_list", [])

            for block in parsing_res_list:
                label = getattr(block, "label", None) or (
                    block.get("block_label") if isinstance(block, dict) else ""
                )
                content = getattr(block, "content", None) or (
                    block.get("block_content") if isinstance(block, dict) else ""
                )
                if label and str(label).lower() in ("table", "table_recognition") and content:
                    df = _parse_table_string(content)
                    if df is not None and not df.empty:
                        dfs.append(df)

            # 3. Inspect result.html property
            if not dfs and hasattr(result, "html"):
                try:
                    html_dict = result.html
                    if isinstance(html_dict, dict):
                        for html in html_dict.values():
                            if html:
                                df = _parse_table_string(html)
                                if df is not None and not df.empty:
                                    dfs.append(df)
                except Exception as exc:
                    logger.debug("Extracting result.html failed: %s", exc)

            # 4. Fallback: inspect full-page markdown for tables
            if not dfs and hasattr(result, "markdown"):
                try:
                    md_text = result.markdown
                    if isinstance(md_text, str) and "|" in md_text:
                        df = _parse_table_string(md_text)
                        if df is not None and not df.empty:
                            dfs.append(df)
                except Exception as exc:
                    logger.debug("Extracting result.markdown failed: %s", exc)

        return dfs

    def _parse_pp_structure_results(self, results: list) -> list[pd.DataFrame]:
        """Extracts DataFrames from PP-StructureV3 predict() results."""
        dfs = []
        for result in results:
            json_data = result.json if hasattr(result, "json") else dict(result)
            res = json_data.get("res", {})
            table_res_list = res.get("table_res_list", [])

            for table_entry in table_res_list:
                html = table_entry.get("pred_html", "")
                if not html:
                    continue

                df = _parse_table_string(html)
                if df is not None and not df.empty:
                    dfs.append(df)
                    logger.debug("Parsed PP-Structure table HTML → shape %s", df.shape)

        return dfs

    @staticmethod
    def _is_latin_cell(text: str, threshold: float) -> bool:
        """Check if a single text value is Latin-dominant. Handles NaN/None safely."""
        if text is None:
            return False
        if isinstance(text, float) and pd.isna(text):
            return False
        return _is_latin_dominant(str(text), threshold)

    def _flag_latin_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds helper boolean columns `_latin_<colname>_<idx>` for text columns
        to flag Latin-dominant cells for the validator.
        """
        for col_idx in range(df.shape[1]):
            col_name = str(df.columns[col_idx])
            if col_name.startswith("_"):
                continue
            if not (df.iloc[:, col_idx].dtype == "object" or pd.api.types.is_string_dtype(df.iloc[:, col_idx])):
                continue

            flag_col = f"_latin_{col_name}_{col_idx}"
            df[flag_col] = [
                self._is_latin_cell(v, self.latin_threshold)
                for v in df.iloc[:, col_idx]
            ]
            latin_count = int(df[flag_col].sum())
            if latin_count:
                logger.info(
                    "Column '%s': %d/%d cells flagged as Latin-dominant.",
                    col_name, latin_count, len(df),
                )
        return df
