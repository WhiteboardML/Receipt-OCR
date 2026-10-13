"""
preprocessor.py — OpenCV-based image cleaning before OCR.

Pipeline order (each step is isolated so you can disable any one of them):
    1. Load image
    2. Perspective correction  → flattens photos taken at an angle
    3. Blue mark removal       → erases handwritten blue checkmarks via HSV mask
    4. Binarization            → stark black-on-white, removes shadows

Why this order?
    Perspective first: a straight image gives better contour detection later.
    Blue removal before binarization: after binarization blue becomes black,
    indistinguishable from printed text.
    Binarization last: gives OCR the cleanest possible input.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

from .quality_checker import ImageQualityChecker, QualityReport

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Cleans a receipt photo so PaddleOCR gets the best possible input.

    Example usage:
        prep = ImagePreprocessor()
        clean = prep.process("receipts/photo.jpg")
        prep.save(clean, "output/debug_clean.png")   # optional debug save
    """

    def __init__(
        self,
        blue_lower:        tuple = (100, 50,  50),
        blue_upper:        tuple = (130, 255, 255),
        block_size:        int   = 31,
        adaptive_c:        int   = 10,
        auto_perspective:  bool  = True,
        remove_blue_marks: bool  = True,
        check_quality:     bool  = True,
    ):
        """
        Args:
            blue_lower:        Lower HSV bound for blue checkmark colour.
            blue_upper:        Upper HSV bound for blue checkmark colour.
            block_size:        Neighbourhood size for adaptive threshold (must be odd).
            adaptive_c:        Constant subtracted from threshold mean.
            auto_perspective:  Whether to auto-detect and flatten document angle.
            remove_blue_marks: Whether to erase blue handwritten marks.
            check_quality:     Whether to evaluate blur, glare, and skew before processing.
        """
        self.blue_lower        = np.array(blue_lower,  dtype=np.uint8)
        self.blue_upper        = np.array(blue_upper,  dtype=np.uint8)
        self.block_size        = block_size
        self.adaptive_c        = adaptive_c
        self.auto_perspective  = auto_perspective
        self.remove_blue_marks = remove_blue_marks
        self.check_quality     = check_quality

        self.quality_checker   = ImageQualityChecker()
        self.last_quality_report: Optional[QualityReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image_path: str) -> np.ndarray:
        """
        Runs the full preprocessing pipeline.

        Returns:
            A 3-channel BGR numpy array (white background, black text).
            Always 3-channel so PaddleOCR / PP-Structure is happy.
        """
        img = self._load(image_path)
        logger.info("Loaded image %s — shape %s", image_path, img.shape)

        if self.check_quality:
            self.last_quality_report = self.quality_checker.evaluate(img)

        if self.auto_perspective:
            img = self._correct_perspective(img)

        if self.remove_blue_marks:
            img = self._remove_blue_marks(img)

        img = self._binarize(img)         # returns 3-channel BGR

        logger.info("Preprocessed image final shape: %s", img.shape)
        return img

    def save(self, image: np.ndarray, output_path: str) -> None:
        """Save a processed image to disk (useful for debug inspection)."""
        cv2.imwrite(output_path, image)
        logger.info("Saved preprocessed image → %s", output_path)

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(
                f"cv2.imread could not open '{path}'. "
                f"Check the path and that the file is a valid image."
            )
        return img

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Detects the document's 4-corner boundary and warps it flat.

        How it works:
            1. Canny edge detection finds sharp edges.
            2. We take the 5 largest contours and look for one with 4 corners.
            3. getPerspectiveTransform + warpPerspective flattens it.

        Falls back gracefully (returns original) if no rectangle found.
        """
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged   = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Sort by area, largest first — the document is (hopefully) the biggest thing
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        doc_contour = None
        for c in contours[:5]:
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is None:
            logger.warning(
                "Perspective correction: no 4-corner document boundary found. "
                "Returning original image. Try photographing on a contrasting surface."
            )
            return img

        logger.info("Perspective correction: boundary found, applying warp.")
        return self._four_point_transform(img, doc_contour.reshape(4, 2))

    def _four_point_transform(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Computes and applies the perspective warp to flatten the document."""
        rect = self._order_points(pts)
        tl, tr, br, bl = rect

        width_a   = np.linalg.norm(br - bl)
        width_b   = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a   = np.linalg.norm(tr - br)
        height_b   = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0,           0           ],
            [max_width-1, 0           ],
            [max_width-1, max_height-1],
            [0,           max_height-1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
        return cv2.warpPerspective(img, M, (max_width, max_height))

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 corner points as: top-left, top-right, bottom-right, bottom-left.
        Required so getPerspectiveTransform maps to the correct output corners.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]    # top-left     has the smallest  x+y
        rect[2] = pts[np.argmax(s)]    # bottom-right has the largest   x+y
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # top-right    has the smallest  y-x
        rect[3] = pts[np.argmax(diff)] # bottom-left  has the largest   y-x
        return rect

    def _remove_blue_marks(self, img: np.ndarray) -> np.ndarray:
        """
        Creates an HSV mask that targets blue handwritten checkmarks and
        replaces them with white pixels so they don't interfere with OCR.

        If your marks are a slightly different shade of blue, adjust
        BLUE_HSV_LOWER / BLUE_HSV_UPPER in config/settings.py.
        Tip: use the `--debug-hsv` flag in main.py to inspect HSV values.
        """
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Dilate slightly to catch anti-aliased edges of the pen strokes
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=1)

        img[mask > 0] = [255, 255, 255]   # overwrite blue pixels with white
        blue_pixel_count = int(np.sum(mask > 0))
        logger.info("Blue mark removal: erased %d pixels.", blue_pixel_count)
        return img

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Converts the colour image to stark black-on-white binary.

        Why adaptiveThreshold instead of a global threshold?
            Receipt photos taken by phone have uneven lighting — brighter
            near the top, darker in shadowed corners. A global threshold
            would leave the dark areas as grey smudge.
            adaptiveThreshold computes the threshold *locally* for each
            small neighbourhood, handling shadows automatically.

        Returns a 3-channel BGR image (white background, black text)
        so PaddleOCR receives the format it expects.
        """
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.block_size,
            C=self.adaptive_c,
        )
        # Convert back to 3-channel BGR — PaddleOCR expects it
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)