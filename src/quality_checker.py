"""
quality_checker.py — Computer Vision Image Quality Gate.

Evaluates receipt photo quality BEFORE feeding into heavy OCR models:
  1. Blur Detection (Variance of Laplacian)
  2. Glare / Specular Highlight Detection (HSV saturation & value)
  3. Skew / Tilt Angle Estimation (Hough line transform)

Why this matters:
  Feeding blurry or severely glare-obscured photos into a 0.9B Vision-Language
  Transformer wastes compute and leads to hallucinatory/corrupted tables.
  Detecting capture flaws upfront gives users actionable feedback (e.g. "re-focus camera").
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Diagnostic report evaluating capture quality of an image."""
    is_acceptable: bool
    blur_score: float         # Higher = sharper (variance of Laplacian)
    is_blurry: bool
    glare_ratio: float        # Percentage of pixels with specular glare (0.0 to 1.0)
    has_glare: bool
    skew_angle: float         # Estimated tilt angle in degrees (-45.0 to +45.0)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_acceptable else "FLAGGED"
        msg = f"Quality Gate [{status}] — Blur: {self.blur_score:.1f} (blurry={self.is_blurry}), " \
              f"Glare: {self.glare_ratio * 100:.2f}% (glare={self.has_glare}), " \
              f"Skew: {self.skew_angle:.1f}°"
        if self.warnings:
            msg += f" | Warnings: {'; '.join(self.warnings)}"
        return msg


class ImageQualityChecker:
    """
    Evaluates physical capture quality of receipt photographs.

    Tunable thresholds:
      - blur_threshold: Below this variance of Laplacian, image is marked blurry (default: 100.0)
      - glare_threshold: Max allowed ratio of overexposed hotspot pixels (default: 0.08 / 8%)
      - max_skew_angle: Skew beyond this angle triggers a warning (default: 15.0°)
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        glare_threshold: float = 0.08,
        max_skew_angle: float = 15.0,
    ):
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold
        self.max_skew_angle = max_skew_angle

    def evaluate(self, image: np.ndarray) -> QualityReport:
        """
        Runs comprehensive image quality analysis on an image array (BGR or grayscale).
        Returns a QualityReport.
        """
        if image is None or image.size == 0:
            return QualityReport(
                is_acceptable=False,
                blur_score=0.0,
                is_blurry=True,
                glare_ratio=0.0,
                has_glare=False,
                skew_angle=0.0,
                warnings=["Empty or invalid image array"],
            )

        # Convert to grayscale for blur and skew checks
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            bgr = image
        else:
            gray = image
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 1. Blur Detection (Variance of Laplacian)
        blur_score = self.calculate_blur_score(gray)
        is_blurry = blur_score < self.blur_threshold

        # 2. Glare Detection (Specular highlights)
        glare_ratio = self.calculate_glare_ratio(bgr)
        has_glare = glare_ratio > self.glare_threshold

        # 3. Skew Angle Estimation
        skew_angle = self.estimate_skew_angle(gray)
        is_skewed = abs(skew_angle) > self.max_skew_angle

        warnings = []
        if is_blurry:
            warnings.append(
                f"Image appears blurry (score: {blur_score:.1f} < {self.blur_threshold:.1f}). "
                "Consider re-taking photo with better focus."
            )
        if has_glare:
            warnings.append(
                f"Severe specular glare detected ({glare_ratio * 100:.1f}% of pixels overexposed). "
                "Try angling camera to avoid direct flash/lamp reflection."
            )
        if is_skewed:
            warnings.append(
                f"High document tilt detected ({skew_angle:.1f}°). "
                "Perspective correction will attempt to align it."
            )

        is_acceptable = not (is_blurry or (has_glare and glare_ratio > 0.20))

        report = QualityReport(
            is_acceptable=is_acceptable,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            glare_ratio=round(glare_ratio, 4),
            has_glare=has_glare,
            skew_angle=round(skew_angle, 2),
            warnings=warnings,
        )

        logger.info(report.summary())
        return report

    @staticmethod
    def calculate_blur_score(gray_img: np.ndarray) -> float:
        """
        Calculates sharpness using variance of Laplacian filter.
        Sharp edges produce high variance; blurry images produce low variance.
        """
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    @staticmethod
    def calculate_glare_ratio(bgr_img: np.ndarray) -> float:
        """
        Calculates ratio of overexposed highlight pixels.
        Pixels with Value >= 250 and low Saturation (< 30) represent glare.
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # White glare: high brightness, low saturation
        glare_mask = (val >= 250) & (sat < 35)
        glare_count = np.count_nonzero(glare_mask)
        total_pixels = bgr_img.shape[0] * bgr_img.shape[1]
        return float(glare_count / total_pixels) if total_pixels > 0 else 0.0

    @staticmethod
    def estimate_skew_angle(gray_img: np.ndarray) -> float:
        """
        Estimates receipt text skew angle in degrees using Hough line transform.
        Returns angle in degrees (-45.0 to +45.0). Positive = rotated clockwise.
        """
        # Downscale for fast edge detection
        h, w = gray_img.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small = cv2.resize(gray_img, (int(w * scale), int(h * scale)))
        else:
            small = gray_img

        edges = cv2.Canny(small, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle_deg = math.degrees(math.atan2(dy, dx))
            # Focus on near-horizontal text lines (-45 to +45 degrees)
            if -45.0 <= angle_deg <= 45.0:
                angles.append(angle_deg)

        if not angles:
            return 0.0

        # Return median angle to reject outlier lines
        return float(np.median(angles))
