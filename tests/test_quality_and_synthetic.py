"""
test_quality_and_synthetic.py — Unit tests for Quality Gate and Synthetic Generator.

Tests run purely with synthetic numpy/Pillow arrays and mock inputs,
requiring zero external OCR downloads or GPUs.
"""

import numpy as np
import pytest

from src.quality_checker import ImageQualityChecker, QualityReport
from tools.synthesize import SyntheticReceiptGenerator


class TestImageQualityChecker:
    """Tests the computer vision pre-check heuristics."""

    def setup_method(self):
        self.checker = ImageQualityChecker(blur_threshold=100.0, glare_threshold=0.08)

    def test_sharp_image_passes_blur_check(self):
        # Create an artificial sharp checkerboard pattern
        sharp = np.zeros((200, 200), dtype=np.uint8)
        sharp[::20, :] = 255
        sharp[:, ::20] = 255
        score = self.checker.calculate_blur_score(sharp)
        assert score > 100.0

    def test_flat_uniform_image_is_blurry(self):
        # Perfectly flat uniform image has zero Laplacian variance
        flat = np.ones((200, 200), dtype=np.uint8) * 128
        score = self.checker.calculate_blur_score(flat)
        assert score == 0.0
        report = self.checker.evaluate(flat)
        assert report.is_blurry is True

    def test_glare_detection_on_overexposed_white(self):
        # Image that is 50% pure saturated white (glare)
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:50, :, :] = 255  # Top half pure white (V=255, S=0)
        ratio = self.checker.calculate_glare_ratio(bgr)
        assert ratio == 0.5
        assert ratio > self.checker.glare_threshold

    def test_empty_image_fails_gracefully(self):
        empty = np.array([])
        report = self.checker.evaluate(empty)
        assert report.is_acceptable is False
        assert len(report.warnings) > 0


class TestSyntheticReceiptGenerator:
    """Tests synthetic receipt and degradation generation."""

    def setup_method(self):
        self.gen = SyntheticReceiptGenerator()

    def test_generate_produces_image_and_metadata(self):
        img, gt = self.gen.generate(num_items=3)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert gt["total_items"] == 3
        assert len(gt["products"]) == 3

    def test_ground_truth_math_is_exact(self):
        img, gt = self.gen.generate(num_items=4)
        calculated_sum = 0.0
        for p in gt["products"]:
            # Check line math: qty * price == total
            assert p["quantity"] * p["price"] == p["total"]
            calculated_sum += p["total"]
        # Check grand total
        assert gt["total_sum"] == calculated_sum

    def test_noise_injection_modes(self):
        img_clean, _ = self.gen.generate(num_items=2, noise_types=[])
        img_noisy, gt_noisy = self.gen.generate(
            num_items=2,
            noise_types=["shadow", "pen_marks", "blur", "noise"]
        )
        assert img_noisy.shape == img_clean.shape
        assert "shadow" in gt_noisy["noise_applied"]
        assert "pen_marks" in gt_noisy["noise_applied"]
